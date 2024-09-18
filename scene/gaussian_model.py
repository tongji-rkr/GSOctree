#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import time
from datetime import timedelta
import torch
from functools import reduce
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from einops import repeat
import math
from torch_scatter import scatter_max

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 sh_degree: int=0, 
                 fork: int=2,
                 visible_threshold: float = -1,
                 dist2level: str = 'round',
                 base_layer: int = 10,
                 progressive: bool = True,
                 extend: float = 1.1
                 ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.fork = fork
        self.visible_threshold = visible_threshold
        self.dist2level = dist2level
        self.base_layer = base_layer
        self.progressive = progressive
        self.extend = extend

        self.clone_gaussians = 0
        self.split_gaussians = 0

        self._xyz = torch.empty(0)
        self._level = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._level,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._level,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_level(self):
        return self._level
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_extra_level(self):
        return self._extra_level

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def set_coarse_interval(self, coarse_iter, coarse_factor):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/coarse_factor
            a1 = coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, dist_ratio)
                dist_min = torch.quantile(dist, 1 - dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - dist_ratio)
        self.standard_dist = dist_max
        if levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        else:
            self.levels = levels
        if init_level == -1:
            self.init_level = int(self.levels/2)
        else:
            self.init_level = init_level

    def octree_sample(self, xyzs, rgbs, init_pos):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self.colors = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda() 
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_candidates = torch.round((xyzs - init_pos) / cur_size)
            new_candidates_unique, inverse_indices = torch.unique(new_candidates, return_inverse=True, dim=0)
            new_positions = new_candidates_unique * cur_size + init_pos + (cur_size/2)
            new_levels = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            new_colors = scatter_max(rgbs, inverse_indices.unsqueeze(1).expand(-1, rgbs.size(1)), dim=0)[0]
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self.colors = torch.concat((self.colors, new_colors), dim=0)
            self._level = torch.concat((self._level, new_levels), dim=0)
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def create_from_pcd(self, pcd, spatial_lr_scale, logger=None):
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(pcd.points, dtype=torch.float, device="cuda")
        colors = torch.tensor(pcd.colors, dtype=torch.float, device="cuda")
        box_min = torch.min(points)*self.extend
        box_max = torch.max(points)*self.extend
        box_d = box_max - box_min
        print(box_d)
        if self.base_layer < 0:
            default_voxel_size = 0.02
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        print(self.voxel_size)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
        self.octree_sample(points, colors, self.init_pos)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, weed_mask = self.weed_out(self.positions, self._level)
        self.colors = self.colors[weed_mask]

        print(f'Branches of Tree: {self.fork}')
        print(f'Base Layer of Tree: {self.base_layer}')
        print(f'Visible Threshold: {self.visible_threshold}')
        print(f'LOD Levels: {self.levels}')
        print(f'Initial Levels: {self.init_level}')
        print(f'Initial Voxel Number: {self.positions.shape[0]}')
        print(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        print(f'Max Voxel Size: {self.voxel_size}')
        logger.info(f'Branches of Tree: {self.fork}')
        logger.info(f'Base Layer of Tree: {self.base_layer}')
        logger.info(f'Visible Threshold: {self.visible_threshold}')
        logger.info(f'LOD Levels: {self.levels}')
        logger.info(f'Initial Levels: {self.init_level}')
        logger.info(f'Initial Voxel Number: {self.positions.shape[0]}')
        logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        logger.info(f'Max Voxel Size: {self.voxel_size}')

        fused_point_cloud = self.positions
        fused_color = RGB2SH(self.colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        dist2 = torch.clamp_min(distCUDA2(self.positions), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._level = self._level.unsqueeze(dim=1)
        self._extra_level = torch.zeros(self._xyz.shape[0], dtype=torch.float, device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    def weed_out(self, gaussian_positions, gaussian_levels):
        visible_count = torch.zeros(gaussian_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((gaussian_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (gaussian_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return gaussian_positions[weed_mask], gaussian_levels[weed_mask], mean_visible, weed_mask

    def set_gaussian_mask(self, cam_center, iteration, resolution_scale, rendering=False):
        gaussian_pos = self._xyz
        dist = torch.sqrt(torch.sum((gaussian_pos - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        
        if self.progressive and not rendering:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._gaussian_mask = (self._level.squeeze(dim=1) <= int_level)    

    def set_gaussian_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        gaussian_pos = self._xyz
        dist = torch.sqrt(torch.sum((gaussian_pos - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._gaussian_mask = (self._level.squeeze(dim=1) <= int_level)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def extent_levels(self, extent):
        level0_mask = (self._level == 0).squeeze(dim=1)
        base_scale = torch.mean(self.get_scaling[level0_mask]).item()
        new_extent = torch.ones((self.get_xyz.shape[0]), device="cuda") * base_scale
        for level in range(1, self.levels):
            level_mask = (self._level == level).squeeze(dim=1)
            new_extent[level_mask] = torch.mean(self.get_scaling[level_mask]).item()
        new_extent = new_extent * extent / base_scale
        return new_extent

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        l.append('info')
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        infos = np.zeros_like(levels, dtype=np.float32)
        infos[0, 0] = self.standard_dist

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, levels, extra_levels, infos, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        self.standard_dist = torch.tensor(plydata.elements[0]["info"][0]).float()

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._gaussian_mask = torch.ones(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        self.levels = torch.max(self._level) - torch.min(self._level) + 1
        self.active_sh_degree = self.max_sh_degree
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]
        self._extra_level = self._extra_level[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_level, new_extra_level, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._level = torch.cat([self._level, new_level], dim=0)
        self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, iteration, update_ratio, extra_ratio, extra_up):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads
        update_value = self.fork ** update_ratio
        cur_threshold = grad_threshold * (update_value ** self.get_level.squeeze(1))
        extra_threshold = cur_threshold * extra_ratio
        scale_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * self.extent_levels(scene_extent)

        selected_pts_mask = torch.where(padded_grad >= cur_threshold, True, False)
        selected_pts_extra_mask = torch.where(padded_grad >= extra_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_mask)
        selected_pts_extra_mask = torch.logical_and(selected_pts_extra_mask, scale_mask)
        if ~self.progressive or iteration > self.coarse_intervals[-1]:
            self._extra_level += extra_up * selected_pts_extra_mask.float()   

        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        new_level = torch.clamp(self._level[selected_pts_mask]+1, 0, self.levels-1)
        new_extra_level = torch.zeros(new_level.shape[0], dtype=torch.float, device="cuda")
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] * 0.6)
        new_rotation = self._rotation[selected_pts_mask]

        new_xyz, _, _, weed_mask = self.weed_out(new_xyz, new_level.squeeze(1))
        new_level = new_level[weed_mask]
        new_extra_level = new_extra_level[weed_mask]
        new_features_dc = new_features_dc[weed_mask]
        new_features_rest = new_features_rest[weed_mask]
        new_opacity = new_opacity[weed_mask]
        new_scaling = new_scaling[weed_mask]
        new_rotation = new_rotation[weed_mask]
        
        self.split_gaussians += len(new_xyz)
        #self._scaling[selected_pts_mask] = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] * 0.8)

        self.densification_postfix(new_xyz, new_level, new_extra_level, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, iteration, update_ratio, extra_ratio, extra_up):
        # Extract points that satisfy the gradient condition
        update_value = self.fork ** update_ratio
        cur_threshold = grad_threshold * (update_value ** self.get_level.squeeze(1))
        extra_threshold = cur_threshold * extra_ratio
        scale_mask = torch.max(self.get_scaling, dim=1).values <= self.percent_dense * self.extent_levels(scene_extent)
        
        selected_pts_mask = torch.where(grads >= cur_threshold, True, False)
        selected_pts_extra_mask = torch.where(grads >= extra_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, scale_mask)
        selected_pts_extra_mask = torch.logical_and(selected_pts_extra_mask, scale_mask)
        if ~self.progressive or iteration > self.coarse_intervals[-1]:
            self._extra_level += extra_up * selected_pts_extra_mask.float()   

        new_xyz = self._xyz[selected_pts_mask]
        new_level = self._level[selected_pts_mask]
        new_extra_level = torch.zeros(new_level.shape[0], dtype=torch.float, device="cuda")
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_xyz, _, _, weed_mask = self.weed_out(new_xyz, new_level.squeeze(1))
        new_level = new_level[weed_mask]
        new_extra_level = new_extra_level[weed_mask]
        new_features_dc = new_features_dc[weed_mask]
        new_features_rest = new_features_rest[weed_mask]
        new_opacity = new_opacity[weed_mask]
        new_scaling = new_scaling[weed_mask]
        new_rotation = new_rotation[weed_mask]
        
        self.clone_gaussians += len(new_xyz)
        self.densification_postfix(new_xyz, new_level, new_extra_level, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration, update_ratio, extra_ratio, extra_up):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)

        self.densify_and_clone(grads_norm, max_grad, extent, iteration, update_ratio, extra_ratio, extra_up)
        if ~self.progressive or iteration > self.coarse_intervals[-1]:
            self.densify_and_split(grads_norm, max_grad, extent, iteration, update_ratio, extra_ratio, extra_up)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, visit_mask, update_filter):
        self.xyz_gradient_accum[visit_mask] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[visit_mask] += 1

    # def replace_tensor_to_optimizer(self, tensor, name):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         if group["name"] == name:
    #             stored_state = self.optimizer.state.get(group['params'][0], None)
    #             stored_state["exp_avg"] = torch.zeros_like(tensor)
    #             stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #     return optimizable_tensors

    # def _prune_optimizer(self, mask):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:
    #             stored_state["exp_avg"] = stored_state["exp_avg"][mask]
    #             stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
    #             optimizable_tensors[group["name"]] = group["params"][0]
    #     return optimizable_tensors

    # def prune_points(self, mask):
    #     valid_points_mask = ~mask
    #     optimizable_tensors = self._prune_optimizer(valid_points_mask)

    #     self._xyz = optimizable_tensors["xyz"]

    #     self._features_dc = optimizable_tensors["f_dc"]
    #     self._features_rest = optimizable_tensors["f_rest"]
    #     self._opacity = optimizable_tensors["opacity"]
    #     self._scaling = optimizable_tensors["scaling"]
    #     self._rotation = optimizable_tensors["rotation"]
    #     self._level = self._level[valid_points_mask]    
    #     self._extra_level = self._extra_level[valid_points_mask]

    #     self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
    #     self.denom = self.denom[valid_points_mask]
    #     self.max_radii2D = self.max_radii2D[valid_points_mask]

    # def cat_tensors_to_optimizer(self, tensors_dict):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         assert len(group["params"]) == 1
    #         extension_tensor = tensors_dict[group["name"]]
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:
    #             stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
    #             stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
    #             optimizable_tensors[group["name"]] = group["params"][0]

    #     return optimizable_tensors

    # def densification_postfix(self, new_xyz, new_level, new_extra_level, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, visit_mask):
        
    #     d = {"xyz": new_xyz,
    #     "f_dc": new_features_dc,
    #     "f_rest": new_features_rest,
    #     "opacity": new_opacities,
    #     "scaling" : new_scaling,
    #     "rotation" : new_rotation}

    #     optimizable_tensors = self.cat_tensors_to_optimizer(d)
    #     self._xyz = optimizable_tensors["xyz"]
    #     self._features_dc = optimizable_tensors["f_dc"]
    #     self._features_rest = optimizable_tensors["f_rest"]
    #     self._opacity = optimizable_tensors["opacity"]
    #     self._scaling = optimizable_tensors["scaling"]
    #     self._rotation = optimizable_tensors["rotation"]
    #     self._level = torch.cat([self._level, new_level], dim=0)
    #     self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)

    #     self.xyz_gradient_accum[visit_mask] = 0.0
    #     padding_xyz_gradient_accum = torch.zeros([self.get_xyz.shape[0] - self.xyz_gradient_accum.shape[0], 1],
    #                                        dtype=torch.float32, 
    #                                        device=self.xyz_gradient_accum.device)
    #     self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, padding_xyz_gradient_accum], dim=0)

    #     self.denom[visit_mask] = 0.0
    #     padding_denom = torch.zeros([self.get_xyz.shape[0] - self.denom.shape[0], 1],
    #                                        dtype=torch.float32, 
    #                                        device=self.denom.device)
    #     self.denom = torch.cat([self.denom, padding_denom], dim=0)

    #     self.max_radii2D[visit_mask] = 0.0
    #     padding_max_radii2D = torch.zeros([self.get_xyz.shape[0] - self.max_radii2D.shape[0]],
    #                                        dtype=torch.float32, 
    #                                        device=self.max_radii2D.device)
    #     self.max_radii2D = torch.cat([self.max_radii2D, padding_max_radii2D], dim=0)
    
    # def generate_random_points_around(self, position, radius, num_points):
    #     # Generate random points in a sphere around the position
    #     theta = 2 * torch.pi * torch.rand(num_points).cuda()
    #     phi = torch.acos(2 * torch.rand(num_points) - 1).cuda()
    #     x = position[:, 0] + radius * torch.sin(phi).cuda() * torch.cos(theta).cuda()
    #     y = position[:, 1] + radius * torch.sin(phi).cuda() * torch.sin(theta).cuda()
    #     z = position[:, 2] + radius * torch.cos(phi).cuda()
    #     random_points = torch.stack([x, y, z], dim=1)
    #     return random_points  

    # def densify_and_split(self, grads, grad_threshold, scene_extent, visit_threshold, iteration, update_ratio, extra_ratio, extra_up, split_ratio, N=2):
    #     # Extract points that satisfy the gradient condition
    #     init_length = self.get_xyz.shape[0]
    #     visit_mask = (self.denom > visit_threshold).squeeze(dim=1)
    #     padding_grads = torch.zeros([self.get_xyz.shape[0] - grads.shape[0]],
    #                                        dtype=torch.float32, 
    #                                        device=grads.device)
    #     grads = torch.cat([grads, padding_grads], dim=0)
    #     update_value = self.fork ** update_ratio
    #     scale_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
    #     cur_threshold = grad_threshold * (update_value ** self.get_level.squeeze(1))
    #     extra_threshold = cur_threshold * extra_ratio
    #     selected_pts_mask = torch.where(grads >= cur_threshold, True, False)
    #     selected_pts_extra_mask = torch.where(grads >= extra_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask, scale_mask)
    #     selected_pts_extra_mask = torch.logical_and(selected_pts_extra_mask, scale_mask)
    #     if ~self.progressive or iteration > self.coarse_intervals[-1]:
    #         self._extra_level += extra_up * selected_pts_extra_mask.float()    

    #     new_xyz = self._xyz[selected_pts_mask]
    #     new_level = torch.clamp(self._level[selected_pts_mask] + 1, 0, self.levels-1)
    #     new_size = self.voxel_size/(float(self.fork) ** new_level)
    #     num_points = new_xyz.shape[0]
    #     new_xyz = self.generate_random_points_around(new_xyz, new_size, num_points)

    #     new_extra_level = torch.zeros(new_level.shape[0], dtype=torch.float, device="cuda")
    #     new_features_dc = self._features_dc[selected_pts_mask]
    #     new_features_rest = self._features_rest[selected_pts_mask]
    #     new_opacities = self._opacity[selected_pts_mask]
    #     new_scaling = self._scaling[selected_pts_mask] * split_ratio
    #     new_rotation = self._rotation[selected_pts_mask]
    #     self.densification_postfix(new_xyz, new_level, new_extra_level, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, visit_mask)

    # def densify_and_clone(self, grads, grad_threshold, scene_extent, visit_threshold, iteration, update_ratio, extra_ratio, extra_up):
    #     # Extract points that satisfy the gradient condition
    #     init_length = self.get_xyz.shape[0]
    #     visit_mask = (self.denom > visit_threshold).squeeze(dim=1)
    #     grads[~visit_mask] = 0.0
    #     update_value = self.fork ** update_ratio
    #     scale_mask = torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
    #     cur_threshold = grad_threshold * (update_value ** self.get_level.squeeze(1))
    #     extra_threshold = cur_threshold * extra_ratio
    #     selected_pts_mask = torch.where(grads >= cur_threshold, True, False)
    #     selected_pts_extra_mask = torch.where(grads >= extra_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask, scale_mask)
    #     selected_pts_extra_mask = torch.logical_and(selected_pts_extra_mask, scale_mask)
    #     if ~self.progressive or iteration > self.coarse_intervals[-1]:
    #         self._extra_level += extra_up * selected_pts_extra_mask.float()    

    #     new_xyz = self._xyz[selected_pts_mask]
    #     new_level = self._level[selected_pts_mask]
    #     new_size = self.voxel_size/(float(self.fork) ** new_level)
    #     num_points = new_xyz.shape[0]
    #     new_xyz = self.generate_random_points_around(new_xyz, new_size, num_points)
    #     new_extra_level = torch.zeros(new_level.shape[0], dtype=torch.float, device="cuda")
    #     new_features_dc = self._features_dc[selected_pts_mask]
    #     new_features_rest = self._features_rest[selected_pts_mask]
    #     new_opacities = self._opacity[selected_pts_mask]
    #     new_scaling = self._scaling[selected_pts_mask]
    #     new_rotation = self._rotation[selected_pts_mask]
    #     self.densification_postfix(new_xyz, new_level, new_extra_level, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, visit_mask)

    #     # Generate random points around each position

    # def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration, check_interval, success_threshold, update_ratio, extra_ratio, extra_up, split_ratio):
    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0
    #     grads_norm = torch.norm(grads, dim=-1)
    #     visit_threshold = check_interval*success_threshold*0.5
        
    #     self.densify_and_clone(grads_norm, max_grad, extent, visit_threshold, iteration, update_ratio, extra_ratio, extra_up)
    #     if ~self.progressive or iteration > self.coarse_intervals[-1]:
    #         self.densify_and_split(grads_norm, max_grad, extent, visit_threshold, iteration, update_ratio, extra_ratio, extra_up, split_ratio)

    #     prune_mask = (self.get_opacity < min_opacity).squeeze()
        
    #     visit_mask = (self.denom > visit_threshold).squeeze(dim=1)
    #     prune_mask = torch.logical_and(prune_mask, visit_mask) # [N] 
    #     if max_screen_size:
    #         big_points_vs = self.max_radii2D > max_screen_size
    #         big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    #         prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    #     self.prune_points(prune_mask)
    #     torch.cuda.empty_cache()

    # def add_densification_stats(self, viewspace_point_tensor, visible_mask):
    #     self.xyz_gradient_accum[visible_mask] += torch.norm(viewspace_point_tensor.grad[visible_mask,:2], dim=-1, keepdim=True)
    #     self.denom[visible_mask] += 1