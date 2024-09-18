exp_name="baseline"
gpu=-1
ratio=1
resolution=-1

base_layer=11
visible_threshold=0.7 
dist_ratio=0.999

# example:
./train.sh -d 'mipnerf360/bicycle' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'mipnerf360/garden' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'mipnerf360/stump' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'mipnerf360/room' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'mipnerf360/counter' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'mipnerf360/kitchen' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'mipnerf360/bonsai' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'mipnerf360/flowers' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'mipnerf360/treehill' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &