scene="mipnerf360/bicycle"
exp_name="baseline"
gpu=-1
ratio=1
resolution=-1
visible_threshold=0.6
base_layer=11
dist_ratio=0.999

./train.sh -d ${scene} -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --gpu ${gpu} --dist_ratio ${dist_ratio}