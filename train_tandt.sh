exp_name="baseline"
gpu=-1
ratio=1
resolution=-1

base_layer=10
visible_threshold=0.6
dist_ratio=0.999

# example:
./train.sh -d 'tandt/truck' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'tandt/train' -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &