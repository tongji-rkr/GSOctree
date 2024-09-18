scene="matrix_city_new1"
exp_name="baseline"
gpu=-1
ratio=2
resolution=-1

base_layer=10
visible_threshold=0.01
dist_ratio=0.999

./train.sh -d ${scene} -l ${exp_name} -r ${resolution} --ratio ${ratio} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu}