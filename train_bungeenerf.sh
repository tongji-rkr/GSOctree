exp_name="baseline"
gpu=-1
ratio=1
resolution=-1

appearance_dim=0
base_layer=-1
visible_threshold=0.1
dist_ratio=0.999

# example:
./train.sh -d 'bungeenerf/amsterdam' -l ${exp_name} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'bungeenerf/barcelona' -l ${exp_name} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'bungeenerf/bilbao' -l ${exp_name} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'bungeenerf/chicago' -l ${exp_name} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'bungeenerf/hollywood' -l ${exp_name} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'bungeenerf/pompidou' -l ${exp_name} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'bungeenerf/quebec' -l ${exp_name} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &
sleep 20s

./train.sh -d 'bungeenerf/rome' -l ${exp_name} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
--base_layer ${base_layer} --visible_threshold ${visible_threshold} --dist_ratio ${dist_ratio} --gpu ${gpu} &