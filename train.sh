function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

iterations=40_000
warmup="False"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        -r|--resolution) resolution="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        --warmup) warmup="$2"; shift ;;
        --base_layer) base_layer="$2"; shift ;;
        --visible_threshold ) visible_threshold="$2"; shift ;;
        --dist_ratio) dist_ratio="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

if [ "$warmup" = "True" ]; then
    python train.py --eval -s /nas/shared/pjlab-lingjun-landmarks/renkerui/data/${data} -r ${resolution} --gpu ${gpu} \
    --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time \
    --base_layer ${base_layer} --visible_threshold ${visible_threshold} --progressive --dist_ratio ${dist_ratio}
else
    python train.py --eval -s /nas/shared/pjlab-lingjun-landmarks/renkerui/data/${data} -r ${resolution} --gpu ${gpu} \
    --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time \
    --base_layer ${base_layer} --visible_threshold ${visible_threshold} --progressive --dist_ratio ${dist_ratio}
fi
