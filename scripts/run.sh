#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

dataset=$1
case $dataset in
    beibei)
        echo "Running SWGCN for beibei dataset..."
        python main.py --model SWGCN --exp_name Run-SWGCN --dataset beibei --n_behavior 3 --lr 1e-3 --lamda 0.2  --multi 1 --self_loop_weight 0.2
        ;;
    taobao)
        echo "Running SWGCN for taobao dataset..."
        python main.py --model SWGCN --exp_name Run-SWGCN --dataset taobao --n_behavior 4 --lr 1e-3 --lamda 0.9 --multi 10 --self_loop_weight 1.0
        ;;
    ijcai)
        echo "Running SWGCN for ijcai dataset..."
        python main.py --model SWGCN --exp_name Run-SWGCN --dataset ijcai --n_behavior 4 --lr 1e-4 --lamda 0.3 --multi 1 --self_loop_weight 0.2
        ;;
    *)
        echo "Error: Unknown dataset '$dataset'."
        exit 1
        ;;
esac