#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

dataset=$1
case $dataset in
    beibei)
        echo "Running BPRMF for prepare beibei dataset..."
        python main.py --model BPRMF --exp_name Prepare-Data --dataset beibei --n_behavior 3 --early_stop 200
        ;;
    taobao)
        echo "Running BPRMF for prepare taobao dataset..."
        python main.py --model BPRMF --exp_name Prepare-Data --dataset taobao --n_behavior 4 --early_stop 200
        ;;
    ijcai)
        echo "Running BPRMF for prepare ijcai dataset..."
        python main.py --model BPRMF --exp_name Prepare-Data --dataset ijcai --n_behavior 4 --early_stop 200
        ;;
    *)
        echo "Error: Unknown dataset '$dataset'."
        exit 1
        ;;
esac