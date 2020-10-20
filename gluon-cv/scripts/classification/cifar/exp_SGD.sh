#!/bin/sh
HOSTFILE=$1

for repeat in 1 2 3 4 5
do
    for lr in 0.05 0.1 0.5 1.0
    do
        for batchsize in 16
        do
            horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd.py \
            --model cifar_wideresnet40_8 --optimizer nag --lr ${lr} --lr-decay 0.2 \
            --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize}

        done
    done
done

