#!/bin/sh
HOSTFILE=$1

for repeat in 1 2 3 4 5
do
    for inputsparse in 2 4 8 16 32
    do
        for outputsparse in 1
        do
            for lr in 0.05 0.1 0.5 1.0
            do
                for batchsize in 16
                do
                    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_efsgd_v1.py \
                    --model cifar_wideresnet40_8 --optimizer efsgdv1 --lr ${lr} --lr-decay 0.2 --nesterov \
                    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize} \
                    --input-sparse ${inputsparse} --output-sparse ${outputsparse} --layer-sparse 1

                done
            done
        done
    done
done

for repeat in 1 2 3 4 5
do
    for inputsparse in 32
    do
        for outputsparse in 2 4 8 16 32
        do
            for lr in 0.05 0.1 0.5 1.0
            do
                for batchsize in 16
                do
                    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_cifar100_hvd_efsgd_v1.py \
                    --model cifar_wideresnet40_8 --optimizer efsgdv1 --lr ${lr} --lr-decay 0.2 --nesterov \
                    --lr-decay-epoch 60,120,160 --wd 0.0005 --num-epochs 200 --batch-size ${batchsize} \
                    --input-sparse ${inputsparse} --output-sparse ${outputsparse} --layer-sparse 1

                done
            done
        done
    done
done