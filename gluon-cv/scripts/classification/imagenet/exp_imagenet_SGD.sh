#!/bin/sh
HOSTFILE=$1

for repeat in 1 2 3 4 5
do
    ########### SGD

    horovodrun -np 8 -hostfile ${HOSTFILE} python3 train_imagenet_hvd.py   \
    --rec-train /home/ubuntu/data/imagenet/train.rec --rec-train-idx /home/ubuntu/data/imagenet/train.idx   \
    --rec-val /home/ubuntu/data/imagenet/val.rec --rec-val-idx /home/ubuntu/data/imagenet/val.idx   \
    --model resnet50_v2 --mode hybrid   --lr 0.1 --lr-mode cosine --num-epochs 120 --batch-size 32 -j 6   \
    --warmup-epochs 5 --use-rec --dtype float16 --optimizer nag \
    --trainer sgd 
    
done
