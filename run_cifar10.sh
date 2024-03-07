#!/usr/bin/bash
# 
# Copyright (c) 2023-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mkdir -p ./log

PPM=True
ARCH=resnetdo34
LOSS=ce
DATASET=cifar10
OPTIM=adam
PRETRAIN=60
MOM=0.9
seed=100


# Parsing arguments
while getopts ":s:" flag; do
  case "${flag}" in
    s) seed=${OPTARG};;
    :)                                         # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal;;                          # Exit abnormally.
    *)                                         # If unknown (any other) option:
      exit_abnormal;;                          # Exit abnormally.
  esac
done

SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_seed-${seed}

### train
python3 -u train.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} --manualSeed ${seed}\
       --dataset ${DATASET} --save ${SAVE_DIR}  --optim ${OPTIM} --ppm ${PPM} --dropoutrate 0.2\
       #2>&1 | tee -a ${SAVE_DIR}.log

### eval
# python -u train.py --arch ${ARCH} --manualSeed ${seed}\
#        --loss ${LOSS} --dataset ${DATASET} \
#        --save ${SAVE_DIR} --evaluate \
#        2>&1 | tee -a ${SAVE_DIR}.log