mkdir -p ./log

DATASET=cifar10
PRETRAIN=60
MOM=0.9
seed=100
SAVE_DIR=logs

PPM=True
ARCH=vgg16_bn
LOSS=ce
OPTIM=sgdori

### train
python3 -u train.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} --manualSeed ${seed}\
       --dataset ${DATASET} --save ${SAVE_DIR}  --optim ${OPTIM} --ppm ${PPM}

PPM=True
ARCH=vgg16_bn
LOSS=sat
OPTIM=sgdori

### train
python3 -u train.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} --manualSeed ${seed}\
       --dataset ${DATASET} --save ${SAVE_DIR}  --optim ${OPTIM} --ppm ${PPM}

PPM=False
ARCH=resnet34
LOSS=ce
OPTIM=sgd1e-3

### train
python3 -u train.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} --manualSeed ${seed}\
       --dataset ${DATASET} --save ${SAVE_DIR}  --optim ${OPTIM} --ppm ${PPM}

PPM=False
ARCH=resnet34
LOSS=sat
OPTIM=sgd1e-3

### train
python3 -u train.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} --manualSeed ${seed}\
       --dataset ${DATASET} --save ${SAVE_DIR}  --optim ${OPTIM} --ppm ${PPM}

PPM=False
ARCH=resnet34
LOSS=ce
OPTIM=adam

### train
python3 -u train.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} --manualSeed ${seed}\
       --dataset ${DATASET} --save ${SAVE_DIR}  --optim ${OPTIM} --ppm ${PPM}

PPM=False
ARCH=resnet34
LOSS=sat
OPTIM=adam

### train
python3 -u train.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} \
       --loss ${LOSS} --manualSeed ${seed}\
       --dataset ${DATASET} --save ${SAVE_DIR}  --optim ${OPTIM} --ppm ${PPM}