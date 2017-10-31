#!/bin/bash

export GPUID=0
export NET="squeezeDet"
export TRAIN_DIR="/tmp/SqueezeDet/"

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/train.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-net                      (squeezeDet|squeezeDet+|vgg16|resnet50)"
  echo "-gpu                      gpu id"
  echo "-train_dir                directory for training logs"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-net                      (squeezeDet|squeezeDet+|vgg16|resnet50)"
      echo "-gpu                      gpu id"
      echo "-train_dir                directory for training logs"
      exit 0
      ;;
    -net)
      export NET="$2"
      shift
      shift
      ;;
    -gpu)
      export GPUID="$2"
      shift
      shift
      ;;
    -train_dir)
      export TRAIN_DIR="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

case "$NET" in
  "squeezeDet")
    export PRETRAINED_MODEL_PATH="./data/SqueezeNet/squeezenet_v1.1.pkl"
    ;;
  "squeezeDet+")
    export PRETRAINED_MODEL_PATH="./data/SqueezeNet/squeezenet_v1.0_SR_0.750.pkl"
    ;;
  "resnet50")
    export PRETRAINED_MODEL_PATH="./data/ResNet/ResNet-50-weights.pkl"
    ;;
  "vgg16")
    export PRETRAINED_MODEL_PATH="./data/VGG16/VGG_ILSVRC_16_layers_weights.pkl"
    ;;
  *)
    echo "net architecture not supported."
    exit 0
    ;;
esac


# python ./src/train.py \
#   --dataset=KITTI \
#   --pretrained_model_path=$PRETRAINED_MODEL_PATH \
#   --data_path=/media/yongkie/YSDC/KITTI \
#   --image_set=train \
#   --train_dir="$TRAIN_DIR/train" \
#   --net=$NET \
#   --summary_step=100 \
#   --checkpoint_step=500 \
#   --gpu=$GPUID

#python src/train.py --dataset=KITTI --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl --data_path=../dataset/KITTI --image_set=train --train_dir="/tmp/logs/SqueezeDet/train" --net="squeezeDet" --summary_step=10 --checkpoint_step=50

python ./src/train.py \
  --dataset=BOSCH \
  --pretrained_model_path=$PRETRAINED_MODEL_PATH \
  --data_path=/media/yongkie/YSDC/dataset_train_rgb_y2 \
  --image_set=train \
  --train_dir="$TRAIN_DIR/train" \
  --net=$NET \
  --summary_step=10 \
  --checkpoint_step=10 \
  --gpu=$GPUID