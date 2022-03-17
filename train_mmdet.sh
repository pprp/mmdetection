#!/bin/bash 

module load cuda/10.1
module load anaconda
source activate pool 

# bash tools/dist_train.sh ./configs/retinanet/retinanet_r50_fpn_1x_coco.py 2
# modify the learning rate for faster rcnn. default gpus=8 our gpus=4
# bash tools/dist_train.sh ./configs/faster_rcnn/faster_rcnn_r50_rf_fpn_1x_coco.py 4
bash tools/dist_train.sh ./configs/centernet/centernet_resnet18_140e_coco.py 8 