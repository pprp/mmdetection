#!/bin/bash 

module load cuda/10.1
module load anaconda
source activate pool 

# bash tools/dist_train.sh ./configs/retinanet/retinanet_r50_fpn_1x_coco.py 2
# modify the learning rate for faster rcnn. default gpus=8 our gpus=4
# bash tools/dist_train.sh ./configs/centernet/centernet_resnet18_140e_coco.py 8 

# 11:30-14:00
# bash tools/dist_train.sh ./configs/ssd/ssd_res50_scratch_600e_coco.py 4

# 11:30-16:13 
bash tools/dist_train.sh ./configs/faster_rcnn/faster_rcnn_r50_rf_fpn_1x_coco.py 4

# load weight test
# bash tools/dist_train.sh ./configs/faster_rcnn/faster_rcnn_r50_rf_fpn_1x_coco_test.py 1

