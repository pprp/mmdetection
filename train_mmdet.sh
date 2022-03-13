#!/bin/bash 

module load cuda/10.1
module load anaconda
source activate pool 

# bash tools/dist_train.sh ./configs/retinanet/retinanet_r50_fpn_1x_coco.py 2
bash tools/dist_train.sh ./configs/retinanet/retinanet_r50_fpn_1x_coco_autorf.py 2