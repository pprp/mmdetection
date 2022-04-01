#!/bin/bash 

module load cuda/10.1
module load anaconda
source activate pool 

# bash tools/dist_train.sh ./configs/retinanet/retinanet_r50_fpn_1x_coco.py 2
# modify the learning rate for faster rcnn. default gpus=8 our gpus=4
# bash tools/dist_train.sh ./configs/centernet/centernet_resnet18_140e_coco.py 8 

# ============================= baseline =============================

# 11:30-14:00
# bash tools/dist_train.sh ./configs/ssd/ssd_res50_scratch_600e_coco.py 4

# 11:30-16:13 
# bash tools/dist_train.sh ./configs/faster_rcnn/faster_rcnn_r50_rf_fpn_1x_coco.py 4

# ====================================================================
# Faster Rcnn 20% coco 

# gridmask + 20% minitrain + full resnet50_rf weights loaded 
# bash tools/dist_train.sh configs/faster_rcnn/loadw_20%_wgridmask_faster_rcnn_r50_rf_fpn_1x_coco.py 4

# without gridmask + 20% minitrain + full resnet50_rf weights loaded 
# bash tools/dist_train.sh configs/faster_rcnn/loadw_20%_wogridmask_faster_rcnn_r50_rf_fpn_1x_coco.py 4

# ===================================================================
# SSD 20% coco 

# gridmask + 20% minitrain + full resnet50_rf ssd 40 epoch
# bash tools/dist_train.sh configs/ssd/wgridmask_40e_ssd_res50_rf_scratch_coco.py 4 


# without gridmask + 20% minitrain + full resnet50_rf ssd 40 epoch 
# bash tools/dist_train.sh configs/ssd/wogridmask_40e_ssd_res50_rf_scratch_coco.py 4 

# without gridmask + 20% minitrain + full resnet50_rf ssd 40 epoch + fix bugs in noiseop
# bash tools/dist_train.sh configs/ssd/wgridmask_40e_ssd_res50_rf_pretrain_coco_minitrain_fixnoiseop.py 4 --work-dir ./work_dirs/wgridmask_40e_ssd_res50_rf_pretrain_coco_minitrain_fixnoiseop_fixlr

# =======================================================================
# Faster RCNN 100% COCO with pretrained model
# bash tools/dist_train.sh configs/faster_rcnn/pretrained_whole_coco_faster_rcnn_r50_rf_fpn_1x_coco.py 8 
# bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_cbnet_coco_12e.py 8 --work-dir ./work_dirs/faster_rcnn_cbnet_coco_12e

# bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco_iou_ohem.py 8
bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco_iou_test.py 8
# bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py 8
#  bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco_ohem.py 8 

# SSD 100% COCO with pretrained model 
# bash tools/dist_train.sh configs/ssd/pretrained_whole_20e_ssd_res50_rf_coco.py 8 

# bash tools/dist_train.sh configs/ssd/pretrained_whole_20e_ssd_res50_rf_coco.py 4 --work-dir ./work_dirs/pretrained_whole_20e_ssd_res50_rf_coco_mixup

