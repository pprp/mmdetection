import torch 
from collections import OrderedDict

from_path = "/data/home/scv6681/run/project/poolformer/output/train/20220303-223131-resnet50_rf-224-acc79.66/model_best.pth.tar"

save_path = '/data/home/scv6681/run/project/mmdetection/work_dirs/resnet50_rf/model_r50_rf.pth.tar'

para_dict = torch.load(from_path)['state_dict']

new_dict = OrderedDict()

print(para_dict.keys())

for key, value in para_dict.items():
    print(key, key.replace('attention', 'receptive_field_attention'))
    new_dict[key.replace('attention', 'receptive_field_attention')] = value 

torch.save(new_dict, save_path)