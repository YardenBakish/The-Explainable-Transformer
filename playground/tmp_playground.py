

#finetune deit from scratch

###########################################################################################
###########################################################################################

#BASIC DEIT

###########################################################################################
###########################################################################################

# Finetune basic DEIT

python main.py --auto-save --results-dir finetuned_models  --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1

#continue finetuning basic DEIT

python main.py --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1





###########################################################################################
###########################################################################################

# ABLATED COMPONENT

###########################################################################################
###########################################################################################


#train ablated component fron finetuned
python main.py --is-ablation --ablated-component bias --auto-save --results-dir finetuned_models   --finetune finetuned_models/none_IMNET100/best_checkpoint.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1



#continue train ablated component (similar for variant)
python main.py --is-ablation --ablated-component bias --auto-save --auto-resume --results-dir finetuned_models   --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1

#eval ablated component
python main.py --ablated-component bias --eval --auto-resume --results-dir finetuned_models --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30 --data-path ./ --num_workers 4 --batch-size 128 --warmup-epochs 1



###########################################################################################
###########################d################################################################

# VARIANT

#train variant
python main.py --variant act_softplus_norm_rms --auto-save --results-dir finetuned_models   --finetune finetuned_models/IMNET/basic/best_checkpoint.pth  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1


#eval variant(same)
python main.py --variant relu --eval --resume finetuned_models/relu/best_checkpoint.pth --results-dir finetuned_models --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30 --data-path ./ --num_workers 4 --batch-size 128 --warmup-epochs 1

#continue finetuning relu
python main.py --variant rmsnorm_softplus --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 80  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1



###########################################################################################
###########################################################################################



###########################################################################################
###########################################################################################

# PERTURBATIONS

###########################################################################################
###########################################################################################


#evaluate perturbations for basic automatically using best_checkpoint

python evaluate_perturbations.py --output-dir finetuned_models  --method transformer_attribution --data-path ./ --batch-size 1 --custom-trained-model finetuned_models/none/best_checkpoint.pth --num-workers 1 --both


#evaluate perturbations for basic with a specific model - RECOMMENDED


python evaluate_perturbations.py --custom-trained-model finetuned_models/none/checkpoint_14.pth --output-dir finetuned_models  --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both

#evalutae for perturbation or ablated

python evaluate_perturbations.py --ablated_component bias   --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both
python evaluate_perturbations.py --variant rmsnorm --custom-trained-model finetuned_models/rmsnorm_IMNET100/checkpoint_29.pth  --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both


#analyze pertubations
python analyze_pert_results.py --variant norm_bias_ablation --mode pertubations --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both

###########################################################################################
###########################################################################################

# HEATMAP


#visualize

###########################################################################################
###########################################################################################


#from here



#BATCH -START
python main.py --variant softplus --auto-save --finetune finetuned_models/none/best_checkpoint.pth  --results-dir finetuned_models   --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 30  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1


#BATCH - CONTINUE
python main.py --variant batchnorm --auto-save --auto-resume --results-dir finetuned_models  --model deit_tiny_patch16_224  --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8   --epochs 50  --data-path ./ --num_workers 4 --batch-size 128  --warmup-epochs 1





val/n01614925/ILSVRC2012_val_00006571.JPEG
val/n01877812/ILSVRC2012_val_00014040.JPEG
val/n02006656/ILSVRC2012_val_00028586.JPEG
val/n01514859/ILSVRC2012_val_00032162.JPEG
val/n01440764/ILSVRC2012_val_00046252.JPEG
val/n01985128/ILSVRC2012_val_00032174.JPEG
finetuned_models/relu/checkpoint_29.pth





python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/attn_act_relu/checkpoint_75.pth --variant attn_act_relu --method transformer_attribution --sample-path val/n01614925/ILSVRC2012_val_00006571.JPEG
python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/attn_act_relu/checkpoint_75.pth --variant attn_act_relu --method transformer_attribution --sample-path val/n01877812/ILSVRC2012_val_00014040.JPEG
python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/attn_act_relu/checkpoint_75.pth --variant attn_act_relu --method transformer_attribution --sample-path val/n02006656/ILSVRC2012_val_00028586.JPEG
python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/attn_act_relu/checkpoint_75.pth --variant attn_act_relu --method transformer_attribution --sample-path val/n01514859/ILSVRC2012_val_00032162.JPEG
python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/attn_act_relu/checkpoint_75.pth --variant attn_act_relu --method transformer_attribution --sample-path val/n01985128/ILSVRC2012_val_00032174.JPEG




python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/basic/best_checkpoint.pth --method transformer_attribution --sample-path val/n01614925/ILSVRC2012_val_00006571.JPEG
python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/basic/best_checkpoint.pth --method transformer_attribution --sample-path val/n01877812/ILSVRC2012_val_00014040.JPEG
python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/basic/best_checkpoint.pth --method transformer_attribution --sample-path val/n02006656/ILSVRC2012_val_00028586.JPEG
python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/basic/best_checkpoint.pth --method custom_lrp --sample-path val/n01514859/ILSVRC2012_val_00032162.JPEG
python visualize_heatmap.py --custom-trained-model finetuned_models/IMNET100/basic/best_checkpoint.pth --method transformer_attribution --sample-path val/n01985128/ILSVRC2012_val_00032174.JPEG







python analyze_pert_results.py --variant relu --mode runPerts --method transformer_attribution --data-path ./ --batch-size 1  --num-workers 1 --both







python visualize_heatmap.py --custom-trained-model finetuned_models/batchnorm_IMNET1000/deit_tiny_prepbn.pth --variant batchnorm_param --method transformer_attribution --sample-path val/n01877812/ILSVRC2012_val_00014040.JPEG


###############

python generate_visualizations.py --method transformer_attribution --data-path /home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/ --data-set IMNET --batch-size 1 --num-workers 1 --variant batchnorm_param --custom-trained-model finetuned_models/batchnorm_IMNET1000/deit_tiny_prepbn.pth --work-env finetuned_models/tmp2/work_env



python generate_perturbations.py --debug --method transformer_attribution --data-set IMNET --variant batchnorm_param --custom-trained-model finetuned_models/batchnorm_IMNET1000/deit_tiny_prepbn.pth  --work-env finetuned_models/tmp2/work_env --output-dir finetuned_models/batchnorm_param_IMNET --neg 1





python generate_visualizations.py --method transformer_attribution --data-path /home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/ --data-set IMNET --batch-size 1 --num-workers 1 --custom-trained-model finetuned_models/basic_IMNET100/deit_tiny_patch16_224-a1311bcf.pth --work-env finetuned_models/tmp/work_env


python generate_perturbations.py  --method transformer_attribution --data-set IMNET --custom-trained-model finetuned_models/basic_IMNET100/deit_tiny_patch16_224-a1311bcf.pth  --work-env finetuned_models/tmp/work_env --neg 1






python evaluate_perturbations.py --work-env finetuned_models/batchnorm_IMNET1000/work_env --variant batchnorm_param  --custom-trained-model finetuned_models/batchnorm_IMNET1000/deit_tiny_prepbn.pth --data-set IMNET  --method transformer_attribution --data-path /home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/ --batch-size 1  --num-workers 1 --both



python evaluate_perturbations.py --work-env finetuned_models/basic_IMNET100/work_env  --custom-trained-model finetuned_models/basic_IMNET100/deit_tiny_patch16_224-a1311bcf.pth --data-set IMNET  --method transformer_attribution --data-path /home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/ --batch-size 1  --num-workers 1 --both --output-dir  finetuned_models/basic_IMNET100/pert_results/


###############################################################################################











python main.py --variant <variant> --resume <model.pth> --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 30  --num_workers 4 --batch-size 128 --warmup-epochs 1


python main.py --variant attn_act_sigmoid --auto-save --auto-resume --model deit_tiny_patch16_224 --seed 0 --lr 5e-6 --min-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --epochs 120  --num_workers 4 --batch-size 128 



python evaluate_perturbations.py --custom-trained-model <model.pth> --variant <variant> --normalized-pert 0 --fract 0.04 --both --work_env finetuned_models/tmp/work_env --output-dir tmp/pert_results --model transformer_attribution 



python generate_perturbations.py --custom-trained-model <model.pth> --variant <variant> --normalized-pert 0 --fract 0.04 --both --work_env finetuned_models/tmp/work_env --output-dir tmp/pert_results --model transformer_attribution 











#from models.model_wrapper import model_env 
from models.model_handler import model_env 

from ViT_explanation_generator import LRP
import torchvision.transforms as transforms
import argparse
from PIL import Image
import torch
from samples.CLS2IDX import CLS2IDX
import numpy as np
import matplotlib.pyplot as plt
import os
import config

import cv2
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])





# create heatmap from mask on image
def show_cam_on_image(img, mask):
   
    x = np.uint8(255 * mask)

    heatmap = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    print(transformer_attribution)
   
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



def generate_visualization_LRP(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="full", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    #print(image_transformer_attribution.shape)
    
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def print_top_classes(predictions, **kwargs):    
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])
    
    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--sample-path', 
                        required = True,
                        help='')

  parser.add_argument('--custom-trained-model', 
                        required = True,
                        help='')
  parser.add_argument('--data-set', default='IMNET100', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

  parser.add_argument('--data-path', default='',)
  
  parser.add_argument('--variant', default = 'basic' , type=str, help="")
  parser.add_argument('--class-index', 
                       # default = "243",
                       type=int,
                        help='') #243 - dog , 282 - cat
  parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                        choices=['transformer_attribution', 'full_lrp'],
                        help='')
      
  
  
  args = parser.parse_args()
  config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)

  image = Image.open(args.sample_path)
  image_transformed = transform(image)

  
  if args.data_set == "IMNET100":
    args.nb_classes = 100
  else:
     args.nb_classes = 1000

  
  model = model_env(pretrained=False, 
                      args = args,
                      hooks = True,
                    )
        #model_LRP.head = torch.nn.Linear(model_LRP.head.weight.shape[1],100)
  checkpoint = torch.load(args.custom_trained_model, map_location='cpu')
  state_dict = model.state_dict()

  if args.variant == "variant_weight_normalization"
  for key, original_weight  in checkpoint['model'].copy().items():
    
    if ".weight" in key:
      if key not in state_dict:
        #print(original_weight.shape)
        weight_norm_value = torch.norm(original_weight, dim=1)
        #print(weight_norm_value.shape)
        weight_direction = original_weight / weight_norm_value[:,None]
        print(key)
        del checkpoint['model'][key]
        checkpoint['model'][key.replace('.weight', '.weight_v')] = weight_direction
        checkpoint['model'][key.replace('.weight', '.weight_g')] = weight_norm_value.unsqueeze(1)



  model.load_state_dict(checkpoint['model'], strict=False)
  model.cuda()
  model.eval()
  attribution_generator = LRP(model)

  output = model(image_transformed.unsqueeze(0).cuda())
  print_top_classes(output)

  filename = os.path.basename(args.sample_path)
    # Remove the file extension
  img_name = os.path.splitext(filename)[0]

  method_name = None
  vis = None
  if args.method == "transformer_attribution":
    vis = generate_visualization(image_transformed, args.class_index)
    method_name = "Att"
  else:
    vis = generate_visualization_LRP(image_transformed, args.class_index)
    method_name = "lrp"

  saved_image_path = f"testing/{img_name}_{method_name}_{args.variant}.png"
 
  plt.imsave(saved_image_path, vis)

  



#print_top_classes(output)



























#MAINSCRIPT###############################################













# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import wandb
from old.model_no_hooks_ablation import deit_tiny_patch16_224 as vit_LRP
from models.model_handler import model_env 
from torch.nn.utils.parametrizations import weight_norm
from modules.layers_ours import *


import os

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from deit.datasets import build_dataset
from deit.engine import train_one_epoch, evaluate
from deit.losses import DistillationLoss
from deit.samplers import RASampler
from deit.augment import new_data_aug_generator
from misc.helper_functions import is_valid_directory ,create_directory_if_not_exists, update_json
import deit.models as models
import deit.models_v2 as models_v2

import deit.utils as utils
import config

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    # monitoring arguments
    parser.add_argument('--name', default=None, type=str, help='run name')
    parser.add_argument('--project', default='', type=str, help='project name')

    # experiment arguments
    parser.add_argument('--auto-start-train', action='store_true')
    parser.add_argument('--backup-interval', type=int,
                        default=3,)
    
    parser.add_argument('--results-dir',type=str)
    parser.add_argument('--auto-resume',action='store_true',)
    parser.add_argument('--auto-save',action='store_true',)
    parser.add_argument('--variant', default ='basic',  type=str, help="")


    
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--verbose', action='store_true')


    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path',  type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET100', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    
    #setup
    config.get_config(args)
    exp_name      = args.variant


    exp_name = f'{args.data_set}/{exp_name}' 
    if args.auto_save:
        results_exp_dir     = f'{args.dirs["results_dir"]}/{exp_name}'
        create_directory_if_not_exists(f'{args.dirs["results_dir"]}/{exp_name}')
        args.output_dir     = results_exp_dir

    if args.auto_resume:
        results_exp_dir     = f'{args.dirs["results_dir"]}/{exp_name}'
        last_check_point    =  is_valid_directory(results_exp_dir)
        if last_check_point == False:
            print(f"problem with work enviroment {results_exp_dir}")
            exit(1)
        else:
            args.resume     = last_check_point
            args.finetune   = last_check_point
    
    #wandb
    if args.project != '':
        if args.verbose:
            print("initiatin wan")
        run_name_with_model = run_name_with_model if not args.name else args.name
        # should_resume = True if args.resume != None and args.resume != '' else False
        wandb.init(project=args.project, name=run_name_with_model, config=args)
        args.wandb = True
    else:
        args.wandb = False



    #FIXME: currently have limited access to gpu. Change later
    #utils.init_distributed_mode(args)
    args.distributed = False
    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    
  
    #FIXME: currently have limited access to gpu (UNIFIED paper set it to true by default). fix later
  
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    if args.verbose:
        print("Training Dataset built successfully")

    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if args.verbose:
        print("Validation Dataset built successfully")

 

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.verbose:
        print(f"Creating model: {args.model}")


    '''model = vit_LRP(
        pretrained=False,
        num_classes=args.nb_classes,
        ablated_component= args.ablated_component
    )'''

    model = model_env(args=args,
                      hooks = False,
                      )

    
    '''model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
    )'''
    
    #FIXME: currently hardcoded, might change we additional datasets are included
    if args.nb_classes == 100:
        model.head = torch.nn.Linear(model.head.weight.shape[1],args.nb_classes)

                    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k not in state_dict:
                continue
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        
        if args.variant == "bias_ablation":
            for key in checkpoint_model.copy():
                if "bias" in key:
                    if key not in state_dict:
                        print(f" removing {key} from pretrained checkpoint")
                        del checkpoint_model[key]
        

        '''if args.variant == "variant_weight_normalization":
          for key, original_weight  in checkpoint_model.copy().items():
            if ".weight" in key:
              if key not in state_dict:
                print(original_weight.shape)
                weight_norm_value = torch.norm(original_weight, dim=1)
                print(weight_norm_value.shape)


                

                weight_direction = original_weight / weight_norm_value[:,None]
                print(key)
                del checkpoint_model[key]

                checkpoint_model[key.replace('.weight', '.weight_v')] = weight_direction
                checkpoint_model[key.replace('.weight', '.weight_g')] = weight_norm_value.unsqueeze(1)'''
                

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        
        
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

  

    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.wandb:
        wandb.log({'n_parameters': n_parameters}, commit=False)
 
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        #
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which 
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
       if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
       else:
            checkpoint = torch.load(args.resume, map_location='cpu')
       

       if args.variant == "variant_weight_normalization":
          checkpoint_modelX = checkpoint['model']
          for key, original_weight  in checkpoint_modelX.copy().items():
            if ".weight" in key:
              pass
              #print(key)
              
              #if key not in state_dict:
              #  print(original_weight.shape)
              #  weight_norm_value = torch.norm(original_weight, dim=1)
              #  print(weight_norm_value.shape)


            
       
       model_without_ddp.load_state_dict(checkpoint['model'])
       if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
       lr_scheduler.step(args.start_epoch)
    if args.eval:
    
      for name, module in model.named_modules():
        print(name)
        continue
        if isinstance(module, WeightNormLinear):
          print(f"Module: {name}")
          print(module._parameters.keys())
          continue
          v_param = module._parameters.get('weight_v')
          g_param = module._parameters.get('weight_g')
            
          print(f"Module: {name}")
          print(f"Weight V shape: {v_param.shape if v_param is not None else 'Not found'}")
          print(f"Weight G shape: {g_param.shape if g_param is not None else 'Not found'}")
            
            # Verify gradients
          if v_param is not None and g_param is not None:
            print(f"V requires grad: {v_param.requires_grad}")
            print(f"G requires grad: {g_param.requires_grad}")
          exit(1)

          test_stats = evaluate(data_loader_val, model, device)
          print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
          utils.save_on_master({
                      'model': model_without_ddp.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_scheduler': lr_scheduler.state_dict(),
                     
                  }, "/content/checkpoint_0.pth")
          
      exit(1)    
      return
    


    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    #train
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        print(f"\n\n epoch: {epoch}\n\n")
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )
        print("managed to finish training for one epoch")

        lr_scheduler.step(epoch)
        if args.output_dir and (epoch %args.backup_interval == 0):
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']}%")
        update_json(f'{args.output_dir}/acc_results.json', {f"{epoch}_acc": f"* Acc@1 {test_stats['acc1']:.1f} Acc@5 {test_stats['acc5']:.1f} loss {test_stats['loss']:.1f}"})


        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / f'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
            update_json(f'{args.output_dir}/acc_results.json', {f'best': f'{max_accuracy:.2f}'})
            
        print(f'Max accuracy: {max_accuracy:.2f}')

        if args.wandb:
            # Log best accuracy
            wandb.log({"best_acc1": max_accuracy}, commit=False)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        
        
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
  
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    
    main(args)



































######################





""" Vision Transformer (ViT) in PyTorch
Hacked together by / Copyright 2020 Ross Wightman
""" 
import torch
import torch.nn as nn
from einops import rearrange
from modules.layers_ours import *
from baselines.ViT.weight_init import trunc_normal_
from baselines.ViT.layer_helpers import to_2tuple
from functools import partial
import inspect
from torch.nn.utils.parametrizations import weight_norm



def safe_call(func, **kwargs):
    # Get the function's signature
    sig = inspect.signature(func)
    
    # Filter kwargs to only include parameters the function accepts
    filtered_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in sig.parameters
    }
    
    # Call the function with only its compatible parameters
    return func(**filtered_kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., isWithBias=True, activation = GELU):
        super().__init__()
        print(f"inside MLP with isWithBias: {isWithBias} and activation {activation}")
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = WeightNormLinear(in_features, hidden_features, bias = isWithBias) 
        self.act = activation
        self.fc2 =  WeightNormLinear(hidden_features, out_features, bias = isWithBias) 
        self.drop = Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0., 
       
                attn_activation = Softmax(dim=-1), 
                isWithBias      = True):
        
        super().__init__()

        print(f"inside attention with activation : {attn_activation} | bias: {isWithBias} ")
        self.num_heads = num_heads

        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
   

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = WeightNormLinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = WeightNormLinear(dim, dim, bias = isWithBias) 
        self.proj_drop = Dropout(proj_drop)
        self.attn_activation = attn_activation

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale
       
        attn = self.attn_activation(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        #attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
      
        cam1 = self.attn_activation.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,   
                isWithBias = True,
                layer_norm = partial(LayerNorm, eps=1e-6),
                activation = GELU,
                attn_activation = Softmax(dim=-1) ):
        super().__init__()
        print(f"Inside block with bias: {isWithBias} | norm : {layer_norm} | activation: {activation} | attn_activation: {attn_activation}  ")

        self.norm1 = safe_call(layer_norm, normalized_shape= dim, bias = isWithBias ) 
        self.attn = Attention(
            dim, num_heads  = num_heads, 
            qkv_bias        = qkv_bias, 
            attn_drop       = attn_drop, 
            proj_drop       = drop, 
            attn_activation = attn_activation,
            isWithBias      = isWithBias,
           )
        
        self.norm2 = safe_call(layer_norm, normalized_shape= dim, bias = isWithBias ) 
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       drop=drop, 
                       isWithBias = isWithBias, 
                       activation = activation)

        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()

    

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
      
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
      
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
       
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
      
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = NormalizedConv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1,2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, mlp_head=False, drop_rate=0., attn_drop_rate=0., 
                isWithBias = True,
                layer_norm = partial(LayerNorm, eps=1e-6),
                activation = GELU,
                attn_activation = Softmax(dim=-1),
                last_norm       = LayerNorm,):
        
        super().__init__()
        print(f"calling vision transformer with bias: {isWithBias} | norm : {layer_norm} | activation: {activation} | attn_activation: {attn_activation}  ")

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.isWithBias = isWithBias

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,         
           
                isWithBias      = isWithBias, 
                layer_norm      = layer_norm,
                activation      = activation,
                attn_activation = attn_activation,)
            for i in range(depth)])

        self.norm = safe_call(last_norm, normalized_shape= embed_dim, bias = isWithBias ) 
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes, 0., isWithBias, activation)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head =  WeightNormLinear(embed_dim, num_classes, bias = isWithBias)
        # FIXME not quite sure what the proper weight init is supposed to be,
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.pool = IndexSelect()
        self.add = Add()

        self.inp_grad = None

    def save_inp_grad(self,grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None and self.isWithBias != False:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if self.isWithBias != False:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        #x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)
     
        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
     
        cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            return cam
            
        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam




def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )

        #checkpoint = torch.load("deit_base_patch16_224-b5f2ef4d.pth", map_location="cpu")
        #model.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint["model"])
    return model




def deit_tiny_patch16_224(pretrained=False, 
                          isWithBias = True,
                          qkv_bias   = True,
                          layer_norm = partial(LayerNorm, eps=1e-6),
                          activation = GELU,
                          attn_activation = Softmax(dim=-1) ,
                          last_norm       = LayerNorm,
                          **kwargs):

    print(f"calling vision transformer with bias: {isWithBias} | norm : {layer_norm} | activation: {activation} | attn_activation: {attn_activation}  ")
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, 
        qkv_bias        = isWithBias, 
        isWithBias      = isWithBias, 
        layer_norm      = layer_norm,
        activation      = activation,
        attn_activation = attn_activation,
        last_norm       = last_norm,
        **kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



















################# relu perturbations - with cp_rule
{
    "transformer_attribution_pos_top": [
        0.5416,
        0.396,
        0.2984,
        0.2432,
        0.1696,
        0.1288,
        0.08,
        0.0536,
        0.04
    ],
    "transformer_attribution_pos_auc_top": 0.16604,
    "transformer_attribution_pos_target": [
        0.508,
        0.3872,
        0.2848,
        0.2248,
        0.1552,
        0.1152,
        0.0728,
        0.0472,
        0.0368
    ],
    "transformer_attribution_pos_auc_target": 0.15596,
    "transformer_attribution_pos_top_blur": [
        0.6424,
        0.5064,
        0.4136,
        0.332,
        0.2608,
        0.2152,
        0.1544,
        0.1056,
        0.076
    ],
    "transformer_attribution_pos_auc_top_blur": 0.23472,
    "transformer_attribution_pos_target_blur": [
        0.596,
        0.4752,
        0.388,
        0.316,
        0.2504,
        0.2,
        0.148,
        0.1,
        0.072
    ],
    "transformer_attribution_pos_auc_target_blur": 0.22116,
    "transformer_attribution_neg_top": [
        0.8848,
        0.8544,
        0.8256,
        0.7856,
        0.7168,
        0.6288,
        0.5168,
        0.3632,
        0.2192
    ],
    "transformer_attribution_neg_auc_top": 0.52432,
    "transformer_attribution_neg_target": [
        0.7944,
        0.7808,
        0.764,
        0.728,
        0.6664,
        0.5872,
        0.4888,
        0.3464,
        0.2104
    ],
    "transformer_attribution_neg_auc_target": 0.48639999999999994,
    "transformer_attribution_neg_top_blur": [
        0.9408,
        0.8976,
        0.8832,
        0.8608,
        0.8288,
        0.8024,
        0.7408,
        0.6608,
        0.5112
    ],
    "transformer_attribution_neg_auc_top_blur": 0.6400399999999999,
    "transformer_attribution_neg_target_blur": [
        0.8104,
        0.7968,
        0.792,
        0.7752,
        0.7576,
        0.7336,
        0.676,
        0.616,
        0.4776
    ],
    "transformer_attribution_neg_auc_target_blur": 0.5791200000000001,
    "custom_lrp_pos_top": [
        0.446,
        0.322,
        0.253,
        0.176,
        0.118,
        0.087,
        0.066,
        0.044,
        0.044
    ],
    "custom_lrp_pos_auc_top": 0.1311,
    "custom_lrp_pos_target": [
        0.423,
        0.316,
        0.247,
        0.18,
        0.115,
        0.087,
        0.064,
        0.039,
        0.039
    ],
    "custom_lrp_pos_auc_target": 0.1279,
    "custom_lrp_pos_top_blur": [
        0.562,
        0.393,
        0.309,
        0.256,
        0.198,
        0.156,
        0.133,
        0.099,
        0.08
    ],
    "custom_lrp_pos_auc_top_blur": 0.1865,
    "custom_lrp_pos_target_blur": [
        0.506,
        0.369,
        0.299,
        0.252,
        0.2,
        0.16,
        0.132,
        0.099,
        0.077
    ],
    "custom_lrp_pos_auc_target_blur": 0.18025,
    "custom_lrp_neg_top": [
        0.868,
        0.828,
        0.793,
        0.734,
        0.658,
        0.567,
        0.454,
        0.335,
        0.233
    ],
    "custom_lrp_neg_auc_top": 0.49195000000000005,
    "custom_lrp_neg_target": [
        0.776,
        0.76,
        0.73,
        0.69,
        0.623,
        0.543,
        0.434,
        0.327,
        0.232
    ],
    "custom_lrp_neg_auc_target": 0.46109999999999995,
    "custom_lrp_neg_top_blur": [
        0.943,
        0.91,
        0.896,
        0.854,
        0.827,
        0.79,
        0.739,
        0.671,
        0.555
    ],
    "custom_lrp_neg_auc_top_blur": 0.6436000000000001,
    "custom_lrp_neg_target_blur": [
        0.813,
        0.807,
        0.802,
        0.78,
        0.751,
        0.737,
        0.675,
        0.622,
        0.511
    ],
    "custom_lrp_neg_auc_target_blur": 0.5836
}
































import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio
import os
from tqdm import tqdm
from utils.metrices import *
from models.model_handler import model_env 
from utils import render
from utils.saver import Saver
from utils.iou import IoU
import config
from data.imagenet_new import Imagenet_Segmentation
from samples.CLS2IDX import CLS2IDX

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from misc.helper_functions import is_valid_directory ,create_directory_if_not_exists, update_json


from ViT_explanation_generator import Baselines, LRP
from old.model import deit_tiny_patch16_224 as vit_LRP
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch.nn.functional as F

plt.switch_backend('agg')


# hyperparameters
num_workers = 0
batch_size = 1

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]

# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')
parser.add_argument('--custom-trained-model', type=str, 
                    help='Model path')

parser.add_argument('--variant', default = 'basic', help="")
parser.add_argument('--input-size', default=224, type=int, help='images input size')

parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")

parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='grad_rollout',
                    choices=[ 'rollout', 'lrp','transformer_attribution', 'full_lrp', 'lrp_last_layer',
                              'attn_last_layer', 'attn_gradcam', 'custom_lrp'],
                    help='')
parser.add_argument('--thr', type=float, default=0.,
                    help='threshold')
parser.add_argument('--K', type=int, default=1,
                    help='new - top K results')
parser.add_argument('--save-img', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-ia', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fgx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-m', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-reg', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--is-ablation', type=bool,
                    default=False,
                    help='')

parser.add_argument('--data-path', type=str,
                     
                        help='')
parser.add_argument('--imagenet-seg-path', type=str, required=True)
args = parser.parse_args()

config.get_config(args, skip_further_testing = True)
config.set_components_custom_lrp(args)


args.checkname = args.method + '_' + args.arc

alpha = 2

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Define Saver
saver = Saver(args)
saver.results_dir = os.path.join(saver.experiment_dir, 'results')
if not os.path.exists(saver.results_dir):
    os.makedirs(saver.results_dir)
if not os.path.exists(os.path.join(saver.results_dir, 'input')):
    os.makedirs(os.path.join(saver.results_dir, 'input'))
if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
    os.makedirs(os.path.join(saver.results_dir, 'explain'))

args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
if not os.path.exists(args.exp_img_path):
    os.makedirs(args.exp_img_path)
args.exp_np_path = os.path.join(saver.results_dir, 'explain/np')
if not os.path.exists(args.exp_np_path):
    os.makedirs(args.exp_np_path)

# Data
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


sizeX = int(args.input_size / args.eval_crop_ratio)

imagenet_trans = transforms.Compose([
        #transforms.Resize(sizeX, interpolation=3),
        #transforms.CenterCrop(args.input_size), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])



test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST),
])

ds = Imagenet_Segmentation(args.imagenet_seg_path,
                           transform=imagenet_trans, target_transform=test_lbl_trans)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

# Model

#FIXME: currently only attribution method is tested. Add support for other methods using other variants 
model = vit_LRP(pretrained=True).cuda()
baselines = Baselines(model)

if args.custom_trained_model != None:
    if args.data_set == 'IMNET100':
        args.nb_classes = 100
    else:
        args.nb_classes = 1000
      
    model_LRP = model_env(pretrained=False, 
                    args = args,
                    hooks = True,
                )
    
    checkpoint = torch.load(args.custom_trained_model, map_location='cpu')

    model_LRP.load_state_dict(checkpoint['model'], strict=False)
    model_LRP.to(device)


model_LRP.eval()
lrp = LRP(model_LRP)

# orig LRP
model_orig_LRP = vit_LRP(pretrained=True).cuda()
model_orig_LRP.eval()
orig_lrp = LRP(model_orig_LRP)

metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)

model.eval()


def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    # pred[0, 0] = 282
    # print('Pred cls : ' + str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = T.cuda()

    return Tt


def eval_batch(image, labels, evaluator, index):
    evaluator.zero_grad()
    # Save input image
    if args.save_img:
        img = image[0].permute(1, 2, 0).data.cpu().numpy()
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype('uint8')
        Image.fromarray(img, 'RGB').save(os.path.join(saver.results_dir, 'input/{}_input.png'.format(index)))
        Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'), 'RGB').save(
            os.path.join(saver.results_dir, 'input/{}_mask.png'.format(index)))

    print(index)
    #np.save(f"stuff/input_test_{index}.npy", image.data.cpu().numpy())
    
    
    image.requires_grad = True

    image = image.requires_grad_()
    predictions = evaluator(image)
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()

    #print(class_indices)
    #print("\n")
    #print(f"PREDICTED CLASS: {CLS2IDX[class_indices[0]]} ")
    #print(f"PROBABILITY: {100 * prob[0, class_indices[0]]} ")
    #print("\n")
    #print("\n")



    #print(100 * prob[0, class_indices[0]])
    #print("\n")
#
    #print(100 * prob[0, class_indices[1]])
    #print(100 * prob[0, class_indices[2]])
#
    #print(100 * prob[0, class_indices[3]])
#
    #print(100 * prob[0, class_indices[4]])



    # segmentation test for the rollout baseline
    if args.method    == 'custom_lrp':
        Res = lrp.generate_LRP(image.cuda(), method="custom_lrp", cp_rule = args.cp_rule).reshape(14, 14).unsqueeze(0).unsqueeze(0) 
    
    elif args.method == 'rollout':
        Res = baselines.generate_rollout(image.cuda(), start_layer=1).reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the LRP baseline (this is full LRP, not partial)
    elif args.method == 'full_lrp':
        Res = orig_lrp.generate_LRP(image.cuda(), method="full", cp_rule = args.cp_rule).reshape(batch_size, 1, 224, 224)
    
    # segmentation test for our method
    elif args.method == 'transformer_attribution':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution", cp_rule = args.cp_rule).reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the partial LRP baseline (last attn layer)
    elif args.method == 'lrp_last_layer':
        Res = orig_lrp.generate_LRP(image.cuda(), method="last_layer", is_ablation=args.is_ablation, cp_rule = args.cp_rule)\
            .reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the raw attention baseline (last attn layer)
    elif args.method == 'attn_last_layer':
        Res = orig_lrp.generate_LRP(image.cuda(), method="last_layer_attn", is_ablation=args.is_ablation, cp_rule = args.cp_rule)\
            .reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the GradCam baseline (last attn layer)
    elif args.method == 'attn_gradcam':
        Res = baselines.generate_cam_attn(image.cuda()).reshape(batch_size, 1, 14, 14)

    if args.method != 'full_lrp':
        # interpolate to full image size (224,224)
        Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
    
    # threshold between FG and BG is the mean    
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1-Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0


    # TEST
    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    if args.save_img:
        # Save predicted mask
        mask = F.interpolate(Res_1, [64, 64], mode='bilinear')
        mask = mask[0].squeeze().data.cpu().numpy()
        # mask = Res_1[0].squeeze().data.cpu().numpy()
        mask = 255 * mask
        mask = mask.astype('uint8')
        imageio.imsave(os.path.join(args.exp_img_path, 'mask_' + str(index) + '.jpg'), mask)

        relevance = F.interpolate(Res, [64, 64], mode='bilinear')
        relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
        # relevance = Res[0].permute(1, 2, 0).data.cpu().numpy()
        hm = np.sum(relevance, axis=-1)
        maps = (render.hm_to_rgb(hm, scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
        imageio.imsave(os.path.join(args.exp_img_path, 'heatmap_' + str(index) + '.jpg'), maps)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    my_prob  = 100 * prob[0, class_indices[0]]
    if my_prob < 30:
      print("pass")

      op = "pass"
    elif my_prob  > 30 and my_prob < 50:
      print("low")
      print(my_prob.data)

      
      op = "low"
    elif my_prob  > 50 and my_prob < 70:
      print("mid")
      op = "mid"
      print(my_prob.data)

    else:
      print("high")
      op = "high"
      print(my_prob.data)

    print("\n")
    if (100 * prob[0, class_indices[0]]) > 70:
      #update_json("stuff/results.json", {})
      print("ding ding ding \n\n")

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target, op


total_inter_low, total_union_low, total_correct_low, total_label_low = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_inter_mid, total_union_mid, total_correct_mid, total_label_mid = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_inter_high, total_union_high, total_correct_high, total_label_high = np.int64(0), np.int64(0), np.int64(0), np.int64(0)

total_ap, total_f1 = [], []
fileH = open("/content/stuff/numbers.txt", "w")

predictions, targets = [], []

count = 0
 
count_low = 0
count_mid = 0
count_high = 0
lst_low    =[]
lst_mid    =[]
lst_high    =[]

for batch_idx, (image, labels) in enumerate(iterator):
    count+=1
   # if count > 5:
   #   exit(1)

    if args.method == "blur":
        images = (image[0].cuda(), image[1].cuda())
    else:
        images = image.cuda()
    labels = labels.cuda()
    # print("image", image.shape)
    # print("lables", labels.shape)

    correct, labeled, inter, union, ap, f1, pred, target, op = eval_batch(images, labels, model, batch_idx)
  

    predictions.append(pred)
    targets.append(target)

    if batch_idx == 4270:
      update_json('/content/stuff/relu_res.json', {
      
          'mIoU_low': mIoU_low,'pixAcc_low':pixAcc_low,
          'mIoU_mid': mIoU_mid,'pixAcc_mid':pixAcc_mid,
          'mIoU_high': mIoU_high,'pixAcc_high':pixAcc_high,
          'count_low': count_low, 'count_mid': count_mid, 
          'count_high': count_high,
          'lst_low': lst_low, 'lst_mid': lst_mid, 'lst_high': lst_high
      })
      exit(1)

    if op == "pass":
      continue
    #fileH.write(f"{batch_idx}\n")
 

    if op =="low":
      lst_low.append(batch_idx)
      count_low+=1
    
      total_correct_low += correct.astype('int64')
      total_label_low   += labeled.astype('int64')


      total_inter_low += inter.astype('int64')
      total_union_low += union.astype('int64')
      total_ap += [ap]
      total_f1 += [f1]
      pixAcc_low = np.float64(1.0) * total_correct_low / (np.spacing(1, dtype=np.float64) + total_label_low)
      IoU_low = np.float64(1.0) * total_inter_low / (np.spacing(1, dtype=np.float64) + total_union_low)
      mIoU_low = IoU_low.mean()
      mAp = np.mean(total_ap)
      mF1 = np.mean(total_f1)

    elif op == "mid":
      lst_mid.append(batch_idx)

      count_mid +=1
      total_correct_mid += correct.astype('int64')
      total_label_mid   += labeled.astype('int64')


      total_inter_mid += inter.astype('int64')
      total_union_mid += union.astype('int64')
      total_ap += [ap]
      total_f1 += [f1]
      pixAcc_mid = np.float64(1.0) * total_correct_mid / (np.spacing(1, dtype=np.float64) + total_label_mid)
      IoU_mid = np.float64(1.0) * total_inter_mid / (np.spacing(1, dtype=np.float64) + total_union_mid)
      mIoU_mid = IoU_mid.mean()
      mAp = np.mean(total_ap)
      mF1 = np.mean(total_f1)

    elif op =="high":
      lst_high.append(batch_idx)

      count_high +=1
      total_correct_high += correct.astype('int64')
      total_label_high   += labeled.astype('int64')


      total_inter_high += inter.astype('int64')
      total_union_high += union.astype('int64')
      total_ap += [ap]
      total_f1 += [f1]
      pixAcc_high = np.float64(1.0) * total_correct_high / (np.spacing(1, dtype=np.float64) + total_label_high)
      IoU_high = np.float64(1.0) * total_inter_high / (np.spacing(1, dtype=np.float64) + total_union_high)
      mIoU_high = IoU_high.mean()
      mAp = np.mean(total_ap)
      mF1 = np.mean(total_f1)
    iterator.set_description(f'ITER:{batch_idx}')

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)
pr, rc, thr = precision_recall_curve(targets, predictions)
np.save(os.path.join(saver.experiment_dir, 'precision.npy'), pr)
np.save(os.path.join(saver.experiment_dir, 'recall.npy'), rc)

plt.figure()
plt.plot(rc, pr)
plt.savefig(os.path.join(saver.experiment_dir, 'PR_curve_{}.png'.format(args.method)))

txtfile = os.path.join(saver.experiment_dir, 'result_mIoU_%.4f.txt' % mIoU)
# txtfile = 'result_mIoU_%.4f.txt' % mIoU
fh = open(txtfile, 'w')
print("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
print("Mean AP over %d classes: %.4f\n" % (2, mAp))
print("Mean F1 over %d classes: %.4f\n" % (2, mF1))

fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
fh.write("Mean F1 over %d classes: %.4f\n" % (2, mF1))
fh.close()
































{
    "transformer_attribution_pos_top": [
        0.465,
        0.28,
        0.195,
        0.135,
        0.1,
        0.06,
        0.045,
        0.055,
        0.07
    ],
    "transformer_attribution_pos_auc_top": 0.11374999999999999,
    "transformer_attribution_pos_target": [
        0.42,
        0.255,
        0.175,
        0.115,
        0.085,
        0.05,
        0.035,
        0.04,
        0.06
    ],
    "transformer_attribution_pos_auc_target": 0.09949999999999999,
    "transformer_attribution_pos_top_blur": [
        0.53,
        0.375,
        0.32,
        0.24,
        0.155,
        0.1,
        0.085,
        0.08,
        0.06
    ],
    "transformer_attribution_pos_auc_top_blur": 0.165,
    "transformer_attribution_pos_target_blur": [
        0.52,
        0.35,
        0.29,
        0.21,
        0.135,
        0.085,
        0.08,
        0.065,
        0.045
    ],
    "transformer_attribution_pos_auc_target_blur": 0.14975,
    "transformer_attribution_neg_top": [
        0.885,
        0.84,
        0.84,
        0.76,
        0.62,
        0.485,
        0.34,
        0.26,
        0.195
    ],
    "transformer_attribution_neg_auc_top": 0.4685,
    "transformer_attribution_neg_target": [
        0.77,
        0.725,
        0.73,
        0.675,
        0.555,
        0.44,
        0.31,
        0.24,
        0.175
    ],
    "transformer_attribution_neg_auc_target": 0.41475,
    "transformer_attribution_neg_top_blur": [
        0.96,
        0.93,
        0.89,
        0.845,
        0.8,
        0.785,
        0.715,
        0.63,
        0.495
    ],
    "transformer_attribution_neg_auc_top_blur": 0.63225,
    "transformer_attribution_neg_target_blur": [
        0.795,
        0.79,
        0.78,
        0.745,
        0.76,
        0.735,
        0.655,
        0.595,
        0.485
    ],
    "transformer_attribution_neg_auc_target_blur": 0.5700000000000001
}






































#for simplified blocsks





{
    "transformer_attribution_pos_top": [
        0.3992,
        0.224,
        0.1544,
        0.1136,
        0.0824,
        0.0592,
        0.044,
        0.0368,
        0.0296
    ],
    "transformer_attribution_pos_auc_top": 0.09288000000000002,
    "transformer_attribution_pos_target": [
        0.3896,
        0.2176,
        0.1488,
        0.1064,
        0.0832,
        0.0592,
        0.0432,
        0.0352,
        0.0312
    ],
    "transformer_attribution_pos_auc_target": 0.09039999999999998,
    "transformer_attribution_pos_top_blur": [
        0.5024,
        0.3248,
        0.2376,
        0.18,
        0.1384,
        0.1128,
        0.1024,
        0.0784,
        0.0704
    ],
    "transformer_attribution_pos_auc_top_blur": 0.14608,
    "transformer_attribution_pos_target_blur": [
        0.4904,
        0.3112,
        0.2232,
        0.1672,
        0.1264,
        0.1,
        0.0944,
        0.072,
        0.0616
    ],
    "transformer_attribution_pos_auc_target_blur": 0.13704,
    "transformer_attribution_neg_top": [
        0.8984,
        0.8488,
        0.8032,
        0.7056,
        0.5944,
        0.472,
        0.376,
        0.2824,
        0.2136
    ],
    "transformer_attribution_neg_auc_top": 0.46384000000000003,
    "transformer_attribution_neg_target": [
        0.7768,
        0.7432,
        0.72,
        0.6432,
        0.5488,
        0.4392,
        0.356,
        0.2656,
        0.2048
    ],
    "transformer_attribution_neg_auc_target": 0.42068000000000005,
    "transformer_attribution_neg_top_blur": [
        0.976,
        0.9584,
        0.944,
        0.9184,
        0.8696,
        0.844,
        0.7896,
        0.6968,
        0.5536
    ],
    "transformer_attribution_neg_auc_top_blur": 0.67856,
    "transformer_attribution_neg_target_blur": [
        0.8064,
        0.7992,
        0.7936,
        0.7864,
        0.7576,
        0.736,
        0.7016,
        0.6248,
        0.4992
    ],
    "transformer_attribution_neg_auc_target_blur": 0.5851999999999999
}