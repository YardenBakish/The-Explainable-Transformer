import os
import config
import argparse
import subprocess
#import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='evaluate perturbations')
    
    parser.add_argument('--mode', required=True, choices = ['segmentations', 'analyze'])
 

    parser.add_argument('--gen-latex', action='store_true')
    parser.add_argument('--otsu-thr', action='store_true')
    parser.add_argument('--variant', default = 'basic',  type=str, help="")
    parser.add_argument('--analyze-comparison', action='store_true')

    parser.add_argument('--check-all', action='store_true')
 
    parser.add_argument('--data-path', type=str,
                  
                        help='')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')

    parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)
    
    parser.add_argument('--num-workers', type=int,
                        default= 1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                        choices=['rollout', 'lrp', 'transformer_attribution', 'full_lrp', 'lrp_last_layer',
                                 'attn_last_layer', 'attn_gradcam', 'custom_lrp'],
                        help='')

 

    parser.add_argument('--imagenet-seg-path', type=str, default = "gtsegs_ijcv.mat",help='')
    args = parser.parse_args()
    return args



MAPPER_HELPER = {
   'basic': 'DeiT-tiny',
   'attn act relu': 'Relu/seqlen',
   'act softplus':   'Softplus Act.',
   'act softplus norm rms': 'Softplus+RMSNorm',
   'norm rms': 'RMSNorm',
   'bias ablation': 'DeiT-tiny w/o Bias',
   'norm bias ablation': 'LayerNorm w/o Bias',
   'attn act sparsemax': 'Sparsemax',
   'variant layer scale': 'DeiT-tiny w/ LayerScale',
   'attn variant light': 'LightNet',
   'variant more ffn': '2XFFN',
   'variant more attn': '2XAttention',
   'variant simplified blocks': 'DeiT-tiny w/o normalization',
   'attn act sigmoid': 'sigmoid attention',
   'attn act relu no cp': 'Relu/seqlen w/o cp',
   'norm batch':           'RepBN (BatchNorm)',
   'custom_lrp': 'lrp',
   'transformer_attribution': 'transformer attribution',
   'variant weight normalization': 'WeightNormalization'
}


def gen_latex_table(global_top_mapper,args):
    latex_code = r'\begin{table}[h!]\centering' + '\n' + r'\begin{tabular}{c c c c c c}' + '\n' + '\hline' +'\n'
    latex_code += '& Pixel Accuracy & mAP & mIoU & mBGI & mFI' r'\\ ' +'\hline \n'
    for elem in global_top_mapper:
      variant,epoch,pixAcc, mAP, mIoU, mBG_I,mFG_I = elem
      variant = variant.split("/")[-1]
      variant = variant.replace("_"," ")
      variant = MAPPER_HELPER[variant]

      row = variant
      row += f' & {pixAcc:.3f}'
      row += f' & {mAP:.3f}'
      row += f' & {mIoU:.3f}'
      row += f' & {mBG_I:.3f}'
      row += f' & {mFG_I:.3f}'
      
      row += r'\\ ' f'\n'

      latex_code += row
    
    latex_code += "\\hline\n\\end{tabular}\n\\caption{Segmentation Results using}\n\\end{table}"

    print(latex_code)





   
   

'''
TEMPORARY! based on current accuarcy results
'''
def filter_epochs(args, epoch, variant):
   return epoch in args.epochs_to_segmentation[variant]



def run_segmentations_env(args):
   choices = args.epochs_to_segmentation.keys()
   for c in choices:
      args.variant = c
      run_segmentation(args)
  



def get_sorted_checkpoints(directory):
    # List to hold the relative paths and their associated numeric values
    checkpoints = []

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file matches the pattern 'checkpoint_*.pth'
            match = re.match(r'checkpoint_(\d+)\.pth', file)
            if match:
                # Extract the number from the filename
                number = int(match.group(1))
                # Get the relative path of the file
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                # Append tuple (number, relative_path)
                checkpoints.append((number, relative_path))

    # Sort the checkpoints by the number
    checkpoints.sort(key=lambda x: x[0])

    # Return just the sorted relative paths
    return [f'{directory}/{relative_path}'  for _, relative_path in checkpoints]


def run_segmentation(args):
    eval_seg_cmd        = "python evaluate_segmentation.py"
   
    
    eval_seg_cmd       +=  f' --method {args.method}'
    eval_seg_cmd       +=  f' --imagenet-seg-path {args.imagenet_seg_path}'
  

    root_dir = f"{args.dirs['finetuned_models_dir']}{args.data_set}"
    
    variant          = f'{args.variant}'
    eval_seg_cmd += f' --variant {args.variant}'

    if args.otsu_thr:
       eval_seg_cmd += " --otsu-thr "
  

    model_dir = f'{root_dir}/{variant}'

    checkpoints =  get_sorted_checkpoints(model_dir)

    suff = (args.method).split("_")[-1]

    for c in checkpoints:
     
       checkpoint_path  = c.split("/")[-1]
       epoch            = checkpoint_path.split(".")[0].split("_")[-1]
       if filter_epochs(args, int(epoch), variant ) == False:
          continue
       print(f"working on epoch {epoch}")
       seg_results_dir =  'seg_results_otsu' if args.otsu_thr else 'seg_results'
       eval_seg_epoch_cmd = f"{eval_seg_cmd} --output-dir {model_dir}/{seg_results_dir}/res_{epoch}_{suff}"
       eval_seg_epoch_cmd += f" --custom-trained-model {model_dir}/{checkpoint_path}" 
       print(f'executing: {eval_seg_epoch_cmd}')
       try:
          subprocess.run(eval_seg_epoch_cmd, check=True, shell=True)
          print(f"generated visualizations")
       except subprocess.CalledProcessError as e:
          print(f"Error: {e}")
          exit(1)




def parse_seg_results(seg_results_path, method,variant):
    suffix = method.split("_")[-1]
    
    lst = []
    
    for res_dir in os.listdir(seg_results_path):
        if suffix not in res_dir:
           continue
        
        epoch = int(res_dir.split('_')[1])

        
        
        seg_results_file = f'{seg_results_path}/{res_dir}/seg_results.json'
        with open(seg_results_file, 'r') as f:
            seg_data = json.load(f)
            mIoU =  seg_data.get(f'mIoU',0)
            pixAcc =  seg_data.get(f'Pixel Accuracy',0)
            mBG_I =  seg_data.get(f'mean_bg_intersection',0)
            mFG_I =  seg_data.get(f'mean_fg_intersection',0)
            mAP   =  seg_data.get(f'mAP',0)
            lst.append((variant,epoch,float(pixAcc), float(mAP), float(mIoU), float(mBG_I),float(mFG_I) ))
    return lst





def parse_subdir(subdir):
   exp_name = subdir.split("/")[-1]
   exp_name = exp_name.replace("_"," ")
   exp_name = exp_name if exp_name != "none" else "basic"
   return exp_name


def analyze(args):
   choices  =  args.epochs_to_segmentation.keys() 
   root_dir = f"{args.dirs['finetuned_models_dir']}{args.data_set}"

   global_lst = []
   seg_results_dir =  'seg_results_otsu' if args.otsu_thr else 'seg_results'

   for c in choices:
       subdir = f'{root_dir}/{c}/{seg_results_dir}'    
       global_lst += parse_seg_results(subdir, args.method,c)
    

   global_lst.sort(reverse=True, key=lambda x: x[4]) #mIOU

   print("oredered by mIOU")
   for elem in global_lst:
      variant,epoch,pixAcc, mAP, mIoU, mBG_I,mFG_I = elem
      print(f"variant: {variant} | e: {epoch} | mIoU: {mIoU} | pixAcc: {pixAcc} | mAP: {mAP}")
   
   
   if args.gen_latex:
      gen_latex_table(global_lst,args)



def analyze_comparison(args):
    filepath = "testing/seg_results_compare.json"
    # Read the JSON file
    with open(filepath, 'r') as file:
       data = json.load(file)
    x_values = [0, 0.2, 0.4, 0.6, 0.8]
    d_res  = {}
    # Loop through the first-level keys (groups)
    for variant, epochs_dict in data.items():
     for epoch, experiments in epochs_dict.items():
        for experiment_name, values in experiments.items(): 
         if experiment_name not in d_res:
             d_res[experiment_name]  ={}
         d_res[experiment_name][f'{variant}_{epoch}'] = values
                # Create a plot for each experiment
    
     for experiment, results in d_res.items():
         plt.figure(figsize=(10, 6))
         plt.title(f"Experiment: {experiment}")
         for test, values  in results.items():
            plt.plot(x_values,values, label=test)
            plt.xlabel('eps')
            plt.ylabel('score')
            plt.legend()
         plt.savefig(f'testing/{experiment}.png')
         plt.close()  

        


   
if __name__ == "__main__":
    args                   = parse_args()
    config.get_config(args, skip_further_testing = True, get_epochs_to_segmentation = True)
    
    if args.analyze_comparison:
       analyze_comparison(args)
       exit(1)
       
    if args.mode == "segmentations":
       if args.check_all:
          run_segmentations_env(args)
       else: 
         run_segmentation(args)
    else:
       analyze(args)
    