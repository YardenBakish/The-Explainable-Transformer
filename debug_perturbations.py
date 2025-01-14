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
from misc.helper_functions import *

from tqdm import tqdm

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import cv2
normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

imagenet_normalize = transforms.Compose([
    #transforms.Resize(256, interpolation=3),
    #transforms.CenterCrop(224),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

IMPORTANT_SAMPLES =  [15, 7,29,37,54, 59,79,90,104,108,118,141,157,165,177,]
SACO_SAMPLES =  [0,15,28,34,55,86,94,127,    54,56 ,83,85,107]

#54,56 - page 11
#83 85 - 17

def compute_SaCo(mapper,img_idx):
    """
    Salience-guided Faithfulness Coefficient (SaCo).

    Parameters:
    - model: Pre-trained model.
    - explanation_method: Explanation method.
    - x: Input image.
    - K: Number of groups G.

    Returns:
    - F: Faithfulness coefficient.
    """

    # initialize F and totalWeight
    F = 0.0
    totalWeight = 0.0


    # Generate Gi based on the saliency map M(x, ŷ) and K
    # Then compute the corresponding s(Gi) and ∇pred(x, Gi) for i = 1, 2, ..., K
    s_G = mapper[img_idx]['sum_salience']  # the list of s(Gi)
    pred_x_G = mapper[img_idx]['perturbated_predicitions_diff']  # the list of ∇pred(x, Gi)

    # Traverse all combinations of Gi and Gj
    for i in range(10 - 1):
        for j in range(i + 1, 10):
            if pred_x_G[i] >= pred_x_G[j]:
                weight = s_G[i] - s_G[j]
            else:
                weight = -(s_G[i] - s_G[j])

            F += weight
            totalWeight += abs(weight)

    if totalWeight != 0:
        F /= totalWeight
    else:
        raise ValueError("The total weight is zero.")

    return F


# create heatmap from mask on image
def show_cam_on_image(img, mask):
   
    x = np.uint8(255 * mask)

    heatmap = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(batch_idx, original_image, class_index=None):
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



def generate_visualization_LRP(original_image, class_index=None, i=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="full", cp_rule=args.cp_rule, index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    #print(image_transformer_attribution.shape)
    
    image_copy = 255 *image_transformer_attribution
    image_copy = image_copy.astype('uint8')
    Image.fromarray(image_copy, 'RGB').save(f'testing_vis2/img_{i}.png')
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def generate_visualization_custom_LRP(batch_idx, thr, original_image, class_index=None,i=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(),  method="custom_lrp", cp_rule=args.cp_rule, index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(14, 14).unsqueeze(0).unsqueeze(0)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear', align_corners=False)
    transformer_attribution = transformer_attribution.squeeze()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    
    print(f"orig image shape: {original_image.shape}")

    image_transformer_attribution = original_image
    #print(f"image_transformer_attribution shape: {image_transformer_attribution.shape}")

    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    
    image_copy = (255 *image_transformer_attribution).cpu().numpy()
    image_copy = image_copy.astype('uint8')

    image_copy_tensor = torch.from_numpy(image_copy).cuda()

    org_shape = image_copy.shape
    print(transformer_attribution.shape)
    transformer_attribution   = transformer_attribution.reshape(1, -1)
    print(transformer_attribution.shape)

    base_size = 224 * 224
    
    _, idx_pred   = torch.topk(transformer_attribution, int(base_size * 0.007), dim=-1)
    idx_pred = idx_pred.squeeze()  # Remove any extra dimensions
    idx_pred = idx_pred.unsqueeze(0)  # Add channel dimension
    idx_pred = idx_pred.expand(3, -1) 

    image_flat = image_copy_tensor.reshape(3, -1)
    
    # Apply scatter operation to black out pixels
    #_data_pred_pertubarted   = image_flat.scatter_(1, idx_pred, 0)
    _data_pred_pertubarted   = image_flat



    res =  _data_pred_pertubarted.reshape(3, 224, 224).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(res, 'RGB').save(f'testing_vis2/img_{i}.png')




def calculate_predictions_custom_lrp(batch_idx, thr, original_image, class_index=None,i=None, mapper = None):
    
    mapper[batch_idx] = {}
    perturbation_steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    image_transformed = imagenet_normalize(original_image)
    output = model(image_transformed.unsqueeze(0).cuda()).detach()
    class_pred_idx = output.data.topk(1, dim=1)[1][0].tolist()[0]
    prob = torch.softmax(output, dim=1)
    pred_prob = prob[0,class_pred_idx]
    print(f"predicted probability: {pred_prob}")
    print(f"predicted entity: {CLS2IDX[class_pred_idx]}")

    
    
    #mapper[batch_idx]['pred'] = pred_prob
    mapper[batch_idx]['perturbated_predicitions_diff'] = []
    mapper[batch_idx]['sum_salience']                  =  []

    
    transformer_attribution = attribution_generator.generate_LRP(image_transformed.unsqueeze(0).cuda(),  method="custom_lrp", cp_rule=args.cp_rule, index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(14, 14).unsqueeze(0).unsqueeze(0)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear', align_corners=False)
    transformer_attribution = transformer_attribution.squeeze()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    
    

    image_copy = original_image.clone()

    image_copy_tensor = image_copy.cuda()

    org_shape = image_copy.shape
    transformer_attribution   = transformer_attribution.reshape(1, -1)
    base_size = 224 * 224
    
 

    mean_pixel0 = None

    for i in range(1, len(perturbation_steps)):
      _data_pred_pertubarted           = image_copy_tensor.clone()
      _, idx_pred   = torch.topk(transformer_attribution, int(base_size * perturbation_steps[i]), dim=-1)
      idx_pred = idx_pred[:,int(base_size * perturbation_steps[i-1]):]

      print(transformer_attribution.flatten()[idx_pred].sum().item())
      mapper[batch_idx]['sum_salience'].append(transformer_attribution.flatten()[idx_pred].sum().item()) 
      idx_pred = idx_pred.squeeze()  # Remove any extra dimensions

      idx_pred = idx_pred.unsqueeze(0)  # Add channel dimension

      idx_pred = idx_pred.expand(3, -1) 
   

      _data_pred_pertubarted                 = _data_pred_pertubarted.reshape(3, -1)
      if i == 1:
        mean_pixel0 = _data_pred_pertubarted.mean(dim=1,keepdim = True)
        #print( f"mean_pixel.shape {mean_pixel0.shape}")
      print( f"mean_pixel.shape {mean_pixel0.shape}")

      
      mean_pixel = mean_pixel0.expand(-1, idx_pred.size(1))
      _data_pred_pertubarted                 = _data_pred_pertubarted.scatter_(1, idx_pred, mean_pixel)
      _data_pred_pertubarted_image           = _data_pred_pertubarted.clone().reshape(3, 224, 224).permute(1, 2, 0)
      
      print(_data_pred_pertubarted.shape)

      res =  (_data_pred_pertubarted_image.cpu().numpy() * 255).astype('uint8')
      Image.fromarray(res, 'RGB').save(f'testing_vis2/img_b{batch_idx}_{i}.png')
      _data_pred_pertubarted               = _data_pred_pertubarted.reshape(*org_shape)

      _norm_data_pred_pertubarted            = normalize(_data_pred_pertubarted.unsqueeze(0))
 
      out_data_pred_pertubarted              = model(_norm_data_pred_pertubarted)

      prob_pert                    = torch.softmax(out_data_pred_pertubarted, dim=1)
      pred_prob_pert_init_class    = prob_pert[0,class_pred_idx]
      print(f"\tpredicted probability: {pred_prob_pert_init_class}  {CLS2IDX[class_pred_idx]}")

      mapper[batch_idx]['perturbated_predicitions_diff'].append(pred_prob -  pred_prob_pert_init_class)

    





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


  parser.add_argument('--custom-trained-model', 
                        required = True,
                        help='')
  parser.add_argument('--data-set', default='IMNET100', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

  parser.add_argument('--data-path', default='',)
  parser.add_argument('--mode', default='test_saco', choices = ['debug_samples', 'test_saco'])

  
  parser.add_argument('--variant', default = 'basic' , type=str, help="")
  parser.add_argument('--class-index', 
                       # default = "243",
                       type=int,
                        help='') #243 - dog , 282 - cat
  parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                        choices=['transformer_attribution', 'full_lrp', 'custom_lrp'],
                        help='')
      
  
  
  args = parser.parse_args()
  config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)
  config.set_components_custom_lrp(args)



  
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

  model.load_state_dict(checkpoint['model'], strict=False)
  model.cuda()
  model.eval()
  attribution_generator = LRP(model)

  mapper = {}

  basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


  dataset_val = datasets.ImageFolder("/content/dataset/val", transform=basic_transform)
  
     
  np.random.seed(42)
  torch.manual_seed(42)
  total_size  = len(dataset_val)
  indices = list(range(total_size))
  subset_size = int(total_size *0.04)
  random_indices = np.random.choice(indices, size=subset_size, replace=False)
  sampler = SubsetRandomSampler(random_indices)  
  #first 0.1
  '''total_size  = len(dataset_val)
  subset_size = int(total_size * 0.1)
  indices     = list(range(subset_size))
  dataset_val = Subset(dataset_val, indices)'''  
  #sampler_val = torch.utils.data.SequentialSampler(dataset_val)  
  #imagenet_ds = ImageNet(args.imagenet_validation_path, split='val', download=False, transform=transform)
  sample_loader = torch.utils.data.DataLoader(    
      dataset_val, sampler=sampler,
      batch_size=1,
      shuffle=False,
      pin_memory=True,
      num_workers=1,
      drop_last = True  
  )

  count = 0
  for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):

     #if count > 5:
     #   break
  
     if batch_idx not in IMPORTANT_SAMPLES and args.mode == 'debug_samples':
        if batch_idx < max(IMPORTANT_SAMPLES):
          count+=1
          continue
        else:
           break
        
     if batch_idx not in SACO_SAMPLES and args.mode == 'test_saco':
        if batch_idx < max(SACO_SAMPLES):
          count+=1
          continue
        else:
           break
    
     method_name = None
     vis = None
     if args.method == "transformer_attribution":
        #print_top_classes(output)
        generate_visualization(batch_idx, data, args.class_index)
       
     elif args.method == "custom_lrp":

       if args.mode ==  'debug_samples':
        generate_visualization_custom_LRP(batch_idx,1000000, data.squeeze(0), args.class_index,batch_idx)
       if args.mode ==  'test_saco':
        calculate_predictions_custom_lrp(batch_idx,1000000, data.squeeze(0), args.class_index,batch_idx, mapper,)
     else:
       generate_visualization_LRP(data, args.class_index,batch_idx)
      
     count+=1
     
  final_res = {}
  if args.mode ==  'test_saco':
     for num in SACO_SAMPLES:
      F = compute_SaCo(mapper,num)
      final_res[str(num)] = F
     update_json(f"testing/res_{args.variant}.json", 
            final_res)
     print(F)











# i *4 - 1

# 7  - black the patch from beginning | zero relevance from end - both models
# 11,10 , 12, 15 - perfect for zero down relevance from the end (for basic) 
# 21, 20 - both zero down important from end, and black the most important ones from the start
# 25 - crucial for blacking out from the start (how the model behaves when there is no eye)
# 32 - crucial from zero relevnce from end
# 45 - zero relevance from end
# 57 - zero patch from start and from end - I want this to be our solution to overcondesniation - somewhere the model has to understand this is a bird
# 63 - black from start
# 65 - zero down relevance and check if somewhere else we understand its a shark - go over everything (attention, all heads, and lrp)
# 74 - also, check where does the model understands by structure that this is a mouse
# 131 - most extreme example - find out where is the bird
# 139 - also,zero down relevance


