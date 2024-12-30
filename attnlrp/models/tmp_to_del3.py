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
from tqdm import tqdm

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import datasets, transforms


import cv2
normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

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
    Image.fromarray(image_copy, 'RGB').save(f'testing_vis/img_{i}.png')
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def generate_visualization_custom_LRP(batch_idx, thr, original_image, class_index=None,i=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), thr, method="custom_lrp", cp_rule=args.cp_rule, index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(14, 14).unsqueeze(0).unsqueeze(0)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear', align_corners=False)
    transformer_attribution = transformer_attribution.squeeze().detach().cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    #transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    #transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    #print(transformer_attribution)
    
    image_copy = 255 *image_transformer_attribution
    image_copy = image_copy.astype('uint8')
    #Image.fromarray(image_copy, 'RGB').save(f'testing_vis/img_{i}.png')
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
                        choices=['transformer_attribution', 'full_lrp', 'custom_lrp'],
                        help='')
      
  
  
  args = parser.parse_args()
  config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)
  config.set_components_custom_lrp(args)

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

  model.load_state_dict(checkpoint['model'], strict=False)
  model.cuda()
  model.eval()
  attribution_generator = LRP(model)

  output = model(image_transformed.unsqueeze(0).cuda())
  print_top_classes(output)

  filename = os.path.basename(args.sample_path)
    # Remove the file extension
  img_name = os.path.splitext(filename)[0]


  dataset_val = datasets.ImageFolder("/content/dataset/val", transform=transform)
  
     
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


  for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):

    
     print("dfd")

  
     method_name = None
     vis = None
     if args.method == "transformer_attribution":
       vis = generate_visualization(batch_idx, data, args.class_index)
       method_name = "Att"
     elif args.method == "custom_lrp":
       for thr in [2,4,6,8,10, 100000000]:
        vis = generate_visualization_custom_LRP(batch_idx,thr, data.squeeze(0), args.class_index,batch_idx)
        method_name = "custom_lrp"
        saved_image_path = f"testing_vis/{batch_idx}_{args.variant}_{thr}.png"
        plt.imsave(saved_image_path, vis)
     else:
       vis = generate_visualization_LRP(data, args.class_index,batch_idx)
       method_name = "lrp"
     
     saved_image_path = f"testing_vis/{batch_idx}_{args.variant}.png"
     
     #plt.imsave(saved_image_path, vis)
     
  



#print_top_classes(output)












'''
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
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules
from zennit.image import imgify


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
    image_transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index)
    #transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    #transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    #transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    #transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    #image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    #image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    #print(transformer_attribution)
   
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



def generate_visualization_LRP(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index)
    #transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    #transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    #transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    #transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    #print(transformer_attribution)
   
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
  print("sdsd")
  



#print_top_classes(output)


'''