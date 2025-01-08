

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
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.fc1 = Linear(in_features, hidden_features, bias = isWithBias)
        self.act = activation
        self.fc2 = Linear(hidden_features, out_features, bias = isWithBias)
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
                isWithBias      = True, 
                depth = 0,
             ):
        
        super().__init__()

        print(f"inside attention with activation : {attn_activation} | bias: {isWithBias} ")
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
        self.depth = depth

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)


    
        
        for i in range(3):
          head_idx = i
          start_idx = head_idx * head_dim
          end_idx = (head_idx + 1) * head_dim

          # Create new linear layer for this head
          #head_proj = Linear(self.dim, self.head_dim * 3, bias=qkv_bias)

          # Extract Q, K, V weights for this head
          q_weight = self.qkv.weight[start_idx:end_idx].view(head_dim, dim)
          k_weight = self.qkv.weight[dim+start_idx:dim+end_idx].view(head_dim, dim)
          v_weight = self.qkv.weight[2*dim+start_idx:2*dim+end_idx].view(head_dim, dim)
          combined_weights = torch.cat([q_weight, k_weight, v_weight], dim=0)  # Shape will be (head_dim*3, dim)
          if i == 0:
            self.qkv_h1 =   Linear(dim, head_dim * 3, bias=qkv_bias)
            self.qkv_h1.weight.data = combined_weights
           # self.qkv_h1.weight.data[:head_dim] = q_weight
           # self.qkv_h1.weight.data[head_dim: 2*head_dim] = k_weight
           # self.qkv_h1.weight.data[2*head_dim:] = v_weight

          elif i == 1:
            self.qkv_h2 =   Linear(dim, head_dim * 3, bias=qkv_bias)

            self.qkv_h2.weight.data[:head_dim] = q_weight
            self.qkv_h2.weight.data[head_dim: 2*head_dim] = k_weight
            self.qkv_h2.weight.data[2*head_dim:] = v_weight

          else:
            self.qkv_h3 =   Linear(dim, head_dim * 3, bias=qkv_bias)

            self.qkv_h3.weight.data[:head_dim] = q_weight
            self.qkv_h3.weight.data[head_dim: 2*head_dim] = k_weight
            self.qkv_h3.weight.data[2*head_dim:] = v_weight


   
          # Handle biases if needed
          if isWithBias and qkv_bias:
              q_bias = self.qkv.bias[start_idx:end_idx]
              k_bias = self.qkv.bias[dim+start_idx:dim+end_idx]
              v_bias = self.qkv.bias[2*dim+start_idx:2*dim+end_idx]
              
              if i == 0:
                self.qkv_h1.bias.data[:head_dim] = q_bias
                self.qkv_h1.bias.data[head_dim: 2*head_dim] = k_bias
                self.qkv_h1.bias.data[2*head_dim:] = v_bias

              elif i==1:
                self.qkv_h2.bias.data[:head_dim] = q_bias
                self.qkv_h2.bias.data[head_dim: 2*head_dim] = k_bias
                self.qkv_h2.bias.data[2*head_dim:] = v_bias
              else:
                self.qkv_h3.bias.data[:head_dim] = q_bias
                self.qkv_h3.bias.data[head_dim: 2*head_dim] = k_bias
                self.qkv_h3.bias.data[2*head_dim:] = v_bias
          #print(combined_weight1)
          #print(combined_bias1.shape)
        #print("\n")
        
 

        #print(self.v_proj.weight.data)
        #print(self.v_proj.weight.shape)

       
        #print((self.qkv.weight.data[0:self.head_dim] == self.qkv_h1.weight.data[0:self.head_dim]).all())
  

        #print((self.qkv.bias.data[0:self.head_dim] == self.qkv_h1.bias.data[0:self.head_dim]).all())

    
        #print((self.qkv.bias.data[0:self.head_dim] == self.qkv_h1.bias.data[0:self.head_dim]).all())

  
        v_weight = self.qkv.weight[dim*2:dim*3].view(dim, dim)
        self.v_proj = Linear(dim, dim, bias=qkv_bias)
        self.v_proj.weight.data = v_weight

        if isWithBias:
            v_bias   = self.qkv.bias[dim*2:dim*3]
            self.v_proj.bias.data = v_bias

      

        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim, bias = isWithBias)
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
        


        #done only for hook
        tmp = self.v_proj(x)
   
        #t = x.clone()
        tmp = self.qkv_h1(x)
        tmp2 = self.qkv_h2(x)
        tmp3 = self.qkv_h3(x)







        #print((self.qkv.bias.data[0:self.head_dim] == self.qkv_h1.bias.data[0:self.head_dim]).all())
       # print("\n")
       # print("sadasdasdasdaSADJASKDJAKSDJSAKD")
      #  print(self.qkv_h1.weight.data[self.head_dim*2:])
       # print(self.qkv_h1.weight.shape)

        #print(self.v_proj.weight.data)
        #print(self.v_proj.weight.shape)

      #  print(self.qkv.weight.data[2*self.dim : 2*self.dim  + self.head_dim])
        #print(self.qkv.weight.data.shape)
     #   print("\n")
        
 


        #print(tmp)
        #print(tmp.shape)
        #print(qkv[0:self.head_dim])
        #print(qkv.shape)


     #   exit(1)
        #######


        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale
       
        attn = self.attn_activation(dots)



                #attention_variance  = attn.var(dim=[2, 3], keepdim=False)
        #attention_variance_mean = attention_variance.mean(dim=1)
        #print(attention_variance_mean.shape)
#
        #threshold = 0.1  # Example threshold, you can tune this based on your needs
        #condensed_attention_mask = attention_variance_mean  < threshold
        #condensed_patch_indices = torch.nonzero(condensed_attention_mask)
#
        #other_patches_mask = ~condensed_attention_mask
        #print(condensed_attention_mask.shape)
        #mean_other_patches_mask = attn[condensed_attention_mask].mean(dim=3, keepdim=True)
#
#
        #attn_replica= attn.clone()
        #attn_replica[condensed_attention_mask] *= 0.5
#
#
        #attn_replica[other_patches_mask] += mean_other_patches_mask
#
        #attn = attn_replica



        '''
        token_norms = torch.norm(v, p=2, dim=-1)  # [batch_size, num_heads, num_tokens]
    
        # Find threshold for each head separately
        thresholds = torch.quantile(token_norms, 98/100, dim=-1, keepdim=True)  # [batch_size, num_heads, 1]

        # Create masks for high and low magnitude tokens
        high_magnitude_mask = token_norms > thresholds  # [batch_size, num_heads, num_tokens]
        low_magnitude_mask = ~high_magnitude_mask

        # Expand masks to match v_proj dimensions
        high_magnitude_mask = high_magnitude_mask.unsqueeze(-1).expand_as(v)
        low_magnitude_mask = low_magnitude_mask.unsqueeze(-1).expand_as(v)

        # Create modified version of v_proj
        v_proj_balanced = v.clone()

        # Reduce high magnitude tokens
        v_proj_balanced[high_magnitude_mask] *= 0.01

        # Increase low magnitude tokens
        v_proj_balanced[low_magnitude_mask] *= 4.2

        v = v_proj_balanced
        
        '''


        '''
        attention_maps = attn.squeeze(0).data.cpu().numpy()
    
        # Create a heatmap for each head
        for head_idx in range(attention_maps.shape[0]):
            plt.figure(figsize=(10, 8))

            # Create heatmap using seaborn
            sns.heatmap(
                attention_maps[head_idx],
                cmap='viridis',
                cbar=True,
                xticklabels=False,
                yticklabels=False
            )

            plt.title(f'Attention Map - Head {head_idx}')

            # Save the figure
            save_path =  f'/content/The-Explainable-Transformer/testing_vis3/attn_{self.depth}_{head_idx}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        '''
        '''
        if 10>= self.depth >= 1:
          for head in range(3):
            column_norms = torch.norm(attn[:, head, :, :], p=2, dim=1)
            _, top_k_indices = torch.topk(column_norms, k=5, dim=1)
            for batch in range(1):
              attn[batch, head, :, top_k_indices[batch]] = 0
        '''



        '''

        if 10>= self.depth >=1  :   
          l2_norms = torch.norm(attn, dim=2)  # [B, H, N]
          # Random number of patches per batch and head
          num_patches = torch.randint(2, 6, (b, h), device=attn.device)
          mask = torch.ones_like(attn)
          _, top_indices = torch.topk(l2_norms, k=5, dim=-1)  # [B, H, 5]
          batch_idx = torch.arange(b, device=attn.device)[:, None, None]
          head_idx = torch.arange(h, device=attn.device)[None, :, None]

          # Create mask using selected number of patches
          for i in range(5):
              patch_mask = (i < num_patches[:, :, None]).to(attn.dtype)
              mask[batch_idx, head_idx, :, top_indices[:, :, i]] *= (1 - patch_mask)

          # Apply mask and rescale
          attn = attn * mask
        '''


        '''
            if 10>= self.depth >=1  :   
                l2_norms = torch.norm(attn, dim=2)  # [B, H, N]
                # Random number of patches per batch and head
                num_patches = torch.randint(1, 6, (b, h), device=attn.device)
                mask = torch.ones_like(attn)
                _, top_indices = torch.topk(l2_norms, k=5, dim=-1)  # [B, H, 5]
                batch_idx = torch.arange(b, device=attn.device)[:, None, None]
                head_idx = torch.arange(h, device=attn.device)[None, :, None]

                mask = torch.ones_like(attn)
 
                patch_range = torch.arange(5, device=attn.device)
                patch_mask = (patch_range[None, None, :] < num_patches[:, :, None])  # [B, H, 5]
    
                for i in range(5):
                    current_indices = top_indices[:, :, i]  # [B, H]
                    mask[batch_idx.squeeze(-1), head_idx.squeeze(-1), :, current_indices] *= (1 - patch_mask[:, :, i:i+1].to(attn.dtype))
                            # Apply mask and rescale
                attn = attn * mask
        '''

        attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam = None,cp_rule = False, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)
        print(f"cam: {cam.shape}"  )
        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        print(f"cam1: {cam1.shape}"  )

        print(f"cam_v: {cam_v.shape}"  )



        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
      
        cam1 = self.attn_activation.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2


        print(f"cam_q: {cam_q.shape}"  )
        print(f"cam_k: {cam_q.shape}"  )



        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)
        print(f"cam_qkv: {cam_qkv.shape}"  )
        
        head_dim = self.head_dim
        dim      = self.dim
        start_idx = 0 * head_dim
        end_idx = 1 * head_dim

        q_h1 = cam_qkv[:,:,start_idx: end_idx]
        k_h1 =  cam_qkv[:,:,dim+start_idx:dim+end_idx] 
        v_h1 =  cam_qkv[:,:,2*dim+start_idx:2*dim+end_idx] 
        cam_qkv_h1 = torch.cat([q_h1, k_h1, v_h1], dim=-1)

        start_idx = 1 * head_dim
        end_idx = 2 * head_dim
        q_h2 = cam_qkv[:,:,start_idx: end_idx]
        k_h2 =  cam_qkv[:,:,dim+start_idx:dim+end_idx] 
        v_h2 =  cam_qkv[:,:,2*dim+start_idx:2*dim+end_idx] 
        cam_qkv_h2 = torch.cat([q_h2, k_h2, v_h2],  dim=-1)

        start_idx = 2 * head_dim
        end_idx = 3 * head_dim
        q_h3 = cam_qkv[:,:,start_idx: end_idx]
        k_h3 =  cam_qkv[:,:,dim+start_idx:dim+end_idx] 
        v_h3 =  cam_qkv[:,:,2*dim+start_idx:2*dim+end_idx] 
        cam_qkv_h3 = torch.cat([q_h3, k_h3, v_h3],  dim=-1)

      

        v_proj_map = cam_qkv[:,:,384:]
       
        if False:
            return self.qkv_h2.relprop(cam_qkv_h2, **kwargs) 

        
        if cp_rule:
            return self.v_proj.relprop(v_proj_map, **kwargs) 
        else:
            return self.qkv.relprop(cam_qkv, **kwargs)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,   
                isWithBias = True,
                layer_norm = partial(LayerNorm, eps=1e-6),
                activation = GELU,
                attn_activation = Softmax(dim=-1),
                depth = 0
             ):
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
            depth = depth,
          
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

    def relprop(self, cam = None, cp_rule = False, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
       
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2,cp_rule=cp_rule, **kwargs)
      
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

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

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
                last_norm       = LayerNorm,
               ):
        
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
                attn_activation = attn_activation,
                depth = i
               )
            for i in range(depth)])

        self.norm = safe_call(last_norm, normalized_shape= embed_dim, bias = isWithBias ) 
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes, 0., isWithBias, activation)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head = Linear(embed_dim, num_classes, bias = isWithBias)

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

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        '''
        token_norms = torch.norm(x, p=2, dim=-1)
        _, top_k_indices = torch.topk(token_norms, k=5, dim=1)

        x[:,top_k_indices[0,1],:] *= 0.1
        '''

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)
     
        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        return x

    def relprop(self, cam=None,method="transformer_attribution",thr = 100000, cp_rule = False, is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.pool.relprop(cam, **kwargs)
     
        cam = self.norm.relprop(cam, **kwargs)
        #print(f"cam shape start: {cam.shape}")

        count = 0
        for blk in reversed(self.blocks):
            count +=1
            if count > thr:
              break
            cam = blk.relprop(cam,cp_rule = cp_rule, **kwargs)
            #print(f"cam shape: {cam.shape}")

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method   == "custom_lrp":
            cam = cam[0, 1:, :]
            #FIXME: slight tradeoff between noise and intensity of important features
            #cam = cam.clamp(min=0)
            norms = torch.norm(cam, p=2, dim=1)  # Shape: [196]
            return norms

        elif method == "full":
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
                          attn_drop_rate  = 0.,
                          FFN_drop_rate   = 0.,
                          projection_drop_rate = 0.,
                        
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





'''
   def calculate_norm_disparity_loss(self):
        """
        Calculate regularization loss that penalizes patches with much higher norms than others.
        Specifically targets outlier patches with unusually high L2 norms.
        """
        if self.patch_embeddings is None:
            return 0.0
        
        # Calculate L2 norms for each patch
        patch_norms = torch.norm(self.patch_embeddings, p=2, dim=-1)  # [B, N]
        
        # Find the threshold norm value at the specified percentile
        threshold = torch.quantile(patch_norms, 95/100.0, dim=-1, keepdim=True)
        
        # Calculate how much each patch's norm exceeds the threshold
        excess_norms = torch.relu(patch_norms - threshold)
        
        # Square the excess to more heavily penalize larger deviations
        loss = torch.mean(excess_norms ** 2)
        
        return 0.2 * loss


'''