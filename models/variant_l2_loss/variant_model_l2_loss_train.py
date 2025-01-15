

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
                depth = 0):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
   

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim, bias = isWithBias)
        self.proj_drop = Dropout(proj_drop)
        self.attn_activation = attn_activation
        self.actLoss = Softmax(dim=-1)


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


        token_norms = torch.norm(attn, p=2, dim=2)
        #print(token_norms.shape)
        token_norms = self.actLoss(token_norms)  # [B, N]
        #print(token_norms)
        thresholds = torch.quantile(token_norms, 92/100, dim=-1, keepdim=True)
 
        excess_norms = torch.relu(token_norms - thresholds)
        #print(f"excess_norms {excess_norms.shape}") #([1, 3, 197])

        loss = torch.sum(excess_norms ,dim = -1) #([1, 3])
        #print(f"layer {self.depth} loss {loss}")
        
        loss = torch.mean(loss,dim = -1)
        #print(f"layer {self.depth} loss {loss}")
     
  
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        #attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, loss 

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

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., projection_drop_rate = 0.,  
                isWithBias = True,
                layer_norm = partial(LayerNorm, eps=1e-6),
                activation = GELU,
                attn_activation = Softmax(dim=-1),
                depth = 0 ):
        super().__init__()

        self.norm1 = safe_call(layer_norm, normalized_shape= dim, bias = isWithBias ) 
        self.attn = Attention(
            dim, num_heads  = num_heads, 
            qkv_bias        = qkv_bias, 
            attn_drop       = attn_drop, 
            proj_drop       = projection_drop_rate, 
            attn_activation = attn_activation,
            isWithBias      = isWithBias,
            depth           = depth,
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
        attn_output, loss_norm = self.attn(self.norm1(x2))
        x = self.add1([x1, attn_output])
        x1, x2 = self.clone2(x, 2)
      
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x, loss_norm

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
                 projection_drop_rate = 0.,
                isWithBias = True,
                layer_norm = partial(LayerNorm, eps=1e-6),
                activation = GELU,
                attn_activation = Softmax(dim=-1),
                last_norm       = LayerNorm,):
        
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.isWithBias = isWithBias
        self.actLoss = Softmax(dim=-1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, projection_drop_rate = projection_drop_rate,         
           
                isWithBias      = isWithBias, 
                layer_norm      = layer_norm,
                activation      = activation,
                attn_activation = attn_activation,
                depth = i)
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def calculate_norm_disparity_loss(self):
        """
        Calculate regularization loss that penalizes patches with much higher norms than others.
        Specifically targets outlier patches with unusually high L2 norms.
        """
        if self.patch_embeddings is None:
            return 0.0
        
        # Calculate L2 norms for each patch
        patch_norms = torch.norm(self.patch_embeddings, p=2, dim=-1)  # [B, N]
        patch_norms = self.actLoss(patch_norms)  # [B, N]
        
        # Find the threshold norm value at the specified percentile
        threshold = torch.quantile(patch_norms, 92/100.0, dim=-1, keepdim=True)
        
        # Calculate how much each patch's norm exceeds the threshold
        excess_norms = torch.relu(patch_norms - threshold)
        #print(excess_norms.shape)
        # Square the excess to more heavily penalize larger deviations
        loss = torch.mean(excess_norms ** 2,dim = -1)
        
        return  120 *loss

    def forward(self, x):
        loss_norm = 0 # torch.zeros(x[0], device = x.device)
        B = x.shape[0]
        x = self.patch_embed(x)

        self.patch_embeddings = x
        norm_loss_patch = self.calculate_norm_disparity_loss()
        
        loss_norm += norm_loss_patch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        #x.register_hook(self.save_inp_grad)
        count = 0
        for blk in self.blocks:
            x, blk_loss_norm = blk(x)


            loss_norm +=  (blk_loss_norm * 10)
     
        x = self.norm(x)
        x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device))
        x = x.squeeze(1)
        x = self.head(x)
        loss_norm = loss_norm.mean()

        #if self.training:
        return x, loss_norm
        #return 
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
                          attn_drop_rate  = 0.,
                          FFN_drop_rate   = 0.,
                          projection_drop_rate = 0.,
                          **kwargs):

    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, 
        qkv_bias        = isWithBias, 
        isWithBias      = isWithBias, 
        layer_norm      = layer_norm,
        activation      = activation,
        attn_activation = attn_activation,
        last_norm       = last_norm,
        attn_drop_rate  = attn_drop_rate,
        drop_rate       = FFN_drop_rate,
        projection_drop_rate = projection_drop_rate,
        **kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


