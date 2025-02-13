from modules.layers_ours import *
from functools import partial

DEFAULT_MODEL = {
    'norm'                 : partial(LayerNorm, eps=1e-6) ,
    'last_norm'            : LayerNorm ,
    'activation'           : GELU(),
    'isWithBias'           : True,
    'attn_activation'      : Softmax(dim=-1),
    'attn_drop_rate'       : 0.,
    'FFN_drop_rate'        : 0.,
    'projection_drop_rate' : 0.,
    'reg_coeffs'           : None,          

}




DEFAULT_PATHS = {
        
    'imagenet_1k_Dir'        : '/home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/',
    'imagenet_100_Dir'       : './',
    'finetuned_models_dir'   : 'finetuned_models/', 
    'results_dir'            : 'finetuned_models/', 

}


MODEL_VARIANTS = {
            'basic'                          :  DEFAULT_MODEL.copy(),
            'bias_ablation'                  :  {**DEFAULT_MODEL, 'isWithBias': False, },
            #Attention Activation Variants
            'attn_act_relu'                  :  {**DEFAULT_MODEL, 'attn_activation': ReluAttention()},
            'attn_act_relu_normalized'       :  {**DEFAULT_MODEL, 'attn_activation': NormalizedReluAttention()},
            'attn_act_relu_no_cp'            :  {**DEFAULT_MODEL, 'attn_activation': ReluAttention()},
            'variant_relu_softmax'           :  {**DEFAULT_MODEL,},
            'attn_act_sigmoid'               :  {**DEFAULT_MODEL, 'attn_activation': SigmoidAttention()},
            'attn_act_sparsemax'             :  {**DEFAULT_MODEL, 'attn_activation': Sparsemax(dim=-1)},
            'attn_variant_light'             :  {**DEFAULT_MODEL,},
            'attn_act_relu_pos'              :  {**DEFAULT_MODEL, 'attn_activation': ReluAttention(), 'activation': Softplus(), 'isWithBias': False, },
            'variant_layer_scale_relu_attn'  :  {**DEFAULT_MODEL, 'attn_activation': ReluAttention()},

            #Activation Variants
            'act_softplus'                   :  {**DEFAULT_MODEL, 'activation': Softplus()},
            'act_relu'                       :  {**DEFAULT_MODEL, 'activation': ReLU()},

            #Normalization Variants
            'act_softplus_norm_rms'          :  {**DEFAULT_MODEL, 'activation': Softplus(), 'norm': partial(RMSNorm, eps=1e-6), 'last_norm': RMSNorm },
            'norm_rms'                       :  {**DEFAULT_MODEL, 'norm': partial(RMSNorm, eps=1e-6), 'last_norm': RMSNorm },
            'norm_bias_ablation'             :  {**DEFAULT_MODEL, 'norm': partial(UncenteredLayerNorm, eps=1e-6, has_bias=False), 
                                       'last_norm': partial(UncenteredLayerNorm,has_bias=False)},
            'norm_center_ablation'           :  {**DEFAULT_MODEL, 'norm': partial(UncenteredLayerNorm, eps=1e-6, center=False),
                                       'last_norm': partial(UncenteredLayerNorm,center=False)},
            'norm_batch'                     :  {**DEFAULT_MODEL, 'norm': RepBN,'last_norm' : RepBN},

            #Special Variants
            'variant_layer_scale'             : {**DEFAULT_MODEL,},
            'variant_diff_attn'               : {**DEFAULT_MODEL,},
            'variant_diff_attn_relu'          : {**DEFAULT_MODEL,'attn_activation': ReluAttention(), 'norm': partial(RMSNorm, eps=1e-6), 'last_norm': RMSNorm },

            'variant_weight_normalization'    : {**DEFAULT_MODEL,},
            'variant_sigmaReparam_relu'       : {**DEFAULT_MODEL,'attn_activation': ReluAttention()},
            'variant_sigmaReparam'            : {**DEFAULT_MODEL,},


            'variant_more_ffn'                : {**DEFAULT_MODEL,},
            'variant_more_ffnx4'              : {**DEFAULT_MODEL,},     
            'variant_more_attn'               : {**DEFAULT_MODEL,},
            'variant_more_attn_relu'          : {**DEFAULT_MODEL, 'attn_activation': ReluAttention()},


            'variant_simplified_blocks'       : {**DEFAULT_MODEL,},
            'variant_registers'               : {**DEFAULT_MODEL,},
            'variant_proposed_solution'       : {**DEFAULT_MODEL,'attn_activation': ReluAttention()},
            
            #dropout
            'variant_dropout'                 :  {**DEFAULT_MODEL, 'attn_drop_rate': 0.2, 'FFN_drop_rate':0.4, 'projection_drop_rate': 0.4  },
            'variant_dropout_ver2'            :  {**DEFAULT_MODEL, 'attn_drop_rate': 0.4, 'projection_drop_rate': 0.4},
            
            'dropout_layerdrop'               :  {**DEFAULT_MODEL, 'layer_drop_rate': 0.5, 'head_drop_rate': 0., 'attn_drop_rate': 0., 'projection_drop_rate': 0.},
            'dropout_headdrop'                :  {**DEFAULT_MODEL, 'layer_drop_rate': 0., 'head_drop_rate': 0.3,  'attn_drop_rate': 0.0, 'projection_drop_rate': 0.0},
            'dropout_remove_most_important'   :  {**DEFAULT_MODEL,'layer_drop_rate': 0., 'head_drop_rate': 0., },

            #patch embedding
            'variant_patch_embed'             :  {**DEFAULT_MODEL,},  
            'variant_patch_embed_relu'        :  {**DEFAULT_MODEL,'attn_activation': ReluAttention()},               

            #regularization
            'variant_l2_loss'                  :  {**DEFAULT_MODEL, 'reg_coeffs' : [0.85,0.15]}, 

            #drop high norms
            'variant_drop_high_norms_preAct'       : {**DEFAULT_MODEL, 'postActivation': False},

            'variant_drop_high_norms_postAct'       : {**DEFAULT_MODEL,'postActivation': True},
            'variant_drop_high_norms_relu'          : {**DEFAULT_MODEL, 'attn_activation': ReluAttention(), 'postActivation': True},
}


#chosen randmoly
EPOCHS_TO_PERTURBATE = {

    'IMNET100': {
        
            'basic':  [29, 28, 26, 22,10, 14]    ,       # ,12,16, 18  24,
            'attn_act_relu':       [ 70, 52, 71, 72,73,74,75,  31,  33, 35, 45,],    # 14, 20,
            'act_softplus':       [49, 48,45,46,34, 3]   , # 3 34 ,40
            'act_softplus_norm_rms': [78,79,73,],                 #       60,59,58,50,48,46,44,40
            'norm_rms':           [29,13, 18,19,23, 9],   # 1,2,3,
            'norm_bias_ablation':    [29,26,27,28 ,2, 9, 13,19,23,18,] ,  #  
            'bias_ablation':        [56, 58,56,] ,  #  54,44,47,40,37,33,32, 59
            'attn_act_sparsemax':   [69, 68, 67 ], # , 66
            'variant_layer_scale': [203,255],
            'attn_variant_light':  [99],
            'variant_more_ffn' :    [45],
            'variant_more_attn' :    [45],
            'variant_simplified_blocks': [220,210],
            },


    'IMNET' : {
        'basic':          [0],
        'norm_batch':     [0],
        'attn_act_relu': [14],
        'softplus':      [18],
        'norm_rms':       [4]
    }
            
}


EPOCHS_TO_SEGMENTATION = {
    'basic':          [0],
    'norm_batch':     [0],
    'attn_act_relu': [53],
    'attn_act_relu_no_cp': [14,20,50,53,10,12,30,40,48],
    'variant_more_ffn': [155],
    'variant_weight_normalization': [235],
    'attn_act_sigmoid': [65,60,55],
    'variant_simplified_blocks': [35,20],
    'act_softplus': [18],
    'norm_rms': [4]
    #'softplus':      [18],
    #'norm_rms':       [4]      
}



DEFAULT_PARAMETERS = {
    'model'                  : 'deit_tiny_patch16_224',
    'seed'                   : 0,
    'lr'                     : 5e-6, 
    'min_lr'                 : 1e-5,
    'warmup-lr'              : 1e-5,
    'drop_path'              : 0.0,
    'weight_decay'           : 1e-8,
    'num_workers'            : 4,
    'batch_size'             : 128,
    'warmup_epochs'          : 1
}

PRETRAINED_MODELS_URL = {
    'IMNET100': 'finetuned_models/IMNET100/basic/best_checkpoint.pth',
    'IMNET': 'finetuned_models/IMNET/basic/checkpoint_0.pth',


}


def set_components_custom_lrp(args):
    if args.method == "attribution_with_detach":
        args.cp_rule = False
        args.model_components['norm'] = partial(CustomLRPLayerNorm, eps=1e-6) 
        args.model_components['last_norm'] = CustomLRPLayerNorm

        if args.variant == "norm_batch":
            args.model_components['norm']      = partial(RepBN, batchLayer = CustomLRPBatchNorm)
            args.model_components['last_norm'] = partial(RepBN, batchLayer = CustomLRPBatchNorm)

        if args.variant == "norm_rms":
            args.model_components['norm'] = partial(CustomLRPRMSNorm, eps=1e-6)  
            args.model_components['last_norm'] = CustomLRPRMSNorm


    if "custom_lrp" in args.method:
        
        print(f"inside config with custom_lrp")
        if args.variant == "norm_batch":
            args.model_components['norm']      = partial(RepBN, batchLayer = CustomLRPBatchNorm)
            args.model_components['last_norm'] = partial(RepBN, batchLayer = CustomLRPBatchNorm)


            args.cp_rule = True
            return


        args.model_components['norm'] = partial(CustomLRPLayerNorm, eps=1e-6) 
        args.model_components['last_norm'] = CustomLRPLayerNorm

        if ('relu' in args.variant) and (args.variant != 'attn_act_relu') and (args.variant != 'act_relu'):
            args.cp_rule = False
            return

        if args.variant == "norm_rms":
            args.model_components['norm'] = partial(CustomLRPRMSNorm, eps=1e-6)  
            args.model_components['last_norm'] = CustomLRPRMSNorm

        args.cp_rule = True

    else:
        args.cp_rule = False




def SET_VARIANTS_CONFIG(args):
    if args.variant not in MODEL_VARIANTS:
        print(f"only allowed to use the following variants: {MODEL_VARIANTS.keys()}")
        exit(1)
    
    
    args.model_components = MODEL_VARIANTS[args.variant]



def SET_PATH_CONFIG(args):

    
    if args.data_set == 'IMNET100':
        args.data_path = args.dirs['imagenet_100_Dir']
    else:
        args.data_path = args.dirs['imagenet_1k_Dir']


   


def get_config(args, skip_further_testing = False, get_epochs_to_perturbate = False, get_epochs_to_segmentation = False):

    SET_VARIANTS_CONFIG(args)
    
    #if args.custom_lrp:
    #    set_components_custom_lrp(args)
    args.dirs = DEFAULT_PATHS

    
    if get_epochs_to_perturbate:
        args.epochs_to_perturbate = EPOCHS_TO_PERTURBATE
        
    if get_epochs_to_segmentation:
        args.epochs_to_segmentation = EPOCHS_TO_SEGMENTATION

   # if skip_further_testing == False:
   #     vars(args).update(DEFAULT_PARAMETERS)


    

    if args.data_path == None:
        SET_PATH_CONFIG(args)

    if skip_further_testing:
        return
    
    if args.auto_start_train:
        args.finetune =  PRETRAINED_MODELS_URL[args.data_set]
      

    if args.eval and args.resume =='' and args.auto_resume == False:
        print("for evaluation please add --resume  with your model path, or add --auto-resume to automatically find it ")
        exit(1)
    if args.verbose:
        print(f"working with model {args.model} | dataset: {args.data_set} | variant: {args.variant}")


    
