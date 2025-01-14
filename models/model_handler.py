from models.model import deit_tiny_patch16_224 as vit_LRP
from models.model_train import deit_tiny_patch16_224 as vit_LRP_train


from models.variant_light_attention.variant_model_light_attn_train import deit_tiny_patch16_224 as model_variant_light_attention_train
from models.variant_light_attention.variant_model_light_attn import deit_tiny_patch16_224 as model_variant_light_attention

from models.variant_layer_scale.variant_model_layer_scale_train import deit_tiny_patch16_224 as model_variant_layer_scale_train
from models.variant_layer_scale.variant_model_layer_scale import deit_tiny_patch16_224 as model_variant_layer_scale

from models.variant_diff_attention.variant_diff_attention_train import deit_tiny_patch16_224 as model_variant_diff_attention_train
from models.variant_diff_attention.variant_diff_attention import deit_tiny_patch16_224 as model_variant_diff_attention

from models.variant_weight_normalization.model_variant_weight_normalization import deit_tiny_patch16_224 as model_variant_weight_normalization
from models.variant_weight_normalization.model_variant_weight_normalization_train import deit_tiny_patch16_224 as model_variant_weight_normalization_train



from models.variant_less_is_more.model_less_is_more_train import deit_tiny_patch16_224 as model_variant_less_is_more_train
from models.variant_less_is_more.model_less_is_more import deit_tiny_patch16_224 as model_variant_less_is_more

from models.variant_simplified_block.variant_model_simplified_block_train import deit_tiny_patch16_224 as variant_model_simplified_block_train
from models.variant_simplified_block.variant_model_simplified_block import deit_tiny_patch16_224 as variant_model_simplified_block

from models.variant_registers.variant_registers_train import deit_tiny_patch16_224 as variant_model_registers_train
from models.variant_registers.variant_registers import deit_tiny_patch16_224 as variant_model_registers


from models.variant_proposed_solution.variant_proposed_solution_train import deit_tiny_patch16_224 as variant_proposed_solution_train
from models.variant_proposed_solution.variant_proposed_solution import deit_tiny_patch16_224 as variant_proposed_solution


from models.variant_relu_softmax.variant_relu_softmax_train import deit_tiny_patch16_224 as variant_relu_softmax_train
from models.variant_relu_softmax.variant_relu_softmax import deit_tiny_patch16_224 as variant_relu_softmax


from models.dropout.model_dropout_train import deit_tiny_patch16_224 as model_dropout_train
from models.dropout.model_dropout import deit_tiny_patch16_224 as model_dropout

from models.variant_patch_embed.variant_patch_embed_train import deit_tiny_patch16_224 as variant_patch_embed_train
from models.variant_patch_embed.variant_patch_embed import deit_tiny_patch16_224 as variant_patch_embed

from models.variant_sigmaReparam.variant_model_sigmaReparam_train import deit_tiny_patch16_224 as variant_model_sigmaReparam_train
from models.variant_sigmaReparam.variant_model_sigmaReparam import deit_tiny_patch16_224 as variant_model_sigmaReparam

from models.variant_l2_loss.variant_model_l2_loss_train import deit_tiny_patch16_224 as variant_model_l2_loss_train
from models.variant_l2_loss.variant_model_l2_loss import deit_tiny_patch16_224 as variant_model_l2_loss



#TODO: add support to remaining variants instead of keeping multiple documents with redundant code

def model_env(pretrained=False,args  = None , hooks = False,  **kwargs):
    
    if 'variant_l2_loss' in args.variant :
        if hooks:
            return variant_model_l2_loss(
            isWithBias           = args.model_components["isWithBias"],
            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],


            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )
        else:
            return variant_model_l2_loss_train(
            isWithBias           = args.model_components["isWithBias"],
            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],


            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )


    
    if 'variant_patch_embed' in args.variant :
        if hooks:
            return variant_patch_embed(
            isWithBias           = args.model_components["isWithBias"],
            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],


            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )
        else:
            return variant_patch_embed_train(
            isWithBias           = args.model_components["isWithBias"],
            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],


            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

    if args.variant == 'variant_relu_softmax':
        if hooks:
            return variant_relu_softmax(
            isWithBias           = args.model_components["isWithBias"],
            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],


            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
            )
        else:
            return variant_relu_softmax_train(
            isWithBias           = args.model_components["isWithBias"],
            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],


            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )



    if 'dropout' in args.variant and 'variant_dropout' not in args.variant:
        remove_most_important = True if args.variant == 'dropout_remove_most_important' else False
        if hooks:
            return model_dropout(
                isWithBias           = args.model_components["isWithBias"],
                layer_norm           = args.model_components["norm"],
                last_norm            = args.model_components["last_norm"],
                attn_drop_rate       = args.model_components["attn_drop_rate"],
                FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                projection_drop_rate = args.model_components['projection_drop_rate'],


                activation      = args.model_components["activation"],
                attn_activation = args.model_components["attn_activation"],
                num_classes     = args.nb_classes,
            )

            
        else:
            return model_dropout_train(
                isWithBias           = args.model_components["isWithBias"],
                layer_norm           = args.model_components["norm"],
                last_norm            = args.model_components["last_norm"],
                attn_drop_rate       = args.model_components["attn_drop_rate"],
                FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                projection_drop_rate = args.model_components['projection_drop_rate'],


                activation      = args.model_components["activation"],
                attn_activation = args.model_components["attn_activation"],
                num_classes     = args.nb_classes,

                layer_drop_rate  = args.model_components["layer_drop_rate"],
                head_drop_rate = args.model_components["head_drop_rate"],
                remove_most_important = remove_most_important
            )
            


    if args.variant == "variant_proposed_solution":
        if hooks:
            return variant_model_simplified_block(
                isWithBias      = args.model_components["isWithBias"],
                layer_norm      = args.model_components["norm"],
                last_norm       = args.model_components["last_norm"],

                activation      = args.model_components["activation"],
                attn_activation = args.model_components["attn_activation"],
                num_classes     = args.nb_classes,
            )
        else:
            return variant_model_simplified_block_train(
                isWithBias      = args.model_components["isWithBias"],
                layer_norm      = args.model_components["norm"],
                last_norm       = args.model_components["last_norm"],

                activation      = args.model_components["activation"],
                attn_activation = args.model_components["attn_activation"],
                num_classes     = args.nb_classes,

            )
    


    if args.variant == "variant_registers":
        if hooks:
            return variant_model_registers(
                isWithBias      = args.model_components["isWithBias"],
                layer_norm      = args.model_components["norm"],
                last_norm       = args.model_components["last_norm"],

                activation      = args.model_components["activation"],
                attn_activation = args.model_components["attn_activation"],
                num_classes     = args.nb_classes,

            )
        else:
            return variant_model_registers_train(
                isWithBias      = args.model_components["isWithBias"],
                layer_norm      = args.model_components["norm"],
                last_norm       = args.model_components["last_norm"],

                activation      = args.model_components["activation"],
                attn_activation = args.model_components["attn_activation"],
                num_classes     = args.nb_classes,

            )

    if args.variant == "variant_simplified_blocks":
        if hooks:
            return variant_model_simplified_block(
                isWithBias      = args.model_components["isWithBias"],
                layer_norm      = args.model_components["norm"],
                last_norm       = args.model_components["last_norm"],

                activation      = args.model_components["activation"],
                attn_activation = args.model_components["attn_activation"],
                num_classes     = args.nb_classes,
            )
        else:
            return variant_model_simplified_block_train(
                isWithBias      = args.model_components["isWithBias"],
                layer_norm      = args.model_components["norm"],
                last_norm       = args.model_components["last_norm"],

                activation      = args.model_components["activation"],
                attn_activation = args.model_components["attn_activation"],
                num_classes     = args.nb_classes,

            )

    if 'variant_more' in args.variant:
        less_attention = True if 'more_ffn' in args.variant else False
        ratio = 4 if '4' in args.variant else 2
        if hooks:

            return model_variant_less_is_more(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
            less_attention  = less_attention,
            ratio           = ratio,
        )
            

        else:
            return model_variant_less_is_more_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
            less_attention  = less_attention,
            ratio           = ratio,

        )
            


    if 'variant_sigmaReparam' in args.variant:
        if hooks:
            return variant_model_sigmaReparam(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

        else:
            return variant_model_sigmaReparam_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

    if args.variant == 'variant_weight_normalization':
        if hooks:
            return model_variant_weight_normalization(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

        else:
            return model_variant_weight_normalization_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )



    if args.variant == 'variant_diff_attn' or args.variant == 'variant_diff_attn_relu':
        isWithAttnNorm = True if args.variant == 'variant_diff_attn' else False
        if hooks:
            return model_variant_diff_attention(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
            isWithAttnNorm  = isWithAttnNorm
        )
        
        else:
            return model_variant_diff_attention_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
            isWithAttnNorm  = isWithAttnNorm
        )
        

   
   
   
    if 'layer_scale' in args.variant:
        if hooks:
            return model_variant_layer_scale(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

        else:
            return model_variant_layer_scale_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )
    
    
    if args.variant == 'attn_variant_light':
        if hooks:
            return model_variant_light_attention(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )

        else:
            return model_variant_light_attention_train(
            isWithBias      = args.model_components["isWithBias"],
            layer_norm      = args.model_components["norm"],
            last_norm       = args.model_components["last_norm"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )





    if hooks:
        return vit_LRP(
            isWithBias           = args.model_components["isWithBias"],
            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],


            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )
    else:
        return vit_LRP_train(
            isWithBias           = args.model_components["isWithBias"],
            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],


            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
        )


   