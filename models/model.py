import math
import warnings
from xml.parsers.expat import model
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn, Tensor
from models.context_modules import get_context_module
from models.model_utils import ConvBNAct, Swish, Hswish, SqueezeAndExcitation
from models.decoder import Decoder
from utils.galore import GaLoreAdamW

# DA2 imports
from transformers import AutoImageProcessor
from transformers import DepthAnythingForDepthEstimation

# SAM2 imports
from sam2.build_sam import build_sam2

class Attention(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out



def get_last_n_block_outputs(sam_encoder, image):
    outputs = sam_encoder(image)

    out = outputs["vision_features"] 
    fpn = outputs["backbone_fpn"]  


    skip3 = fpn[0]
    skip2 = fpn[1]
    skip1 = fpn[2]

    return out, skip1, skip2, skip3


class CrossAttentionFuse(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, num_heads):
        super().__init__()
        self.embed_dim1 = embed_dim1  
        self.embed_dim2 = embed_dim2 
        self.num_heads = num_heads

        self.proj_mat2 = nn.Linear(embed_dim2, embed_dim1)
        self.cross_attn = nn.MultiheadAttention(embed_dim1, num_heads)
        
    def forward(self, mat1, mat2):
        B, C1, H, W = mat1.shape
        _, C2, _, _ = mat2.shape

        # Flatten spatial dimensions
        x1 = mat1.view(B, C1, -1).permute(2, 0, 1) 
        x2 = mat2.view(B, C2, -1).permute(2, 0, 1)  

        x2_proj = self.proj_mat2(x2)  

        attn_output, _ = self.cross_attn(query=x1, key=x2_proj, value=x2_proj)

        # Add & reshape to original mat1 shape
        fused = attn_output.permute(1, 2, 0).contiguous().view(B, C1, H, W)  
        return fused



class Model(nn.Module):
    def __init__(self,
                 height=256,
                 width=480,
                 num_classes=4,
                 channels_decoder=None,

                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None, 
                 weighting_in_encoder='None',
                 upsampling='bilinear'):
        super(Model, self).__init__()

        if channels_decoder is None:
            channels_decoder=[256, 256, 256]

        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.weighting_in_encoder = weighting_in_encoder

        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError('Only relu, swish and hswish as '
                                      'activation function are supported so '
                                      'far. Got {}'.format(activation))
        

        # -------------- Add SAM2 encoder --------------
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        checkpoint = "sam2.1_hiera_large.pt" 

        sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
        sam2_encoder = sam2_model.image_encoder

        # SRFT GaLore
        target_modules_list = ["attn.qkv", "attn.proj"]
        galore_params = []
        id_galore_params = []

        for module_name, module in sam2_encoder.named_modules():
            if isinstance(module, nn.Linear) and any(t in module_name for t in target_modules_list):
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.requires_grad = True
                    galore_params.append(module.weight)
                    id_galore_params.append(id(module.weight))
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad = False 
            else:
                for param in module.parameters():
                    param.requires_grad = False

        if isinstance(module, nn.Linear) and any(t in module_name for t in target_modules_list):
            module.weight.requires_grad = True
            galore_params.append(module.weight)
            id_galore_params.append(id(module.weight))
        else:
            for param in module.parameters():
                param.requires_grad = False
        self._galore_params = galore_params
        self._id_galore_params = id_galore_params
        self.sam2_encoder = sam2_encoder


        # ----------------------------------------------
        self.first_conv = ConvBNAct(1, 3, kernel_size=1,
                                    activation=self.activation)

        self.mid_img_conv = ConvBNAct(512, 256, kernel_size=1,
                                      activation=self.activation)

        self.channels_decoder_in = 256 

        if weighting_in_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExcitation(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExcitation(
                self.encoder.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExcitation(
                self.encoder.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExcitation(
                self.encoder.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExcitation(
                self.encoder.down_32_channels_out,
                activation=self.activation)
        else:
            self.se_layer0 = nn.Identity()
            self.se_layer1 = nn.Identity()
            self.se_layer2 = nn.Identity()
            self.se_layer3 = nn.Identity()
            self.se_layer4 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = get_context_module(
            context_module,
            self.channels_decoder_in,
            channels_decoder[0],
            
            input_size=(height // 32, width // 32),
            activation=self.activation,
            upsampling_mode=upsampling_context_module)

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

        self.fuse_model = CrossAttentionFuse(embed_dim1=256, embed_dim2=256, num_heads=2) 

        # DA2 encoder for depth
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        self.da2_processor = AutoImageProcessor.from_pretrained(model_name)
        da2_model = DepthAnythingForDepthEstimation.from_pretrained(model_name)
        self.da2_encoder = da2_model.backbone 
        # freeze the DA2 encoder
        for param in self.da2_encoder.parameters():
            param.requires_grad = False
        self.depth_conv = ConvBNAct(channels_in=384, channels_out=256, kernel_size=1, activation=self.activation)

    def get_galore_optimizer(self, lr, weight_decay, rank):
        print("Applying GaLore to model with rank:", rank)
        regular_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in self._id_galore_params
        ]

        param_groups = [
            {'params': regular_params},
            {'params': self._galore_params, 'rank': rank, 'update_proj_gap': 50, 'scale': 1, 'proj_type': "std"}
        ]

        optimizer = GaLoreAdamW(param_groups, lr=lr, weight_decay=weight_decay)
        return optimizer

    def forward(self, image, depth):
        # ----- DA2 encoder for depth ------
        with torch.no_grad():
            inputs = self.da2_processor(images=image, return_tensors="pt", do_rescale=False)
            device = next(self.da2_encoder.parameters()).device
            pixel_values = inputs["pixel_values"].to(device)
            features = self.da2_encoder(pixel_values)

            feat = features.feature_maps[-1] 
            B, N, C = feat.shape
            H = W = int(N ** 0.5) 
            feat = feat[:, :H*W, :] 
            feat = feat.permute(0, 2, 1).reshape(B, C, H, W) 
            feat = self.depth_conv(feat) 

        depth_feat = F.interpolate(feat, size=(64, 64), mode="bilinear", align_corners=False) # [B, 256, 64, 64]

        # ----- SAM encoder for RGB ------
        out, skip1, skip2, skip3  = get_last_n_block_outputs(self.sam2_encoder, image)

       # ----- Cross Attention Fusion ------
        out_resized = F.interpolate(out, size=depth_feat.shape[-2:], mode='bilinear', align_corners=False)
        fused_feat = self.fuse_model(depth_feat, out_resized) 
        
        outs = [fused_feat, skip3, skip2, skip1]

        outs, out_visual = self.decoder(enc_outs=outs) 
        outs = F.log_softmax(outs, dim=1) 
        return outs, depth_feat
