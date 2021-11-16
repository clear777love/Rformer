import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum

#########################################
class Rformer(nn.Module):
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='ffn', se_layer=False,
                 dowsample=Downsample, upsample=NearestUpsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        
        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        self.Bottleneck_0 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.Bottleneck0_conv = WConvBlock(embed_dim*16, embed_dim*16, strides=1)
        
        # RICB
        self.Wup_0 = upsample(embed_dim*16, embed_dim*8)
        self.Wconv_d0 = WConvBlock(embed_dim*8, embed_dim*8, strides=1)
        
        self.Wup_1 = upsample(embed_dim*8, embed_dim*4)
        self.Wconv_d1 = WConvBlock(embed_dim*4, embed_dim*4, strides=1)

        self.Wup_2 = upsample(embed_dim*4, embed_dim*2)
        self.Wconv_d2 = WConvBlock(embed_dim*2, embed_dim*2, strides=1)
        
        self.Wup_3 = upsample(embed_dim*2, embed_dim)
        self.Wconv_d3 = WConvBlock(embed_dim, embed_dim, strides=1)
        
        self.Wout = OutputProj(in_channel=embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        self.Win = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        
        self.Wconv_e0 = WConvBlock(embed_dim*2, embed_dim, strides=1)
        self.Wdown_0 = dowsample(embed_dim, embed_dim*2)
         
        self.Wconv_e1 = WConvBlock(embed_dim*4, embed_dim*2, strides=1)
        self.Wdown_1 = dowsample(embed_dim*2, embed_dim*4)
        
        self.Wconv_e2 = WConvBlock(embed_dim*8, embed_dim*4, strides=1)
        self.Wdown_2 = dowsample(embed_dim*4, embed_dim*8)
        
        self.Wconv_e3 = WConvBlock(embed_dim*16, embed_dim*8, strides=1)
        self.Wdown_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        self.Bottleneck_1 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer) 
        self.Bottleneck1_conv = WConvBlock(embed_dim*16, embed_dim*16, strides=1)

        # Decoder
        self.merge_conv_0 = WConvBlock(embed_dim*32, embed_dim*16, strides=1)
        self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
        #self.decoderbuf_0 = WConvBlock(embed_dim*16, embed_dim*16, strides=1)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        
        self.merge_conv_1 = WConvBlock(embed_dim*16, embed_dim*8, strides=1)
        self.upsample_1 = upsample(embed_dim*8, embed_dim*4)
        #self.decoderbuf_1 = WConvBlock(embed_dim*8, embed_dim*8, strides=1)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        
        self.merge_conv_2 = WConvBlock(embed_dim*8, embed_dim*4, strides=1)
        self.upsample_2 = upsample(embed_dim*4, embed_dim*2)
        #self.decoderbuf_2 = WConvBlock(embed_dim*4, embed_dim*4, strides=1)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        
        self.merge_conv_3 = WConvBlock(embed_dim*4, embed_dim*2, strides=1)
        self.upsample_3 = upsample(embed_dim*2, embed_dim)
        #self.decoderbuf_3 = WConvBlock(embed_dim*2, embed_dim*2, strides=1)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        self.merge_conv_out = WConvBlock(embed_dim*2, embed_dim, strides=1)
        #self.outbuf = WConvBlock(2*embed_dim, embed_dim, strides=1)
        self.apply(self._init_weights)

    def forward(self, x, stage, mask=None):

        # To prevent code leakage, this part of the code is temporarily hidden
        
        return x + RICB_out, x + y
