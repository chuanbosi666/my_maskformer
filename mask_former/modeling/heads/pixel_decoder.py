# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.position_encoding import PositionEmbeddingSine
from ..transformer.transformer import TransformerEncoder, TransformerEncoderLayer


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`. 根据参数来构建pixel d
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


@SEM_SEG_HEADS_REGISTRY.register()
class BasePixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],  # ShapeSpec(channels, height, width, stride)
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"  这也也就意味着时使用的backbone为resnet
        feature_channels = [v.channels for k, v in input_shape]  # [64, 128, 256, 512]

        lateral_convs = []  # 创建一个空的list，用于存放lateral convs
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):  # 
            if idx == len(self.in_features) - 1:  # 当idx为最后一层时
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )  # 设定卷积的参数
                weight_init.c2_xavier_fill(output_conv)  # 对卷积的参数进行初始化
                self.add_module("layer_{}".format(idx + 1), output_conv)  # 将卷积层加入到module中

                lateral_convs.append(None)  # 将lateral convs加入到lateral_convs中，也就是说最后一层没有lateral convs
                output_convs.append(output_conv)  # 将output convs加入到output_convs中
            else:  # 当idx不为最后一层时
                lateral_norm = get_norm(norm, conv_dim)  # 获取归一化，这些都会在detectron2.layers中
                output_norm = get_norm(norm, conv_dim)  # 获取归一化，和上面一样

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )  # 设定卷积的参数  1x1卷积
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )  # 设置卷积的参数，3x3卷积，步长为1，padding为1，激活函数为relu
                weight_init.c2_xavier_fill(lateral_conv)  # 对卷积的参数进行初始化
                weight_init.c2_xavier_fill(output_conv)  # 对卷积的参数进行初始化
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)  # 将两个卷积层加入到module中

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]  # 将lateral convs反转，也就是从高分辨率到低分辨率
        self.output_convs = output_convs[::-1]  # 同上操作

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 设定卷积的参数，3x3卷积，步长为1，padding为1
        weight_init.c2_xavier_fill(self.mask_features)  # 再对其进行初始化

    @classmethod # 将其变为类方法，可以直接通过类名调用
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):  # 传进来参数，cfg和input_shape，都是字典
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }  # 将input_shape中的k和v取出来，如果k在cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES中，就将其加入到ret["input_shape"]中
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM  # 将cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM加入到ret["conv_dim"]中
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM  #同上
        return ret  # 返回ret，就会得到一个字典，包含输入形状，卷积维度，mask维度，归一化

    def forward_features(self, features):  # 对特征进行前向传播
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):  # 从res5到res2进行操作，分别取出索引和特征，即对应的层数
            x = features[f]  # 将特征取出来
            lateral_conv = self.lateral_convs[idx]  # 取出lateral convs
            output_conv = self.output_convs[idx]  #  取出output convs
            if lateral_conv is None:  # 会进行这一步操作，因为最后一层没有lateral convs
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)  # 对特征进行lateral convs操作
                # Following FPN implementation, we use nearest upsampling here   这里使用最近邻上采样
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y) # 再进行output convs操作
        return self.mask_features(y), None

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)  # 先获取一个logger
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")  # 再进行警告
        return self.forward_features(features)  # 最后进行前向传播


class TransformerEncoderOnly(nn.Module):  # 这个是仅用来进行transformer encoder的
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        ) # 导入transformer encoder layer，这个是在transformer.py中定义的，包含了多头注意力机制，前馈网络，残差连接，归一化等
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None  # 进行层归一化
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)  # 这是由多个TransformerEncoderLayer组成的，这里是6层

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape  # 获取src的形状，[B, c, h, w]
        src = src.flatten(2).permute(2, 0, 1)  # 维度变换  [B,C,H,W]->[B,C,H*W]->[H*W,B,C]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # 先展平再变换维度 同上操作
        if mask is not None:
            mask = mask.flatten(1)  # 展平

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # 将我们的图像和其位置编码传入到transformer的encoder中,进行逐像素的编码处理
        return memory.permute(1, 2, 0).view(bs, c, h, w)  # 将获得结果进行维度变换，[H*W,B,C]->[B,C,H*W]->[B,C,H,W]


@SEM_SEG_HEADS_REGISTRY.register()
class TransformerEncoderPixelDecoder(BasePixelDecoder):  # 这个是用来进行transformer encoder和pixel decoder的，将两个的操作放在一起
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        transformer_pre_norm: bool,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5" [res2, res3, res4, res5]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]  # 获取特征的通道数

        in_channels = feature_channels[len(self.in_features) - 1]  #代表了最后一层的通道数，也就是res5的通道数
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)  # 设置卷积
        weight_init.c2_xavier_fill(self.input_proj)  # 对其进行初始化
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )# 对transformer的encoder进行初始化
        N_steps = conv_dim // 2  # 在这里的除以2是因为我们的transformer的encoder的输出是一半的维度，以匹配我们的输入维度
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)  

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )  # 对卷积进行设置
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):  # 对参数进行配置
        ret = super().from_config(cfg, input_shape)
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        return ret

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution) 将特征图反转为自上而下的顺序（从低分辨率到高分辨率）
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]  # 获取特征
            lateral_conv = self.lateral_convs[idx]  # 将索引和特征取出来
            output_conv = self.output_convs[idx]  # 同上操作
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer  # 这一部分的代码就是将transformer的encoder的输出作为transformer的decoder的输入
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)  # 当为最后一层时，就会进行这一步操作
        return self.mask_features(y), transformer_encoder_features  # 会返回两个值，一个是mask features，就是经过一个3x3的padding为1的卷积，一个是transformer encoder features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")  # 创建日志
        return self.forward_features(features)  # 将特征进行前向传播，来获得mask features和transformer encoder features
