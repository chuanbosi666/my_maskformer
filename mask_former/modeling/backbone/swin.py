# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/mmseg/models/backbones/swin_transformer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# 从timm中导入DropPath, to_2tuple, trunc_normal_，这三个函数，分别是dropout，tuple，正态分布，用于初始化
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# 开源计算机视觉框架 Detectron2 中的一个模块，从detectron2中导入BACKBONE_REGISTRY：一个用于注册和管理不同骨干（backbone）模型的注册表
# Backbone：一个表示深度学习模型骨干的抽象类或对象, ShapeSpec：一个类或对象，用于定义特征图的形状规范
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec  


class Mlp(nn.Module):
    """Multilayer perceptron. 多层感知机
    """

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):  # 对参数的初始化
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 线性层的初始化
        self.act = act_layer()  # 激活函数的初始化
        self.fc2 = nn.Linear(hidden_features, out_features)  # 线性层的初始化
        self.drop = nn.Dropout(drop)  # dropout层的初始化

    def forward(self, x):  # 构建多层感知机
        x = self.fc1(x)  # 线性层
        x = self.act(x)  # 激活函数
        x = self.drop(x)  # dropout
        x = self.fc2(x)  # 线性层
        x = self.drop(x)  # dropout
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape  # 从传入的x中获取B, H, W, C
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)  # 将x的形状变为(B, H//window_size, window_size, W//window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) # 获取窗口，contiguous是一个确保张量在内存中是连续存储的操作，将x的形状变为(-1, window_size, window_size, C)，其中-1表示自动计算
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))  # 计算B，其中windows.shape[0]为num_windows*B，H*W/window_size/window_size为num_windows
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)  # 将windows的形状变为(B, H//window_size, W//window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  # 将x的形状变为(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,  # 7
        num_heads,  # [3, 6, 12, 24]
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 这里对应的是根号d

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH  将张量变成可学习的参数

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  #7
        coords_w = torch.arange(self.window_size[1])  # 7
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww 通过torch.meshgrid()函数生成网格点坐标矩阵，再通过torch.stack()函数将两个网格点坐标矩阵合并为一个三维矩阵
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        #通过广播机制，将coords_flatten[:, :, None]和coords_flatten[:, None, :]的形状变为(2, Wh*Ww, Wh*Ww)，然后相减得到相对坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0 从下面的类可知，窗口大小为7
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 这是在水平维度将其从0开始
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 通过Swin-tr可以知道，这是为了求得全部坐标系内的坐标
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww  将其相加得到相对位置索引
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性层的初始化，输出的通道是dim*3
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)  # 对relative_position_bias_table进行正态分布初始化
        self.softmax = nn.Softmax(dim=-1)  # 对最后一个维度进行softmax操作

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  # 获取x的形状 B_为num_windows*B, N为Wh*Ww, C为C
        qkv = (
            self.qkv(x)  # qkv() = [ batch_size*num_windows,Mh*Mw,3*hidden_dim]
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)  # 将其形状变为(B_, N, 3, nH, C//nH)  这个3代表的是qkv三个参数
            .permute(2, 0, 3, 1, 4)  # 将x的形状变为(3, batch_size*num_windows, num_head, MH*Mw, embed_dim_per_head)
        )  # qkv = 3, B_, nH, N, C//nH
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
# q = 3，k= B_，v = nH
        q = q * self.scale  # 对q进行缩放
        attn = q @ k.transpose(-2, -1)  # 将q和k进行矩阵乘法，,对其进行转置，将q的最后两个维度进行转置，得到attn的形状为(3, B_, nH, N, N)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH  view函数是将张量维度进行改变，这一步就是先将其变成的一维的，然后再按照Wh*Ww,Wh*Ww,nH的形状变换成三维的形状
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww  先使用contiguous()函数将其变为连续存储，然后将其转置，得到相对位置偏置的形状为(nH, Wh*Ww, Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)  # 这时我们的attn，就是q和k的矩阵乘法加上相对位置偏置，得到的attn的形状为(3, B_, nH, N, N)

        if mask is not None:  # 当使用mask时，但下面的代码并未使用mask，所以这里的mask是没有用的
            nW = mask.shape[0]  # 获取mask的形状
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)   # 这里的attn是q和k的矩阵乘法加上相对位置偏置，然后进行softmax操作，得到的attn的形状为(3, B_, nH, N, N)

        attn = self.attn_drop(attn)  # 再将attn进行dropout操作

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # 对attn和v进行矩阵乘法，然后将其转置，将其形状变为(B_, N, C)
        x = self.proj(x)  # 再将其进行线性变换，将其形状变为(B_, N, C)
        x = self.proj_drop(x)  # 再将其进行dropout操作，将其形状变为(B_, N, C)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,  # 使用的激活函数是GELU
        norm_layer=nn.LayerNorm,  # 使用的归一化方式是LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)  # 对输入进行归一化
        self.attn = WindowAttention(  # 进行窗口注意力机制
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # dropout层的初始化
        self.norm2 = norm_layer(dim)  # 对输入再次进行归一化
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )  # 对输入进行多层感知机操作

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape  # 获取图像的形状，为(B, H*W, C)
        H, W = self.H, self.W  # 获取H和W
        assert L == H * W, "input feature has wrong size"

        shortcut = x  # 将x赋值给shortcut
        x = self.norm1(x)  # 对x进行归一化
        x = x.view(B, H, W, C)  # 进行变维

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0  # 初始化pad_l和pad_t
        pad_r = (self.window_size - W % self.window_size) % self.window_size  # 计算pad_r
        pad_b = (self.window_size - H % self.window_size) % self.window_size  # 计算pad_b,只计算填充了右边和下面的部分
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # 将x进行填充
        _, Hp, Wp, _ = x.shape  # 再次获取x的形状

        # cyclic shift
        if self.shift_size > 0:
            # 这一行代码就是将x进行循环移位，将其移动shift_size个像素
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
            attn_mask = mask_matrix
        else:
            # 滚动大小为0时，不进行循环移位
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C  这一步是将图像分割为窗口，其中nW为窗口的数量
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C  这里 -1 表示自动计算该维度的大小，以确保总元素数量不变。

        # W-MSA/SW-MSA
        # 这一步是进行窗口注意力机制，其中x_windows的形状为nW*B, window_size*window_size, C，attn_mask的形状为None
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows 将窗口合并
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # 将注意力窗口进行维度调换
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C  将其进行反向操作，得到shifted_x的形状为B H' W' C

        # reverse cyclic shift 将其进行反向循环移位
        if self.shift_size > 0:  # 当shift_size大于0时，才进行反向循环移位
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x  # 这是不进行反向位移循环

        if pad_r > 0 or pad_b > 0:  # 当pad_r或pad_b大于0时，才进行切片操作
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)  # 对得到的结果进行处理

        # FFN
        x = shortcut + self.drop_path(x)  # 进行dropout操作，相当于一个残差
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 进行多层感知机操作，再进行一个相加

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))  # 这个代表的是所要填充的数值为0，分别填充图像的左右不进行填充，上、下两个方向，填充的数值为W%2, H%2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C  这个是取得H和W的偶数行和偶数列
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C  这个是取得H和W的奇数行和偶数列
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C  这个是取得H和W的偶数行和奇数列
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C  这个是取得H和W的奇数行和奇数列
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C  将其对最后一个维度进行拼接
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)  # 再对其进行归一化
        x = self.reduction(x)  # 再继续进行线性层的操作

        return x  # 构建完成patch的合并


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(  # 定义的self变量，维度，深度，头的数量，窗口的大小，mlp的比例，是否使用偏置，是否使用qk_scale，dropout的比例，注意力dropout的比例，dropout的路径，归一化层，下采样，是否使用checkpoint
        self,
        dim,
        depth,
        num_heads,  # [3, 6, 12, 24]
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)  # 这里的depth是指的深度,也就是说，这里的blocks是一个列表，其中的元素是SwinTransformerBlock，深度代表了block的数量
            ]  # 这里的深度为[2, 2, 6, 2],在四个阶段的block的数量
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size  # 使用ceil函数对其取整，然后乘以窗口的大小，这样处理是为了保证能够被窗口大小整除
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )  # 在高度上进行切片
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )# 从宽度维度上进行切片   对于这里的切片，就是在高度和宽度上分别进行切片，第一个是0到倒数第七个，第二个是倒数第七个到倒数第四个，第三个是倒数第四个到最后一个，这样就形成了九个区域
        cnt = 0
        for h in h_slices:  # 对于这一步的循环，就是对上面的九个区域进行编号，从第一行开始，从左到右，从上到下，每个区域分别为0-8
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1  获得窗口
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  #   [nW, Mh,Mw,1]对于mask_windows进行变维，这样变维是为了和下面的mask_matrix进行相加
        # 在这一部分，利用了广播机制，先将[nW,Mh*mW]扩维，分别为[nW,1,Mh*mW]和[nW,Mh*mW,1]，然后相减，得到的结果为[nW,Mh*mW,Mh*mW]
        # 注意的一点是，（广播机制）这个两个矩阵相减，会对1这个维度进行复制Mh*mW次，对2这个维度进行复制Mh*mW次，然后再相减，在其相同的地方为0，不同的地方空出来
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW,Mh*mW] 
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )  # 这一步对上面的attn_mask进行填充，将其不为0的地方填充为-100，为0的地方填充为0

        for blk in self.blocks:  # 这一步是构建我们的swin_transformer_block，根据设定的次数来构建
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:  # 进行我们的patch_merging操作
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W  # 经过block操作加上patch_merging操作后，构成了我们的Swin_transformer的base_layer


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()  # 取出高和宽
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))  # 当W不能被patch_size[1]整除时，对其进行填充
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))  # 同理

        x = self.proj(x)  # B C Wh Ww 进行卷积操作
        if self.norm is not None:  # 从下面看的话，对其进行操作
            Wh, Ww = x.size(2), x.size(3)  # 取出高和宽
            x = x.flatten(2).transpose(1, 2)
             # 对高进行展平，从第二维开始展平，所以H会和W相乘，然后进行转置
             # 这里的x的形状为B H*W C
            x = self.norm(x)   # 这里只对C进行归一化,因为归一化操作是针对通道维度进行的
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww) # 再进行转置，然后进行变维

        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,  # 为True时，使用绝对位置编码
        patch_norm=True,
        out_indices=(0, 1, 2, 3),  # 输出的索引
        frozen_stages=-1,
        use_checkpoint=False,
    ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)  # dropout层的初始化

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))  # 使用官方的方法，将drop_path_rate从0升到drop_path_rate，需要sum(depths)次
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 这里的num_layers为4，所以会遍历四次
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        # x:[B, C, H, W]
        x = self.patch_embed(x)  # 先对图像进行一个patch的切分，实现四倍的下采样

        Wh, Ww = x.size(2), x.size(3)  # 获得图像的高和宽
        if self.ape:  # 对使用绝对位置编码的情况进行处理
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:  # 这里是不使用绝对位置编码的情况，直接对其进行处理
            x = x.flatten(2).transpose(1, 2)  # B C H W->B C H*W -> B H*W C
        x = self.pos_drop(x)  # 对图像进行dropout操作

        outs = {} # 创建一个字典
        for i in range(self.num_layers):  # 对我们的layers进行遍历，就是文中的stage1-stage4
            layer = self.layers[i]  # 取出第i个layer
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)  # 传进去图像和高宽，得到输出

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs["res{}".format(i + 2)] = out

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


@BACKBONE_REGISTRY.register()
class D2SwinTransformer(SwinTransformer, Backbone):  # 这个类继承了SwinTransformer和Backbone，是为了适用于特定的深度学习任务
    def __init__(self, cfg, input_shape):

        pretrain_img_size = cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE
        patch_size = cfg.MODEL.SWIN.PATCH_SIZE
        in_chans = 3
        embed_dim = cfg.MODEL.SWIN.EMBED_DIM
        depths = cfg.MODEL.SWIN.DEPTHS
        num_heads = cfg.MODEL.SWIN.NUM_HEADS
        window_size = cfg.MODEL.SWIN.WINDOW_SIZE
        mlp_ratio = cfg.MODEL.SWIN.MLP_RATIO
        qkv_bias = cfg.MODEL.SWIN.QKV_BIAS
        qk_scale = cfg.MODEL.SWIN.QK_SCALE
        drop_rate = cfg.MODEL.SWIN.DROP_RATE
        attn_drop_rate = cfg.MODEL.SWIN.ATTN_DROP_RATE
        drop_path_rate = cfg.MODEL.SWIN.DROP_PATH_RATE
        norm_layer = nn.LayerNorm
        ape = cfg.MODEL.SWIN.APE
        patch_norm = cfg.MODEL.SWIN.PATCH_NORM

        super().__init__(
            pretrain_img_size,
            patch_size,
            in_chans,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            ape,
            patch_norm,
        )

        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
