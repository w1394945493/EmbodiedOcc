# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Union
import torch, torch.nn as nn

from mmengine import MODELS
from mmengine.model import BaseModule
try:
    from .ops import DeformableAggregationFunction as DAF
except:
    DAF = None
"""
anchor_encoder = dict(
    type='SparseGaussian3DEncoder',
    embed_dims=_dim_, 
    semantic_dim=cls_dims,
)
refine_layer = dict(
    type='SparseGaussian3DRefinementModule',
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    restrict_xyz=True,
    unit_xyz=[4.0, 4.0, 1.0],
    refine_manual=[0, 1, 2],
    semantic_dim=cls_dims,
    semantics_activation=semantics_activation,
)
spconv_layer=dict(
    type='SparseConv3D',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    grid_size=[0.8]*3,
    kernel_size=3,
)
spconv_layer_fillhead=dict(
    type='SparseConv3D',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    grid_size=[0.8]*3,
    kernel_size=3,
    dilation=2
)
"""
@MODELS.register_module()
class SparseGaussianFormer(BaseModule):
    def __init__(
        self,
        anchor_encoder,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        mid_refine_layer: dict = None,
        num_decoder: int = 6,
        spconv_layer: dict = None,
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg):
            if cfg is None:
                return None
            return MODELS.build(cfg)
        
        self.anchor_encoder = build(anchor_encoder)
        self.op_config_map = {
            "norm": norm_layer,
            "ffn": ffn,
            "deformable": deformable_model,
            "refine": refine_layer,
            "mid_refine":mid_refine_layer,
            "spconv": spconv_layer,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )
        
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        anchor,
        instance_feature,
        feature_maps: Union[torch.Tensor, List], # mlvl_img_feats
        metas: dict,
    ):
        
        # *【单帧流程 4：GaussianFormer 输入】
        # * anchor 是相机局部显式参数，instance_feature 已含 Depth Anything 提示，
        # * feature_maps 是当前帧 EfficientNet 多尺度特征；没有历史 query/feature。
        if DAF is not None:
            feature_maps = DAF.feature_maps_format(feature_maps)

        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        anchor_embed = self.anchor_encoder(anchor) # [1, 21600, 96]

        # 配置 num_decoder=3，每个 refine 产生一版新 anchor；但函数最后只返回
        # prediction[-1]，因此只有第三层 Gaussian 进入 Occ head 接受直接监督。
        prediction = []
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                # *【4.2 Gaussian-Gaussian 交互】对三维空间相邻 query 做稀疏卷积；
                # * 第一层 operation_order 无 spconv，第二、三层才执行该操作。
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    metas)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                # *【4.1 Gaussian-图像交互】由 anchor 生成三维关键点并投影到当前
                # * RGB 的多尺度 feature map，把采样到的外观证据写回 query。
                # assert feature_queue is None and meta_queue is None and self.depth_module is None
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif "refine" in op:
                # *【4.3 显式属性更新】根据 query 预测位置、尺度、旋转、opacity
                # * 和 semantic 增量，生成当前 Decoder 层的新 Gaussian anchor。
                anchor, gaussian, cls = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    metas,
                )
                prediction.append(anchor)
                
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        # *【单帧流程 4 输出】只将最后一层 Gaussian 交给 Occupancy head。
        return prediction[-1]
