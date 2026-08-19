import torch
import numpy as np
from copy import deepcopy
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG
import sys
# sys.path.append('/data1/code/wyq/gaussianindoor/EmbodiedOcc/EfficientNet-PyTorch')
from efficientnet_pytorch import EfficientNet
import sys
# sys.path.append('/data1/code/wyq/gaussianindoor/EmbodiedOcc')
# sys.path.append('/data1/code/wyq/gaussianindoor/EmbodiedOcc/Depth-Anything-V2/metric_depth')
# sys.path.append('/data1/code/wyq/gaussianindoor/EmbodiedOcc/model/depthbranch')
from depth_anything_v2.dpt import DepthAnythingV2
from depthnet import DepthNet
from unet2d import DecoderBN
import torch.nn as nn
from PIL import Image
import cv2
import torch.nn.functional as F

@MODELS.register_module()
class GaussianSegmentor(BaseModule):

    def __init__(
        self,
        flag_depthbranch=False,
        flag_depthanything_as_gt=False,
        depthbranch=None,
        backbone=None,
        neck=None,
        lifter=None,
        encoder=None,
        future_decoder=None,
        head=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.flag_depthbranch = flag_depthbranch
        self.flag_depthanything_as_gt = flag_depthanything_as_gt
        if flag_depthbranch:
            if flag_depthanything_as_gt:
                # ======================================================== #
                # 第二条 RGB 分支：Depth Anything V2 几何先验分支。
                # 输入来自 meta['img_depthbranch']，输出是一张单目深度图。它不负责
                # 提供 GaussianFormer 的图像 feature map，也不直接产生 Gaussian；
                # 深度图会在 GaussianNewLifter 中按 anchor 投影位置被查询。
                # ======================================================== #
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                self.depthanything = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth':20})
                # checkpoint = torch.load('/data1/code/wyq/gaussianindoor/EmbodiedOcc/checkpoints/finetune_scannet_depthanythingv2.pth', map_location='cpu')['model']
                checkpoint = torch.load('/c20250502/wangyushen/Weights/gpocc/finetune_scannet_depthanythingv2.pth', map_location='cpu')['model']

                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        new_key = k[len('module.'):]
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                self.depthanything.load_state_dict(new_state_dict)

            # ============================================================ #
            # 第一条 RGB 分支：EfficientNet-B7 + DecoderBN 主视觉分支。
            # 输入是 train_mono.py 传入的 imgs，输出四尺度 96 维特征，供
            # GaussianFormer 根据 Gaussian 投影位置做 deformable feature sampling。
            # 该分支和 Depth Anything 是两套独立网络，只在 lifter/encoder 阶段
            # 通过“深度增强的 query + 多尺度图像特征”间接汇合。
            # ============================================================ #
            basemodel_name = "tf_efficientnet_b7_ns"
            num_features = 2560
            print("Loading base model ()...".format(basemodel_name), end="")
            # basemodel = torch.hub.load(
            #     "/home/wyq/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master", basemodel_name, pretrained=True, trust_repo=True, source='local'
            # )
            basemodel = torch.hub.load(
                "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
            )   # efficient net
            print("Done.")
            # Remove last layer
            print("Removing last two layers (global_pool & classifier).")
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()

            self.backbone = basemodel

            self.neck = DecoderBN(
                out_feature=96,
                use_decoder=True,
                bottleneck_features=num_features,
                num_features=num_features,
            )
        else:
            basemodel_name = "tf_efficientnet_b7_ns"
            num_features = 2560
            print("Loading base model ()...".format(basemodel_name), end="")
            basemodel = torch.hub.load(
                "/home/wyq/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master", basemodel_name, pretrained=True, trust_repo=True, source='local'
            )
            print("Done.")
            # Remove last layer
            print("Removing last two layers (global_pool & classifier).")
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()

            self.backbone = basemodel

            self.neck = DecoderBN(
                out_feature=96,
                use_decoder=True,
                bottleneck_features=num_features,
                num_features=num_features,
            )
        if lifter is not None:
            self.lifter = MODELS.build(lifter)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        if future_decoder is not None:
            self.future_decoder = MODELS.build(future_decoder)
        if head is not None:
            self.head = MODELS.build(head)

    def extract_img_feat(self, imgs):
        """第一条 RGB 分支：从主输入 imgs 提取 GaussianFormer 使用的视觉特征。

        返回的 mlvl features 会在三层 SparseGaussianFormer 中被 Gaussian query
        投影采样；返回值不是深度图，也不会送入 Depth Anything。
        """
        # Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth" to /home/wyq/.cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth
        # *【主视觉分支实际输入】默认单帧配置下 imgs 为 [B,1,3,480,640]；
        # * 这里展平相机维后，EfficientNet 接收到 [B,3,480,640]。
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W) # 1, 3, 480, 640

        feature_x = [imgs]
        feature_idx = 0
        this_x = feature_x[-1]
        for k, v in self.backbone._modules.items(): # backbone: GenEfficientNet
            if k == "blocks":
                for ki, vi in v._modules.items():
                    this_x = vi(this_x)
                    feature_idx += 1
                    if feature_idx in [4, 5, 6, 8, 11]:
                        feature_x.append(this_x)
            else:
                this_x = v(this_x)
                feature_idx += 1
                if feature_idx in [4, 5, 6, 8, 11]:
                    feature_x.append(this_x)

        img_feats_backbone = feature_x  # 6:(1 3 480 640) (1 32 240 320) (1 48 120 160) (1 80 60 80) (1 224 30 40) (1 2560 15 20)

        # list of [2560, 15, 20]
        img_feats_out = self.neck(img_feats_backbone) # dict

        img_feats_reshaped = []
        for img_feat in img_feats_out.values():
            BN, C, H, W = img_feat.size()
            if W != 640:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))   # 4:(1 1 96 240 320) (1 1 96 120 160) (1 1 96 60 80) (1 1 96 30 40)

        return img_feats_reshaped, img_feats_out['1_1'] # 4:(1 1 96 240 320) (1 1 96 120 160) (1 1 96 60 80) (1 1 96 30 40); (1 96 480 640)

    def obtain_bev(self, imgs, metas):
        """由单帧 RGB 构造并细化稀疏 Gaussian。

        函数名保留为 obtain_bev，但返回的不是二维 BEV feature，而是最终一层
        Gaussian 参数 [B*F,G,C]；双 RGB 分支、深度注入和 GaussianFormer 在此串联。
        """
        B, F, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B*F, N, C, H, W)
        # *【单帧流程 2：同一 RGB 的双分支特征提取】
        # ================================================================ #
        # 两条 RGB 分支在这里并行产生互补信息：
        #   分支一 imgs -> EfficientNet/DecoderBN -> mlvl_img_feats（视觉外观）；
        #   分支二 meta['img_depthbranch'] -> Depth Anything -> depthnet_output
        #          （逐像素几何深度）。
        # 两者源于同一帧 RGB，但数据预处理、网络权重和下游用途不同。
        # ================================================================ #

        # 分支一：主视觉特征。后面的 encoder 会反复从 mlvl_img_feats 投影采样。
        mlvl_img_feats, feature_x_4 = self.extract_img_feat(imgs) # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]

        # 分支二：用冻结的 Depth Anything V2 预测当前帧深度。这里刻意读取
        # metas 内的专用输入，而不是上面经过主分支增强/归一化后的 imgs。
        if self.flag_depthbranch: # True
            if self.flag_depthanything_as_gt:
                # depth branch
                self.depthanything.eval()

                # 仅支持 bs=1 的原实现：只读取 metas[0] 并生成一张深度图。
                # image_ = metas[0]['img_depthbranch']    # (1 3 490 644)
                # depth_pred = self.depthanything.infer_image(image_, 480, 640, 480)  # (480 640)
                # depthnet_output = depth_pred

                # 支持 bs>1：逐样本调用单图接口，再堆叠成 [B,H,W] 深度图。
                # train_mono.py 已将 Depth Anything 参数冻结；这里再设 eval()，保证
                # 即使外层 model.train()，该几何先验网络仍使用固定推理行为。
                depth_preds = []
                for meta in metas:
                    # *【Depth 分支输入与输出尺寸】meta['img_depthbranch'] 已由 Dataset
                    # * 预处理为 [1,3,490,644]，这是 Depth Anything 的真实网络输入。
                    # * 这里额外传入 h_=480、w_=640；metric-depth 版本的 infer_image
                    # * 会在网络预测后用双线性插值把深度恢复成 [480,640]，所以 lifter
                    # * 可以使用主图像的 480x640 像素坐标查询深度。
                    depth = self.depthanything.infer_image(
                        meta["img_depthbranch"], 480, 640, 480
                    )
                    depth_preds.append(depth)
                depthnet_output = torch.stack(depth_preds, dim=0)

            else:
                depthnet_output = None
        else:
            depthnet_output = None

        # ------------------------- 两条分支的首次汇合 ----------------------
        # lifter 不会把深度图拼到图像 feature map：它把每个初始 Gaussian 投影到
        # depthnet_output 上查询表面深度，将其编码进 instance_feature/query。
        # mlvl_img_feats 主要用于确定 batch，并原样继续传给下方 encoder。
        # *【单帧流程 3：Lifter 初始化 depth-aware Gaussian query】
        # * 固定 anchor 根据场景/相机重新定位；每个候选再从深度图查询表面深度，
        # * 将几何提示编码进 instance_feature。
        anchor, instance_feature, depth2occ, depthnet_output_loss, predtoreturn = self.lifter(self.flag_depthbranch, self.flag_depthanything_as_gt, depthnet_output, mlvl_img_feats, metas)    # b, g, c

        # ------------------------- 两条分支的联合利用 ----------------------
        # encoder 接收“含深度先验的 instance_feature”和“主视觉多尺度特征”，
        # Gaussian query 一边携带深度提示，一边从当前 RGB feature map 采样外观信息，
        # 经过 deformable aggregation / FFN / refinement 更新显式 Gaussian。
        # *【单帧流程 4：三层 GaussianFormer 细化】
        # * 深度增强 query 从主 RGB 多尺度特征采样，并经 SparseConv/FFN/refine
        # * 更新显式 Gaussian 的位置、尺度、旋转、opacity 和语义。
        anchor = self.encoder(anchor, instance_feature, mlvl_img_feats, metas) # b, g, c

        return anchor, depth2occ, depthnet_output_loss, predtoreturn    # (2 16200 23) None None None

    def forward(
        self,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        grad_frames=None,
        test_mode=False,
        **kwargs,
    ):
        # *【单帧流程总入口】默认 [B,F=1,N=1,3,480,640]。F 只是通用接口维度；
        # * train_mono 不传播任何跨帧状态，所以这是严格单帧模型。
        B, F, N, C, H, W = imgs.shape   # (1 1 1 3 480 640)
        # 仅支持 bs=1 的原限制；后续 lifter/encoder/head 已保留 batch 维，因此不再执行。
        # assert B==1, 'bs > 1 not supported'
        if grad_frames is not None:
            assert grad_frames < F
            imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index = self.frame_split(grad_frames, imgs, metas)
            bev_grad = self.obtain_bev(imgs_grad, metas_grad)
            with torch.no_grad():
                bev_no_grad = self.obtain_bev(imgs_no_grad, metas_no_grad)
            bev = torch.cat([bev_grad, bev_no_grad], dim=0)[inv_index]
        else:
            bev, depth2occ, depthnet_output_loss, predtoreturn = self.obtain_bev(imgs, metas)

        # BF, H, W, C = bev.shape
        BF, G, C = bev.shape # bev is actually anchors [1, 21600, 24]
        bev = bev.reshape(B, F, G, C)   # (2 1 16200 23)
        if hasattr(self, 'future_decoder'):
            output_dict = self.future_decoder(bev, metas)
            bev_predict = output_dict.pop('bev')
        else:
            bev_predict = bev
            output_dict = dict()
        # *【单帧流程 5～7：Gaussian 解码与体素聚合】
        # * 最终 anchor 解码为显式 Gaussian、变到世界坐标，然后在当前局部
        # * 60x60x36 体素中心聚合成语义 Occupancy logits。
        output_dict = self.head(
            bev_feat=bev_predict,   # (2 1 16200 23)
            points=points,
            label=label,            # (2 1 60 60 36)
            output_dict=output_dict,
            metas=metas,
            test_mode=test_mode)

        return output_dict, depth2occ, predtoreturn

    def frame_split(self, grad_frames, imgs, metas):
        F = imgs.shape[1]
        index = np.random.permutation(F)
        inv_index = np.argsort(index)
        imgs_grad = imgs[:, index[:grad_frames]]
        imgs_no_grad = imgs[:, index[grad_frames:]]
        metas_grad = deepcopy(metas)
        metas_no_grad = deepcopy(metas)
        for meta, meta_grad, meta_no_grad in zip(metas, metas_grad, metas_no_grad):
            lidar2img = np.asarray(meta['lidar2img'])
            meta_grad['lidar2img'] = lidar2img[index[:grad_frames]]
            meta_no_grad['lidar2img'] = lidar2img[index[grad_frames:]]
            img_aug_matrix = meta['img_aug_matrix']
            meta_grad['img_aug_matrix'] = img_aug_matrix[index[:grad_frames]]
            meta_no_grad['img_aug_matrix'] = img_aug_matrix[index[grad_frames:]]

        return imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index

    def forward_autoreg(self,
                        imgs=None,
                        metas=None,
                        points=None,
                        label=None,
                        test_mode=True,
                        **kwargs,
        ):
        B, F, N, C, H, W = imgs.shape
        # 仅支持 bs=1 的原限制；保留作对照，不再执行。
        # assert B==1, 'bs > 1 not supported'

        bev = self.obtain_bev(imgs, metas)
        BF, G, C = bev.shape # bev is actually anchors
        bev = bev.reshape(B, F, G, C)

        output_dict = self.future_decoder.forward_autoreg(bev, metas)
        bev_predict = output_dict.pop('bev')
        output_dict = self.head(
            bev_feat=bev_predict,
            points=points,
            label=label,
            output_dict=output_dict,
            metas=metas,
            test_mode=test_mode)

        return output_dict
