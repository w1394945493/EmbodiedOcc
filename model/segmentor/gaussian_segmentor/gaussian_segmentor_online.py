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
import open3d as o3d
from ...encoder.gaussianformer.utils import safe_sigmoid

@MODELS.register_module()
class GaussianSegmentorOnline(BaseModule):

    def __init__(
        self,
        reuse_instance_feature=True,
        flag_depthbranch=False,
        flag_depthanything_as_gt=False,
        gaussian_scale_max=0.4,
        semantic_dim=12,
        depthbranch=None,
        backbone=None,
        neck=None,
        lifter=None,
        encoder=None,
        future_decoder=None,
        head=None, 
        globalhead=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.reuse_instance_feature = reuse_instance_feature
        self.flag_depthbranch = flag_depthbranch
        self.flag_depthanything_as_gt = flag_depthanything_as_gt
        self.gaussian_scale_max = gaussian_scale_max
        self.semantic_dim = semantic_dim    
        if flag_depthbranch:
            if flag_depthanything_as_gt:
                # depth branch
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
            
            basemodel_name = "tf_efficientnet_b7_ns"
            num_features = 2560
            print("Loading base model ()...".format(basemodel_name), end="")
            # basemodel = torch.hub.load(
            #     "/home/wyq/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master", basemodel_name, pretrained=True, trust_repo=True, source='local'
            # )
            basemodel = torch.hub.load(
                "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
            )
            print("Done.")
            # Remove last layer
            print("Removing last two layers (global_pool & classifier).")
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()
            # # self.unet_encoder = basemodel
            self.backbone = basemodel
            # self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
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
            # basemodel = torch.hub.load(
            #     "/home/wyq/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master", basemodel_name, pretrained=True, trust_repo=True, source='local'
            # )
            basemodel = torch.hub.load(
                "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
            )
            print("Done.")
            # Remove last layer
            print("Removing last two layers (global_pool & classifier).")
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()
            # # self.unet_encoder = basemodel
            self.backbone = basemodel
            # self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
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
        if globalhead is not None:
            self.globalhead = MODELS.build(globalhead)
         

    def extract_img_feat(self, imgs):
        # Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth" to /home/wyq/.cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W) # 1, 3, 480, 640
        
        feature_x = [imgs]
        feature_idx = 0
        this_x = feature_x[-1]
        for k, v in self.backbone._modules.items():
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
            
        img_feats_backbone = feature_x
        
        # list of [2560, 15, 20]
        img_feats_out = self.neck(img_feats_backbone) # dict
        
        img_feats_reshaped = []
        for img_feat in img_feats_out.values():
            BN, C, H, W = img_feat.size()
            if W != 640:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
                
        return img_feats_reshaped, img_feats_out['1_1'] # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]
    
    def obtain_bev(self, imgs, metas, scenemeta):
        B, F, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B*F, N, C, H, W)
        
        mlvl_img_feats, feature_x_4 = self.extract_img_feat(imgs) # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]
        
        if self.flag_depthbranch:
            if self.flag_depthanything_as_gt:
                # depth branch
                self.depthanything.eval()
                image_ = metas[0]['img_depthbranch']
                depth_pred = self.depthanything.infer_image(image_, 480, 640, 480)
                depthnet_output = depth_pred
            else:  
                depthnet_output = None
        else:
            depthnet_output = None
        
        # 【在线全局记忆：读】self.gaussian_tensor 是当前场景持续维护的 Gaussian 池。
        # 其中 Gaussian 的中心、旋转均以统一的世界坐标系保存，因此相机移动后不需要
        # 搬动整张地图；当前帧只需在 lifter 中依据 world2cam 取出并对齐附近 Gaussian。
        gaussian_pool = self.gaussian_tensor # 1, N, 25（含 predicted/splat 两个状态标记）
        global_mask_thistime = self.global_mask_thistime
        instance_feature_pool = self.gaussian_instance_feature
        anchor, instance_feature, depth2occ, depthnet_output_loss, predtoreturn, gaussian_pool_new, anchor_new_tag, instance_feature_pool_new = self.lifter(scenemeta, gaussian_pool, instance_feature_pool, global_mask_thistime, self.flag_depthbranch, self.flag_depthanything_as_gt, depthnet_output, mlvl_img_feats, metas)    # b, g, c 
        # lifter 已从池中暂时移除“当前可见、即将被重新预测”的旧 Gaussian；
        # 不在当前视野内的历史 Gaussian 原样留在 gaussian_pool_new 中。
        # 当前帧细化后的版本会在 forward 返回后由 scene_update() 写回。
        self.gaussian_tensor = gaussian_pool_new
        self.gaussian_instance_feature = instance_feature_pool_new
        
        anchor, instance_feature_cache = self.encoder(anchor, instance_feature, mlvl_img_feats, metas, anchor_new_tag) # b, g, c
        
        return anchor, depth2occ, depthnet_output_loss, predtoreturn, anchor_new_tag, instance_feature_cache
    
    def scene_init(self, scenemeta):
        # 【在线全局记忆：按场景初始化】每进入一个新 ScanNet 场景都重新调用本函数，
        # 因而 Gaussian memory 只在同一场景的视频帧之间持续，不会跨场景传播。
        self.scene_name = scenemeta['scene_name']
        self.global_scene_dim = scenemeta['global_scene_dim']
        self.global_scene_size = scenemeta['global_scene_size']
        self.global_labels = scenemeta['global_labels'] # [x_dim, y_dim, z_dim]
        self.global_xyz = scenemeta['global_pts']
        self.global_scene_origin = scenemeta['global_scene_origin']
        self.K_frams = len(scenemeta['valid_img_paths'])
        device = torch.device('cuda')
        self.global_mask_thistime = torch.zeros_like(scenemeta['global_mask']).to(device).to(torch.bool)
        
        local_meter = 0.16
        self.gaussian_random_num = int(scenemeta['global_scene_size'][0]/local_meter) * int(scenemeta['global_scene_size'][1]/local_meter) * int(scenemeta['global_scene_size'][2]/local_meter)
        
        # 在“完整场景的世界坐标包围盒”内预置随机 Gaussian 种子。
        # 后续相机运动到某一区域时，lifter 才取出该区域的种子并结合当前观测细化；
        # 尚未被观察的种子仍保留在池中，但不会参与最终全局 Occ 聚合。
        xyz = torch.rand((self.gaussian_random_num, 3), dtype=torch.float32).to(device)
        xyz_world = xyz * self.global_scene_size + self.global_scene_origin
        xyz_world = xyz_world.to(dtype=torch.float32)
        scale = torch.rand_like(xyz).to(device)
        rots = torch.zeros(self.gaussian_random_num, 4, dtype=torch.float).to(device)
        rots[:, 0] = 1
        opacity = 0.1 * torch.ones((self.gaussian_random_num, int(True)), dtype=torch.float).to(device)
        semantic = torch.randn(self.gaussian_random_num, self.semantic_dim, dtype=torch.float).to(device)
        self.gaussian_tensor = torch.cat([xyz_world, scale, rots, opacity, semantic], dim=-1).unsqueeze(0).to(device)
        # with tag
        self.gaussian_tensor_flag = torch.zeros((self.gaussian_tensor.shape[0], self.gaussian_tensor.shape[1], 1), dtype=torch.float).to(device)
        self.splat_flag = torch.zeros((self.gaussian_tensor.shape[0], self.gaussian_tensor.shape[1], 1), dtype=torch.float).to(device)
        self.gaussian_tensor = torch.cat([self.gaussian_tensor, self.gaussian_tensor_flag], dim=-1)
        self.gaussian_tensor = torch.cat([self.gaussian_tensor, self.splat_flag], dim=-1)
        self.gaussian_instance_feature = torch.randn((1, self.gaussian_random_num, 96), dtype=torch.float).to(device)
    
    def scene_update(self, scenemeta, gaussianfromhead, instance_feature_fromhead, global_mask_from_thisframe):
        # 【在线全局记忆：写】当前帧 head 输出的是已经由 cam2world 转回世界坐标的
        # Gaussian。这里将其写回场景级池，供后续帧继续读取和细化。
        # detach 明确切断跨帧计算图：数值状态会跨帧保存，但第 t+1 帧损失不能沿着
        # memory 反向优化第 t 帧生成 Gaussian 的过程。
        gaussianfromhead = gaussianfromhead.detach()
        instance_feature_fromhead = instance_feature_fromhead.detach()
        gaussianfromhead_xyz = gaussianfromhead[..., :3]
        scene_near = self.global_scene_origin
        scene_far = scene_near + self.global_scene_size
        
        # 只允许完整场景包围盒内的预测进入全局池，避免异常 Gaussian 污染地图。
        mask = (gaussianfromhead_xyz[..., 0] >= scene_near[0]) & (gaussianfromhead_xyz[..., 0] <= scene_far[0]) & (gaussianfromhead_xyz[..., 1] >= scene_near[1]) & (gaussianfromhead_xyz[..., 1] <= scene_far[1]) & (gaussianfromhead_xyz[..., 2] >= scene_near[2]) & (gaussianfromhead_xyz[..., 2] <= scene_far[2])
        
        gaussianfromhead_add = gaussianfromhead[mask].unsqueeze(0)
        instance_feature_fromhead_add = instance_feature_fromhead[mask].unsqueeze(0)
        
        # 当前视野内的旧版本已在 lifter 中移除，所以这里的 cat 整体上相当于
        # “保留不可见历史区域 + 写回当前区域的新版本”，而不是无条件累积所有旧版本。
        self.gaussian_tensor = torch.cat([self.gaussian_tensor, gaussianfromhead_add], dim=1)
        self.gaussian_instance_feature = torch.cat([self.gaussian_instance_feature, instance_feature_fromhead_add], dim=1)
        global_mask_from_thisframe = global_mask_from_thisframe.to(dtype=torch.bool)
        # 累积截至当前帧曾经被观察过的全局体素区域，供最终全局评估/聚合使用。
        self.global_mask_thistime = self.global_mask_thistime | global_mask_from_thisframe
        
    
    def get_global_occ(self, scenemeta, vox_origin, scene_size):
        # 【全局 Occ 聚合】该函数不再读取当前 RGB，而是把截至当前帧累计得到的
        # 世界坐标 Gaussian 一次性 splat/聚合到完整场景级体素网格中。
        scene_result_dict = dict()
        bev = self.gaussian_tensor # [1, N, 24]
        bev_tag = bev[..., -1]
        bev = bev[..., :-2]
        # splat_flag==1 表示该 Gaussian 所在区域至少被某一帧实际处理过；
        # 排除 scene_init 中尚未观察、仍为随机值的全局 Gaussian 种子。
        bev_valid = bev[bev_tag == 1]
        
        scene_result_dict = self.globalhead(
                        bev_feat=bev_valid.unsqueeze(0).unsqueeze(0),  # [1, 1, N, 23]
                        points=None, 
                        label=self.global_labels.unsqueeze(0).unsqueeze(0), 
                        output_dict=scene_result_dict, 
                        metas=scenemeta,
                        test_mode=False,
                        label_mask=self.global_mask_thistime,
                        v_origin=vox_origin,
                        s_size=scene_size)
        return scene_result_dict
    
    def get_global_gaussian(self, scenemeta, vox_origin, scene_size):
        scene_result_dict = dict()
        bev = self.gaussian_tensor # [1, N, 24]

        bev = bev[..., :-2].squeeze(0)
        bev_world_xyz = bev[..., :3]
        scene_near = self.global_scene_origin
        scene_far = scene_near + self.global_scene_size
        epislon = 1e-3
        mask1 = (bev_world_xyz[..., 0] > scene_near[0]+epislon) & (bev_world_xyz[..., 0] < scene_far[0]-epislon) & (bev_world_xyz[..., 1] > scene_near[1]+epislon) & (bev_world_xyz[..., 1] < scene_far[1]-epislon) & (bev_world_xyz[..., 2] > scene_near[2]+epislon) & (bev_world_xyz[..., 2] < scene_far[2]-epislon)
        bev = bev[mask1]
        bev_world_xyz = bev[..., :3]
        bev_world_idx = torch.floor((bev_world_xyz - scene_near) / 0.08).to(torch.long)
        
        global_mask = scenemeta['global_mask'].to(torch.float)
        mask2 = (global_mask[bev_world_idx[:, 0], bev_world_idx[:, 1], bev_world_idx[:, 2]]==1)
        bev = bev[mask2]
        gaussians = self.globalhead.forward_gaussian(
                        # bev_feat=self.gaussian_tensor[..., :-1].unsqueeze(0),  # [1, 1, N, 23]
                        bev_feat=bev.unsqueeze(0).unsqueeze(0),  # [1, 1, N, 23]
                        # bev_feat=bev_valid.unsqueeze(0).unsqueeze(0),  # [1, 1, N, 23]
                        points=None, 
                        label=self.global_labels.unsqueeze(0).unsqueeze(0), 
                        output_dict=scene_result_dict, 
                        metas=scenemeta,
                        test_mode=False,
                        label_mask=self.global_mask_thistime,
                        v_origin=vox_origin,
                        s_size=scene_size)
        return gaussians
    
    
    def forward(
        self,
        scenemeta=None,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        grad_frames=None,
        test_mode=False,
        **kwargs,
    ):
        B, F, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'
        if grad_frames is not None:
            assert grad_frames < F
            imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index = self.frame_split(grad_frames, imgs, metas)
            bev_grad = self.obtain_bev(imgs_grad, metas_grad)
            with torch.no_grad():
                bev_no_grad = self.obtain_bev(imgs_no_grad, metas_no_grad)
            bev = torch.cat([bev_grad, bev_no_grad], dim=0)[inv_index]
        else:
            bev, depth2occ, depthnet_output_loss, predtoreturn, anchor_new_tag, instance_feature_cache = self.obtain_bev(imgs, metas, scenemeta)

        # BF, H, W, C = bev.shape
        BF, G, C = bev.shape # bev is actually anchors [1, 21600, 24]
        bev = bev.reshape(B, F, G, C)
        if hasattr(self, 'future_decoder'):
            output_dict = self.future_decoder(bev, metas)
            bev_predict = output_dict.pop('bev')
        else:
            bev_predict = bev
            output_dict = dict()
        
        output_dict, gaussianstensor_to_return, instance_feature_toreturn, gaussians_to_vis = self.head(
            bev_feat=bev_predict,  # [1, 1, 21600, 23]
            points=points, 
            label=label, 
            output_dict=output_dict, 
            metas=metas,
            test_mode=test_mode,
            anchor_new_tag=anchor_new_tag,
            instance_feature_cache=instance_feature_cache,)
        
        return output_dict, depth2occ, predtoreturn, gaussianstensor_to_return, instance_feature_toreturn, gaussians_to_vis
    
    
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
        assert B==1, 'bs > 1 not supported'
        
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
