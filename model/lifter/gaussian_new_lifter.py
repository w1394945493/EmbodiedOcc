import numpy as np, math
import torch
from torch import nn
import numpy as np
from PIL import Image
from mmengine import MODELS
import torchvision.transforms as transforms
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import torch.nn.functional as F
from ..encoder.gaussianformer.utils import safe_sigmoid
from ..encoder.gaussianformer.utils import safe_get_quaternion, batch_quaternion_multiply, get_rotation_matrix, safe_sigmoid

LOGIT_MAX = 0.99

def depth2occ(points_world, vox_origin, scene_size):
    vox_near = vox_origin
    vox_far = vox_origin + scene_size
    delta = 1e-3
    points_inroom_mask = (points_world[..., 0] > (vox_near[0]+delta)) & (points_world[..., 0] < (vox_far[0]-delta)) & (points_world[..., 1] > (vox_near[1]+delta)) & (points_world[..., 1] < (vox_far[1]-delta)) & (points_world[..., 2] > (vox_near[2]+delta)) & (points_world[..., 2] < (vox_far[2]-delta))
    points_inroom = points_world[points_inroom_mask]
    grid_size = 0.08
    points_idx = ((points_inroom - vox_near) / grid_size).long()
    occ_label = torch.zeros(60, 60, 36, dtype=torch.float32).to(points_world.device)
    occ_label[points_idx[:, 0], points_idx[:, 1], points_idx[:, 2]] = 1
    return occ_label


def safe_inverse_sigmoid(tensor): # 逆 Sigmoid 函数
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))

def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = (
            num_bins
            * (torch.log(1 + depth_map) - math.log(1 + depth_min))
            / (math.log(1 + depth_max) - math.log(1 + depth_min))
        )
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds (-2, -1, 0, 1, ..., num_bins, num_bins +1) --> (num_bins, num_bins, 0, 1, ..., num_bins, num_bins)
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices.long()

def sample_3d_feature(feature_3d, pix_xy, pix_z, fov_mask):
    """
    Args:
        feature_3d (torch.tensor): 3D feature, shape (C, D, H, W).
        pix_xy (torch.tensor): Projected pix coordinate, shape (N, 2).
        pix_z (torch.tensor): Projected pix depth coordinate, shape (N,).
    
    Returns:
        torch.tensor: Sampled feature, shape (N, C)
    """
    pix_x, pix_y = pix_xy[:, 0][fov_mask], pix_xy[:, 1][fov_mask]
    pix_z = pix_z[fov_mask].to(pix_y.dtype)
    ret = feature_3d[:, pix_z, pix_y, pix_x].T
    return ret

class DepthAwareLayer(nn.Module):
    def __init__(self, embed_dim):
        super(DepthAwareLayer, self).__init__()
        self.fc1 = nn.Linear(2, 64)   
        self.fc2 = nn.Linear(64, 128) 
        self.fc3 = nn.Linear(128, embed_dim) 
        self.relu = nn.ReLU()       

    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)            
        return x


@MODELS.register_module()
class GaussianNewLifter(nn.Module):
    def __init__(
        self,
        embed_dims, # 96
        num_anchor=25600, # 21600
        anchor=None,
        anchor_grad=False, 
        feat_grad=False,
        semantic_dim=0, # 13
        include_opa=True,
        include_v=False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        elif anchor is None:
            total_anchor = num_anchor
            xyz = torch.rand(num_anchor, 3, dtype=torch.float)
            assert xyz.shape[0] == num_anchor
            xyz = safe_inverse_sigmoid(xyz)
    
            scale = torch.rand_like(xyz)
            scale = safe_inverse_sigmoid(scale)
            rots = torch.zeros(num_anchor, 4, dtype=torch.float)
            rots[:, 0] = 1
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((
                num_anchor, int(include_opa)), dtype=torch.float))
            semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
            self.semantic_dim = semantic_dim
            
            anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)

        self.num_anchor = total_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        
        self.instance_feature_layer = nn.Linear(
            3 + 3 + 4 + int(include_opa) + semantic_dim, embed_dims)
        
        self.depth_aware_layer = DepthAwareLayer(embed_dims)
        

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
    
    def forward(self, flag_depthbranch, flag_depthanything_as_gt, depthnet_output, mlvl_img_feats, metas):
        
        batch_size = mlvl_img_feats[0].shape[0]
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # 1, 16200, 23
        # 仅支持 bs=1 的原实现：所有坐标、内参、深度和旋转都固定读取 metas[0]，
        # 并通过 squeeze(0) 删除 batch 维；保留作实现对照，不再执行。
        # world_near = metas[0]['vox_origin']
        # world_far = metas[0]['vox_origin'] + metas[0]['scene_size']
        # anchor_xyz_logits = anchor[:, :, :3]
        # anchor_xyz_01 = safe_sigmoid(anchor_xyz_logits)
        # anchor_xyz_world = anchor_xyz_01 * (world_far - world_near) + world_near
        # anchor_xyz_world = anchor_xyz_world.squeeze(0)
        # world2cam = metas[0]['world2cam'].to(torch.float32)
        # anchor_xyz_world_ = torch.cat((anchor_xyz_world, torch.ones((anchor_xyz_world.shape[0], 1), device=anchor_xyz_world.device)), dim=1).to(torch.float32)
        # anchor_xyz_cam_ = (world2cam @ anchor_xyz_world_.unsqueeze(-1)).squeeze(-1)
        # anchor_xyz_cam = anchor_xyz_cam_[:, :3]
        # f_l_x = torch.tensor(metas[0]['cam_k'][0, 0]).cuda()
        # f_l_y = torch.tensor(metas[0]['cam_k'][1, 1]).cuda()
        # c_x = torch.tensor(metas[0]['cam_k'][0, 2]).cuda()
        # c_y = torch.tensor(metas[0]['cam_k'][1, 2]).cuda()
        # anchor_pix_x = f_l_x * anchor_xyz_cam[:, 0] / anchor_xyz_cam[:, 2] + c_x
        # anchor_pix_y = f_l_y * anchor_xyz_cam[:, 1] / anchor_xyz_cam[:, 2] + c_y
        # z = depthnet_output if flag_depthanything_as_gt else metas[0]['depth_gt']
        # anchor_depth_from_z = z[anchor_pix_y.long(), anchor_pix_x.long()]
        # nyu_pc_range = metas[0]['cam_vox_range']
        # anchor_points = points_cam.float().unsqueeze(0)
        # w2c_quat = safe_get_quaternion(metas[0]['world2cam'][:3, :3].unsqueeze(0)).squeeze(0)
        # anchor_rots_cam = batch_quaternion_multiply(w2c_quat, anchor_rots.squeeze(0)).unsqueeze(0)

        # 支持 bs>1 的实现：将每个样本的场景范围和相机参数堆叠，始终保留 batch 维。
        device = anchor.device
        dtype = anchor.dtype
        world_near = torch.stack([meta['vox_origin'] for meta in metas], dim=0).to(device=device, dtype=dtype)
        scene_size = torch.stack([meta['scene_size'] for meta in metas], dim=0).to(device=device, dtype=dtype)
        world2cam = torch.stack([meta['world2cam'] for meta in metas], dim=0).to(device=device, dtype=torch.float32)
        cam_k = torch.stack([meta['cam_k'] for meta in metas], dim=0).to(device=device, dtype=dtype)
        cam_vox_range = torch.stack([meta['cam_vox_range'] for meta in metas], dim=0).to(device=device, dtype=dtype)

        anchor_xyz_01 = safe_sigmoid(anchor[..., :3])
        anchor_xyz_world = anchor_xyz_01 * scene_size[:, None, :] + world_near[:, None, :]
        anchor_xyz_world_h = F.pad(anchor_xyz_world.to(torch.float32), (0, 1), value=1.0)
        anchor_xyz_cam_h = torch.matmul(world2cam[:, None], anchor_xyz_world_h[..., None]).squeeze(-1)
        anchor_xyz_cam = anchor_xyz_cam_h[..., :3].to(dtype)

        depth_safe = anchor_xyz_cam[..., 2].clamp_min(1e-6)
        anchor_pix_x = cam_k[:, None, 0, 0] * anchor_xyz_cam[..., 0] / depth_safe + cam_k[:, None, 0, 2]
        anchor_pix_y = cam_k[:, None, 1, 1] * anchor_xyz_cam[..., 1] / depth_safe + cam_k[:, None, 1, 2]

        if flag_depthbranch:
            if flag_depthanything_as_gt:
                z = depthnet_output
            else:
                z = torch.stack([meta['depth_gt'] for meta in metas], dim=0)
        else:
            raise RuntimeError('GaussianNewLifter 需要深度输入，当前 flag_depthbranch=False。')

        # 允许深度分支返回 [B,1,H,W]，统一为 [B,H,W] 后逐 batch 索引。
        if z.ndim == 4 and z.shape[1] == 1:
            z = z[:, 0]
        z = z.to(device=device)
        depth_h, depth_w = z.shape[-2:]
        anchor_pix_x = anchor_pix_x.clamp(0, depth_w - 1).long()
        anchor_pix_y = anchor_pix_y.clamp(0, depth_h - 1).long()
        batch_index = torch.arange(batch_size, device=device)[:, None]
        anchor_depth_from_z = z[batch_index, anchor_pix_y, anchor_pix_x]
        anchor_depth_real = anchor_xyz_cam[..., 2]
        anchor_depth_feature = self.depth_aware_layer(
            torch.stack((anchor_depth_from_z, anchor_depth_real), dim=-1))

        range_min = cam_vox_range[:, None, :3]
        range_max = cam_vox_range[:, None, 3:]
        points_cam = torch.maximum(torch.minimum(anchor_xyz_cam, range_max), range_min)
        anchor_points = (points_cam - range_min) / (range_max - range_min).clamp_min(1e-6)
        anchor_points = anchor_points.float().to(device)

        anchor_points_ = anchor[..., 3:].clone()
        anchor_rots = anchor_points_[..., 3:7]
        # safe_get_quaternion 内部包含标量分支，逐样本计算后再堆叠，支持 bs>1。
        w2c_quats = torch.stack([
            safe_get_quaternion(world2cam[b, :3, :3].unsqueeze(0)).squeeze(0)
            for b in range(batch_size)
        ], dim=0)
        anchor_rots_cam = torch.stack([
            batch_quaternion_multiply(w2c_quats[b], anchor_rots[b])
            for b in range(batch_size)
        ], dim=0)
        anchor_points_[..., 3:7] = anchor_rots_cam
        
        anchor_points = torch.cat([
            safe_inverse_sigmoid(torch.clamp(anchor_points, 0.001, 0.999)),
            anchor_points_
        ], dim=-1)
        
        anchor = anchor_points
        
        instance_feature = self.instance_feature_layer(anchor)
        # 仅支持 bs=1 的原实现：anchor_depth_feature 没有 batch 维。
        # instance_feature = instance_feature + anchor_depth_feature.unsqueeze(0)
        # 支持 bs>1：深度特征已经是 [B, G, C]，可与实例特征逐样本相加。
        instance_feature = instance_feature + anchor_depth_feature
        
        return anchor, instance_feature, None, None, None
