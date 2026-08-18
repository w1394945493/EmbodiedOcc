import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from mmengine import MODELS
from mmengine.model import BaseModule
from ...encoder.gaussianformer.utils import \
    cartesian, safe_sigmoid, GaussianPrediction, get_rotation_matrix
import sys
import numpy as np
import matplotlib.pyplot as plt

@MODELS.register_module()
class GaussianOccHead(BaseModule):
    def __init__(
        self,
        empty_label=17, # 12
        num_classes=18, # 13
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=0.4),
        with_empty=False,
        empty_args=dict(),
        pc_range=[],
        scale_range=[],
        include_opa=True,
        semantics_activation='softmax'
    ):
        super().__init__()

        self.empty_label = empty_label
        self.num_classes = num_classes
        self.classes = list(range(num_classes))

        # sys.path.append(
        #     "/vepfs-mlp2/c20250502/haoce/wangyushen/EmbodiedOcc/model/head/gaussian_occ_head/ops/localagg"
        # )
        # from local_aggregate import LocalAggregator
        from model.head.gaussian_occ_head.ops.localagg.local_aggregate import LocalAggregator
        self.aggregator = LocalAggregator(**cuda_kwargs)

        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float))
            # self.empty_scalar = nn.Parameter(torch.tensor([10], dtype=torch.float))
            # self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.include_opa = include_opa
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = self.num_classes if not with_empty else self.num_classes - 1
        self.semantics_activation = semantics_activation

    def anchor2gaussian(self, anchor, metas):

        # vox_near = metas[0]['vox_origin']
        # scene_size = metas[0]['scene_size']
        # vox_far = vox_near + scene_size
        # nyu_pc_range = torch.cat([vox_near, vox_far], dim=0).to(anchor.device)

        # 仅支持 bs=1 的原实现：所有 anchor 都使用 metas[0] 的相机体素范围。
        # cam_vox_range = metas[0]['cam_vox_range'].to(anchor.device)
        # xyz = cartesian(anchor, cam_vox_range)
        # 支持 bs>1：每个样本使用自己的 [xmin, ymin, zmin, xmax, ymax, zmax]。
        cam_vox_range = torch.stack([m['cam_vox_range'] for m in metas]).to(anchor.device, anchor.dtype)
        xyz_01 = safe_sigmoid(anchor[..., :3])
        xyz = xyz_01 * (
            cam_vox_range[:, None, 3:] - cam_vox_range[:, None, :3]
        ) + cam_vox_range[:, None, :3]

        # xyz = cartesian(anchor, nyu_pc_range)
        gs_scales = safe_sigmoid(anchor[..., 3:6])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        rot = anchor[..., 6: 10]
        opas = safe_sigmoid(anchor[..., 10: (10 + int(self.include_opa))])
        shs = torch.zeros(*anchor.shape[:-1], 0, device=anchor.device, dtype=anchor.dtype)
        semantics = anchor[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]

        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)
        # import pdb; pdb.set_trace()
        # softrelu
        # semantics = F.softplus(semantics)

        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            harmonics=shs.unflatten(-1, (3, -1)),
            opacities=opas,
            semantics=semantics
        )
        return gaussian

    def prepare_gaussian_args(self, gaussians, metas):
        means = gaussians.means # b, g, 3
        b_, g_, _ = means.shape
        # 仅支持 bs=1 的原实现：展平 batch 后统一乘 metas[0] 的 cam2world。
        # means_flat = means.reshape(-1, 3)
        # means_cam = torch.cat((means_flat, torch.ones((means_flat.shape[0], 1), device=means.device)), dim=1).to(torch.float32)
        # cam2world = metas[0]['cam2world'].to(torch.float32)
        # means_world_ = (cam2world @ means_cam.unsqueeze(-1)).squeeze(-1)
        # means_world = means_world_[:, :3].reshape(b_, g_, 3)
        # 支持 bs>1：保留 [B, G]，为每个样本应用自己的 cam2world。
        means_cam = F.pad(means.to(torch.float32), (0, 1), value=1.0)
        cam2world = torch.stack([m['cam2world'] for m in metas]).to(means.device, torch.float32)
        means_world = torch.matmul(cam2world[:, None], means_cam[..., None]).squeeze(-1)[..., :3]
        # means_world_homogeneous = means_cam @ cam2world.T
        # means_world = means_world_homogeneous[:, :3] / means_world_homogeneous[:, 3][:, None]
        # means_world = torch.cat((means_world[:,1][:, None], means_world[:,0][:, None], means_world[:,2][:, None]), dim=-1)
        means = means_world
        # endfix
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1

        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            # 仅支持 bs=1 的原实现：只生成一个样本的 empty Gaussian，buffer 也保持 batch=1。
            # vox_origin = metas[0]['vox_origin']
            # scene_size = metas[0]['scene_size']
            # vox_center = vox_origin + scene_size / 2
            # self.empty_mean = vox_center[None, None, :]
            # 支持 bs>1：为每个样本分别生成 empty Gaussian，并扩展公共属性 buffer。
            vox_origin = torch.stack([m['vox_origin'] for m in metas]).to(means.device, means.dtype)
            scene_size = torch.stack([m['scene_size'] for m in metas]).to(means.device, means.dtype)
            empty_mean = (vox_origin + scene_size / 2)[:, None, :]
            # self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])

            # opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1) # FIXME
            opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1) # FIXME

            means = torch.cat([means, empty_mean], dim=1)
            scales = torch.cat([scales, self.empty_scale.expand(b_, -1, -1)], dim=1)
            rotations = torch.cat([rotations, self.empty_rot.expand(b_, -1, -1)], dim=1)
            empty_sem = self.empty_sem.expand(b_, -1, -1).clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            opacities = torch.cat([opacities, empty_sem], dim=1)
            # import pdb; pdb.set_trace()
            origi_opa = torch.cat([origi_opa, self.empty_opa.expand(b_, -1, -1)], dim=1)

        bs, g, _ = means.shape

        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]

        R = get_rotation_matrix(rotations) # b, g, 3, 3

        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)

        # 仅支持 bs=1 的原实现：所有 covariance 都使用 metas[0] 的旋转。
        # c2w_rot = metas[0]['cam2world'][:3, :3]
        # c2w_rot_T = metas[0]['cam2world'][:3, :3].T
        # c2w_rot = c2w_rot.unsqueeze(0).unsqueeze(0).repeat(bs, g, 1, 1).to(torch.float32)
        # c2w_rot_T = c2w_rot_T.unsqueeze(0).unsqueeze(0).repeat(bs, g, 1, 1).to(torch.float32)
        # 支持 bs>1：每个 batch 元素使用自己的相机到世界旋转矩阵。
        c2w_rot = cam2world[:, :3, :3].to(Cov.dtype).unsqueeze(1).expand(-1, g, -1, -1)
        c2w_rot_T = c2w_rot.transpose(-1, -2)
        Cov = torch.matmul(c2w_rot, torch.matmul(Cov, c2w_rot_T))

        CovInv = torch.linalg.inv(Cov.float()).to(Cov.device) # b, g, 3, 3
        return means, origi_opa, opacities, scales, CovInv

    def prepare_gt_xyz(self, metas, tensor):
        # gt_xyz = []
        # for meta in metas:
        #     gt_xyz.append(meta['occ_xyz'])
        # gt_xyz = torch.from_numpy(np.array(gt_xyz)).to(tensor.device, tensor.dtype)

        # 仅支持 bs=1 的原实现：只返回 metas[0] 的 occupancy 查询点。
        # gt_xyz = metas[0]['occ_xyz'].unsqueeze(0)
        # 支持 bs>1：堆叠 batch 内所有样本的 occupancy 查询点。
        gt_xyz = torch.stack([m['occ_xyz'] for m in metas], dim=0).to(tensor.device, tensor.dtype)
        # import pdb; pdb.set_trace()
        return gt_xyz

    def forward(self, bev_feat, points, label, output_dict, metas, test_mode=False):
        # means3D:
        # gt_xyz: b, x, y, z, 3
        # gt_label: b, x, y, z

        # sampled_xyz: b, n, 3
        # sampled_label: b, n
        # 仅支持 bs=1 的原限制；新实现保留 batch 维，因此不再执行。
        # assert bev_feat.shape[0] == 1
        anchors = bev_feat # [1, 1, 21600, 24]
        # 当前单目训练配置 num_frames=1；batch 与 frame 不混合，避免 metadata 对应关系丢失。
        B, Fm, G, _ = anchors.shape
        assert Fm == 1, '支持 bs>1 的 GaussianOccHead 当前要求 num_frames=1'
        gt_xyz = self.prepare_gt_xyz(metas, anchors)

        anchors = anchors[:, 0] # [B, G, C]
        gaussians = self.anchor2gaussian(anchors, metas)
        means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussians, metas)
        sampled_xyz = gt_xyz.flatten(1, 3).float()
        # 仅支持 bs=1 的原实现会 flatten opacity，并在随后用跨 batch mask 统一筛选。
        # origi_opa = origi_opa.flatten(1, 2)

        semantics = []
        # 仅支持 bs=1 的原实现：一个 mask 会把 batch 和 Gaussian 维混在一起，随后伪装成 batch=1。
        # nyu_pc_min = metas[0]['vox_origin']
        # nyu_pc_max = nyu_pc_min + metas[0]['scene_size']
        # mask = (...)
        # means = means[mask].unsqueeze(0)
        # origi_opa = origi_opa[mask].unsqueeze(0)
        # opacities = opacities[mask].unsqueeze(0)
        # scales = scales[mask].unsqueeze(0)
        # CovInv = CovInv[mask].unsqueeze(0)
        # 支持 bs>1：每个样本独立筛选并调用聚合器，允许有效 Gaussian 数量不同。
        epsilon = 1e-3
        for i in range(B):
            nyu_pc_min = metas[i]['vox_origin'].to(means.device, means.dtype)
            nyu_pc_max = nyu_pc_min + metas[i]['scene_size'].to(means.device, means.dtype)
            mask = (
                (means[i, :, 0] > nyu_pc_min[0] + epsilon) &
                (means[i, :, 0] < nyu_pc_max[0] - epsilon) &
                (means[i, :, 1] > nyu_pc_min[1] + epsilon) &
                (means[i, :, 1] < nyu_pc_max[1] - epsilon) &
                (means[i, :, 2] > nyu_pc_min[2] + epsilon) &
                (means[i, :, 2] < nyu_pc_max[2] - epsilon))
            origin_use = nyu_pc_min.to(torch.float32)
            semantic = self.aggregator(
                sampled_xyz[i:(i+1)], 
                means[i:i+1, mask],
                origi_opa[i:i+1, mask],
                opacities[i:i+1, mask],
                scales[i:i+1, mask],
                CovInv[i:i+1, mask],
                [metas[i]],
                origin_use) # n, c
            semantics.append(semantic)  # (129600 13)

        semantics = torch.stack(semantics, dim=0).transpose(1, 2) # [1, 13, 129600]
        spatial_shape = label.shape[2:] # [60, 60, 36]

        result_dict = {
            'ce_input': semantics.unflatten(-1, spatial_shape), # [1, 13, 60, 60, 36]
            # 仅支持 bs=1 的原实现：label.squeeze(0) 和 metas[0] 会丢失/忽略 batch。
            # 'ce_label': label.squeeze(0),
            # 'fov_mask': metas[0]['fov_mask'],
            # 支持 bs>1：只移除单帧维，保留 batch，并堆叠每个样本的可见区域。
            'ce_label': label[:, 0],
            'fov_mask': torch.stack([m['fov_mask'] for m in metas], dim=0).to(label.device),
            # 'fov_mask_4': metas[0]['fov_mask_4'],               # [15, 15, 9]
        }
        # import pdb; pdb.set_trace()
        output_dict.update(result_dict)

        output_dict.update({
                'gaussians': gaussians
            })
        return output_dict
