import os
import json
import glob
import numpy as np
import numba as nb
import torch
from torch.utils import data
import pickle
from PIL import Image
from mmcv.image.io import imread
import copy
from pyquaternion import Quaternion
from . import OPENOCC_DATASET
from dataset.nyu_utils import vox2pix
from torchvision import transforms
from mmcv.image.io import imread
import math, cv2
from torchvision.transforms import Compose
from dataset.transform_ import Resize, NormalizeImage, PrepareForNet

@OPENOCC_DATASET.register_module()
class Scannet_Scene_OpenOccupancy_Dataset(data.Dataset):
    def __init__(
        self,
        data_path, 
        num_frames=1,
        offset=0,
        grid_size_occ=[60, 60, 36],
        coarse_ratio=2,
        empty_idx=0,
        phase='train',
        num_pts=21600,
        data_tg='base'
        ):

        self.occscannet_root = data_path
        self.phase = phase

        self.num_frames = num_frames
        self.offset = offset
        self.grid_size_occ = grid_size_occ
        self.grid_size_occ_coarse = (np.array(grid_size_occ) // coarse_ratio).astype(np.uint32)
        self.coarse_ratio = coarse_ratio
        self.empty_idx = empty_idx
        self.phase = phase

        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        if data_tg == 'base':
            subscenes_list = f'{self.occscannet_root}/{self.phase}_final.txt'
        elif data_tg == 'mini':
            subscenes_list = f'{self.occscannet_root}/{self.phase}_mini_final.txt'
        with open(subscenes_list, 'r') as f:
            self.used_subscenes = f.readlines()
            for i in range(len(self.used_subscenes)):
                self.used_subscenes[i] = f'{self.occscannet_root}/' + self.used_subscenes[i].strip()
        #!!!
        # self.used_subscenes = self.used_subscenes[:20]
        
        self.num_pts = num_pts

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.used_subscenes)

    def __getitem__(self, index):
        # *【单帧流程 0：读取一个局部 Occupancy 样本】
        # * train_mono 配置 num_frames=1，因此一个 index 对应 ScanNet 场景中的
        # * 一个相机时间步：读取 RGB、深度、相机参数和 60x60x36 局部体素标签；
        # * 这里不会读取或缓存历史帧。
        name = self.used_subscenes[index]
        with open(name, 'rb') as f:
            data = pickle.load(f)

        name_without_ext = os.path.splitext(name)[0]
        this_name = name_without_ext.split('gathered_data/')[-1]

        meta = {}
        meta['name'] = this_name # 'scene0000_00/00000'
        meta['scene_size'] = self.scene_size
        cam_pose = data['cam_pose']
        # cam2world/world2cam 后续用于初始化 Gaussian、投影查询图像以及把最终
        # Gaussian 转回世界坐标，是两条 RGB 分支共享的几何标定。
        meta['cam2world'] = cam_pose
        world2cam = np.linalg.inv(cam_pose)
        meta['world2cam'] = world2cam

        rgb_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.jpg'
        depth_path = f'{self.occscannet_root}/posed_images/' + f'{this_name}.png'
        depth_gt_np = Image.open(depth_path).convert('I;16')
        depth_gt_np = np.array(depth_gt_np) / 1000.0

        # ================================================================ #
        # 第二条 RGB 输入：专供冻结的 Depth Anything 深度分支。
        # 它与下方 N_img 使用的是同一个 rgb_path，但采用 Depth Anything 自己的
        # Resize/Normalize/PrepareForNet 预处理，并作为 meta['img_depthbranch']
        # 单独传给模型；不会作为 GaussianFormer 的多尺度视觉特征输入。
        # ================================================================ #
        transform = Compose([
            # *【Depth 分支网络输入尺寸】这里的 width/height=480 不是强制输出
            # * 480x480。keep_aspect_ratio=True 会保持前面 640:480 的宽高比，
            # * resize_method='lower_bound' 保证两边不小于 480；随后
            # * ensure_multiple_of=14 将宽高调整到 ViT patch size 14 的整数倍。
            # * 因而当前 480x640 图像实际会变成 HxW=490x644，再输入 Depth Anything。
            Resize(
                width=480,
                height=480,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        img_depthbranch = cv2.imread(rgb_path)
        # *【Depth 分支尺寸流-第1步】先把原始 RGB 显式缩放成 HxW=480x640。
        img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
        img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0
        # *【Depth 分支尺寸流-第2步】执行上面的 Depth Anything 专用 Resize 后，
        # * sample['image'] 为 CHW=3x490x644，而不是 3x480x640。
        sample = transform({'image': img_depthbranch, 'depth': depth_gt_np})
        img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0)
        depth_gt_np = torch.from_numpy(sample['depth']).unsqueeze(0)
        meta['depth_gt_np'] = depth_gt_np
        depth_valid_mask = (torch.isnan(depth_gt_np) == 0)
        depth_gt_np[depth_valid_mask == 0] = 0
        # train_mono.py 会把该 Tensor 搬到 GPU；GaussianSegmentor.obtain_bev()
        # 随后将它送入 depthanything.infer_image() 得到当前帧深度先验。
        meta['img_depthbranch'] = img_depthbranch
        meta['depth_gt_np_valid'] = depth_gt_np

        meta['rgb_path'] = rgb_path
        # ================================================================ #
        # 第一条 RGB 输入：主视觉/Gaussian Occupancy 分支。
        # 该图像会作为 dataset 返回值 imgs，随后再经过 DatasetWrapper 的图像增强
        # 与归一化，并输入 EfficientNet-B7 + DecoderBN 提取多尺度视觉特征。
        # 注意两条分支源于同一 RGB 文件，但分别维护输入，预处理并不完全相同。
        # ================================================================ #
        N_img = []
        this_img = imread(rgb_path, 'unchanged').astype(np.float32)
        this_H, this_W, _ = this_img.shape
        # *【主视觉分支网络输入尺寸】主分支直接将同一张 RGB 缩放成
        # * HxW=480x640；后续 DatasetWrapper 的 final_dim 也是 [480,640]，
        # * 且 480/640 均可被 padding divisor=32 整除，因此 EfficientNet 最终
        # * 实际接收到的仍是 480x640，不会像 Depth 分支那样变成 490x644。
        new_H, new_W = 480, 640
        # resize
        new_img = cv2.resize(this_img, (new_W, new_H))
        W_factor = new_W / this_W
        H_factor = new_H / this_H
        N_img.append(new_img)
        img = np.stack(N_img, 0) # [1, 968, 1296, 3]
        this_H, this_W= new_H, new_W
        img = [img] # [1, 1, 968, 1296, 3]

        cam_intrin = data['intrinsic']
        cam_intrin[0, 0] *= W_factor
        cam_intrin[0, 2] *= W_factor
        cam_intrin[1, 1] *= H_factor
        cam_intrin[1, 2] *= H_factor

        meta['cam_k'] = cam_intrin[:3, :3]
        viewpad = np.eye(4)
        viewpad[:meta['cam_k'].shape[0], :meta['cam_k'].shape[1]] = meta['cam_k']
        meta['cam2img'] = viewpad
        world2img = (viewpad @ world2cam)
        meta['world2img'] = world2img

        meta['depth_path'] = depth_path
        depth_gt = Image.open(depth_path).convert('I;16')
        depth_gt = np.array(depth_gt) / 1000.0
        meta['depth_gt'] = depth_gt

        vox_origin = data["voxel_origin"]
        meta['vox_origin'] = np.round(np.array(vox_origin, dtype=np.float32), 4)
        target = data["target_1_4"] # 60, 60, 36
        target = np.transpose(target, (1, 0, 2))
        # 把代表unknown的255换成0，把代表空的0换成12
        target[target == 0] = 12
        target[target == 255] = 0 
        occ = target # (60, 60, 36)
        nonemptymask = (occ != 12)
        occ = [occ] # [1, 60, 60, 36]

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z, occ_xyz = vox2pix(
            world2cam,
            meta['cam_k'],
            meta['vox_origin'],
            self.voxel_size,
            this_W,
            this_H,
            self.scene_size,
            dim_60_60_36=True,
        )
        _, fov_mask_4, _, _ = vox2pix(
            world2cam,
            meta['cam_k'],
            meta['vox_origin'],
            self.voxel_size * 4,
            this_W,
            this_H,
            self.scene_size,
            dim_60_60_36=False,
        )
        meta['projected_pix'] = projected_pix
        meta['fov_mask'] = fov_mask.reshape(60, 60, 36)
        meta['fov_mask_4'] = fov_mask_4.reshape(15, 15, 9)

        meta['pix_z'] = pix_z
        meta['occ_xyz'] = occ_xyz.reshape(60, 60, 36, 3)

        vox_near = meta['vox_origin']
        vox_far = vox_near + meta['scene_size']
        nyu_pc_range = np.concatenate([vox_near, vox_far], axis=0)
        meta['nyu_pc_range'] = nyu_pc_range

        scan = meta['occ_xyz'][nonemptymask]
        meta['occ_xyz_nonempty'] = scan
        meta['num_depth'] = self.num_pts
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.01
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], nyu_pc_range[0], nyu_pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], nyu_pc_range[1], nyu_pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], nyu_pc_range[2], nyu_pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]

        scan[:, 0] = (scan[:, 0] - nyu_pc_range[0]) / (nyu_pc_range[3] - nyu_pc_range[0])
        scan[:, 1] = (scan[:, 1] - nyu_pc_range[1]) / (nyu_pc_range[4] - nyu_pc_range[1])
        scan[:, 2] = (scan[:, 2] - nyu_pc_range[2]) / (nyu_pc_range[5] - nyu_pc_range[2])

        meta['anchor_points'] = scan

        cam_vox_near = np.array([-5, -6, -3])
        cam_vox_far = np.array([5, 6, 8])
        cam_vox_range = np.concatenate([cam_vox_near, cam_vox_far], axis=0).astype(np.float32)
        meta['cam_vox_range'] = cam_vox_range

        meta['occ_mask_valid'] = (occ != 0)
        meta['occ_mask_valid_fov'] = (occ != 0) & fov_mask
        meta['label'] = occ
        imgs = np.stack(img, 0)
        occs = np.stack(occ, 0)
        data_tuple = (imgs, meta, occs)
        return data_tuple

    def get_meshgrid(self, ranges, grid, reso):
        pass

    def get_data_info(self, info):
        pass

    def get_scene_index(self, scene_name=None):
        pass
