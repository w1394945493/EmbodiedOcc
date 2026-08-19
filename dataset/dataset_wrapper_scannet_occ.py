import numpy as np
import torch
from torch.utils import data
from . import OPENOCC_DATAWRAPPER
from dataset.transform_3d import PadMultiViewImage, NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage, ImageAug3D


img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

@OPENOCC_DATAWRAPPER.register_module()
class Scannet_Scene_Occ_DatasetWrapper(data.Dataset):
    def __init__(self, in_dataset, final_dim=[256, 704], resize_lim=[0.45, 0.55], phase='train'):
        self.dataset = in_dataset
        self.phase = phase
        if phase == 'train':
            transforms = [
                # *【主视觉分支尺寸确认】配置 final_dim=[480,640]，所以 imgs 在这里
                # * 被整理为 HxW=480x640；该变换不作用于 meta['img_depthbranch']。
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, is_train=True),
                PhotoMetricDistortionMultiViewImage(),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        else:
            transforms = [
                # *【主视觉分支尺寸确认】验证/推理同样使用 final_dim=[480,640]。
                ImageAug3D(final_dim=final_dim, resize_lim=resize_lim, is_train=False),
                NormalizeMultiviewImage(**img_norm_cfg),
                # * 480 和 640 已经是 32 的整数倍，因此该 padding 不改变空间尺寸。
                PadMultiViewImage(size_divisor=32)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        imgs, metas, occ = data

        # 这里只处理第一条主视觉分支 imgs：训练时包括 ImageAug3D、颜色扰动、
        # ImageNet 归一化和 padding。第二条 Depth Anything 输入已经保存在
        # metas['img_depthbranch']，不会经过这里的 PhotoMetricDistortion；它使用
        # dataset 中为 Depth Anything 单独定义的预处理，二者后续在 lifter 汇合。
        F, N, H, W, C = imgs.shape
        imgs_dict = {'img': imgs.reshape(F*N, H, W, C)}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = np.stack([img.transpose(2, 0, 1) for img in imgs], axis=0)
        FN, C, H, W = imgs.shape
        imgs = imgs.reshape(F, N, C, H, W)
        metas['img_shape'] = imgs_dict['img_shape']
        if imgs_dict.get('img_aug_matrix'):
            img_aug_matrix = np.stack(imgs_dict['img_aug_matrix'], axis=0)
            metas['img_aug_matrix'] = img_aug_matrix.reshape(F, N, 4, 4)
        
        data_tuple = (imgs, metas, occ)
        return data_tuple
