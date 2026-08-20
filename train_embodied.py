import os, time, argparse, os.path as osp, numpy as np
import torch
import gc
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torchrun --nproc_per_node=4 train_embodied.py
from utils.iou_eval import IOUEvalBatch
from utils.iou_as_iso import SSCMetrics
from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2, revise_ckpt_notddp

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
from mmengine.registry import MODELS
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")
import sys
# sys.path.append('/data1/code/wyq/gaussianindoor/EmbodiedOcc')
# sys.path.append('/data1/code/wyq/gaussianindoor/EmbodiedOcc/Depth-Anything-V2/metric_depth')
from PIL import Image

def pass_print(*args, **kwargs):
    pass

def is_main_process():
    if not dist.is_available():
        return True
    elif not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0

def main(args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    set_random_seed(cfg.seed)
    cfg.work_dir = args.work_dir
    max_num_epochs = cfg.max_epochs
    eval_freq = cfg.eval_freq
    print_freq = cfg.print_freq

    # # init DDP
    # distributed = True
    # world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
    # rank = int(os.environ["RANK"])  # node id
    # gpu = int(os.environ['LOCAL_RANK'])

    # dist.init_process_group(
    #     backend="nccl", init_method=f"env://",
    #     world_size=world_size, rank=rank
    # )

    # # dist.barrier()
    # torch.cuda.set_device(gpu)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        # init DDP
        # distributed = True
        world_size = int(os.environ["WORLD_SIZE"])  # number of nodes
        rank = int(os.environ["RANK"])  # node id

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("分布式训练需要至少一张可用的 CUDA GPU")
        # 仅支持特定启动方式的原实现：后续 DDP 使用了未定义的 gpu。
        # gpu = rank % num_gpus
        # 支持 torchrun 单机/多机：LOCAL_RANK 才是当前节点上的 CUDA 设备编号。
        gpu = int(os.environ.get("LOCAL_RANK", rank % num_gpus))
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        # torch.cuda.set_device(0)

    if not is_main_process():
        import builtins
        builtins.print = pass_print

    # configure logger
    if is_main_process():
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='indoor_nyu', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)

    if cfg.flag_depthanything_as_gt:
        my_model.depthanything.requires_grad_(False)
    my_model.globalhead.requires_grad_(False)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        if cfg.get('track_running_stats', False):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()

    # 后续需要调用 scene_init/update_global_mask 等 Online 专用方法；DDP 包装后
    # 这些方法位于 my_model.module，因此保留一份解包后的 model 引用。
    model = my_model.module if distributed else my_model

    print('done ddp model')
    # build dataloader
    from dataset import build_dataloader
    train_dataset_loader, val_dataset_loader = \
        build_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_wrapper_config,
            cfg.val_wrapper_config,
            cfg.train_loader_config,
            cfg.val_loader_config,
            dist=distributed,
        )

    # get optimizer, loss, scheduler
    amp = cfg.get('amp', True)
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    from loss import GPD_LOSS
    loss_func = GPD_LOSS.build(cfg.loss).cuda()
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=500, # FIXME
        warmup_lr_init=1e-6,
        t_in_epochs=False
    )

    CalMeanIou = SSCMetrics(n_classes=12)
    CalMeanIou_Fov = SSCMetrics(n_classes=12)
    CalMeanIou_Global = SSCMetrics(n_classes=12)
    # resume and load
    epoch = 0
    best_val_iou = 0
    best_val_miou = 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_iou' in ckpt:
            best_val_iou = ckpt['best_val_iou']
        if 'best_val_miou' in ckpt:
            best_val_miou = ckpt['best_val_miou']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        if not distributed:
            state_dict = revise_ckpt_notddp(state_dict)
        else:
            state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))

    scenemeta_keys = ['global_scene_dim', 'global_scene_size', 'global_labels', 'global_pts', 'global_scene_origin', 'global_mask']
    metas_tensor_keys_inv = ['name', 'cam2img', 'world2img', 'rgb_path', 'depth_path','num_depth', 'occ_mask_valid', 'img_shape', 'img_aug_matrix', 'img_depthbranch']

    if is_main_process():
        my_writer = SummaryWriter(args.work_dir)

    # training
    while epoch < max_num_epochs:
        my_model.train()
        CalMeanIou_Global.reset()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_record = LossRecord(loss_func=loss_func)
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].cuda()
            (imgs, metas, labels) = data # imgs [1, 1, 30, 3, 480, 640]  labels [1, 30, 60, 60, 36]
            scenemeta = metas[0]
            for k, v in scenemeta.items():
                if k in scenemeta_keys:
                    scenemeta[k] = torch.tensor(v).cuda()
            
            K_Frames = len(scenemeta['monometa_list'])
            monometa_list_cuda = []
            for i in range(K_Frames):
                monometa = scenemeta['monometa_list'][i]
                for k, v in monometa.items():
                    if not (k in metas_tensor_keys_inv):
                        monometa[k] = torch.tensor(v).cuda()
                monometa['img_depthbranch'] = monometa['img_depthbranch'].cuda()
                monometa_list_cuda.append(monometa)

            # forward + backward + optimize
            data_time_e = time.time()

            # 【在线场景开始】为当前场景新建世界坐标 Gaussian memory；该状态随后在
            # K_Frames 个连续视角之间持续更新，处理下一个场景时会重新初始化。
            model.scene_init(scenemeta) 

            for i in range(K_Frames):   # K_Frames:30
                # 每次只向网络输入当前一帧；历史 RGB/视觉特征不会组成队列，
                # 历史信息来自模型内部持续维护的世界坐标 Gaussian memory。
                img = imgs[:, :, i, :, :, :].unsqueeze(2) # 1, 1, 1, 3, H, W    # (1 1 1 3 480 640)
                label = labels[:, i, :, :, :].unsqueeze(1) # 1, 1, 60, 60, 36   # (1 1 60 60 36)
                meta = [monometa_list_cuda[i]]

                with torch.cuda.amp.autocast(enabled=amp):
                    #* result_dict 是当前帧局部 Occupancy：只由当前局部体积中被取出并细化的高斯生成，
                    #* 不包含“历史帧已细化、但位于当前帧局部体积之外”的高斯。
                    result_dict, my_occ, predtoreturn, gaussianstensor_to_return, instance_feature_toreturn, gaussian_to_vis = my_model(scenemeta=scenemeta, imgs=img, metas=meta, points=None, label=label, grad_frames=cfg.grad_frames, test_mode=False)

                # 将当前帧局部预测（已转到世界坐标）detach 后写回全局 memory，
                # 下一帧会读取更新后的 Gaussian，但梯度不会跨越相邻帧。
                model.scene_update(scenemeta, gaussianstensor_to_return, instance_feature_toreturn, meta[0]['mask_in_global_from_this'])
                loss, loss_dict = loss_func(result_dict)
                loss_record.update(loss=loss.item(), loss_dict=loss_dict)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step_update(global_iter)
                global_iter += 1

                if (i == K_Frames - 1):
                    # myfix
                    model.globalhead.empty_scalar = model.head.empty_scalar
                    model.globalhead.empty_scale = model.head.empty_scale
                    model.globalhead.empty_rot = model.head.empty_rot
                    model.globalhead.empty_sem = model.head.empty_sem
                    model.globalhead.empty_opa = model.head.empty_opa
                    # endfix

                    # 所有视角处理结束后，将累计观测过的世界 Gaussian 聚合到
                    # 完整场景体素网格，得到场景级 global Occupancy。
                    #* 这里才是场景级全局 Occupancy：会使用 memory 中所有 splat_flag == 1 的历史高斯，
                    #* 包括已经不在最后一帧局部区域内、但曾被历史帧处理过的高斯。
                    scene_result_dict = model.get_global_occ(scenemeta, meta[0]['vox_origin'], meta[0]['scene_size'])

                    global_valid_mask = scene_result_dict['mask']
                    global_label = scene_result_dict['label'][global_valid_mask].unsqueeze(0)
                    global_predict = scene_result_dict['predict'][global_valid_mask].unsqueeze(0)

                    global_predict[global_predict == 0] = 255
                    global_predict[global_predict == 12] = 0
                    global_label[global_label == 0] = 255
                    global_label[global_label == 12] = 0
                    global_predict = global_predict.cpu()
                    global_label = global_label.cpu()
                    CalMeanIou_Global.add_batch(global_predict, global_label)

                    model.scene_init(scenemeta)

                gc.collect()
                torch.cuda.empty_cache()

            valid_grad = True
            time_e = time.time()
            if not valid_grad and is_main_process():
                logger.info('[Nan Grad] Epoch %d Iter %5d' % (epoch+1, i_iter))
                params, grads = [], []
                for name, param in my_model.named_parameters():
                    if param.requires_grad:
                        params.append(param.abs().mean().item())
                        grads.append(param.grad.abs().mean().item())
                logger.info('%.5f     %.5f     %.5f' % (loss.item(), torch.mean(torch.tensor(params)).item(), torch.mean(torch.tensor(grads)).item()))

            if i_iter % print_freq == 0 and is_main_process():

                lr = optimizer.param_groups[0]['lr']
                loss_info = loss_record.loss_info()
                logger.info('[TRAIN] ' + scenemeta['scene_name'])
                logger.info('[TRAIN] Epoch %d Iter %5d/%d   ' % (epoch+1, i_iter, len(train_dataset_loader)) + loss_info +
                            'GradNorm: %.3f,   lr: %.7f,   time: %.3f (%.3f)' % (grad_norm, lr, time_e - time_s, data_time_e - data_time_s))

                loss_record.reset()
            data_time_s = time.time()
            time_s = time.time()

            gc.collect()
            torch.cuda.empty_cache()

        global_status = CalMeanIou_Global.get_stats()
        global_sem_cls = global_status["iou_ssc"]
        global_sem = global_status["iou_ssc_mean"]
        global_geo = global_status["iou"]
        logger.info(f'Current global iou of sem is {global_sem_cls}')
        logger.info(f'Current global iou of sem is {global_sem}')
        logger.info(f'Current global iou of geo is {global_geo}')

        if is_main_process():
            my_writer.add_scalar('train/global_sem', global_sem, epoch)
            my_writer.add_scalar('train/global_geo', global_geo, epoch)

        # save checkpoint
        if is_main_process():
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_val_iou': best_val_iou,
                'best_val_miou': best_val_miou
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1

        # eval
        if epoch % eval_freq == 0:
            my_model.eval()
            CalMeanIou.reset()
            CalMeanIou_Fov.reset()
            CalMeanIou_Global.reset()
            loss_record = LossRecord(loss_func=loss_func)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            # 【不是测试时在线学习】验证阶段整体位于 no_grad 中，且下方没有
            # backward/optimizer.step；逐帧改变的只有 Gaussian memory 和可见区域 mask，
            # 网络参数保持固定。因此这里属于 online mapping，而非 test-time training。
            with torch.no_grad():
                for i_iter_val, data in enumerate(val_dataset_loader):
                    for i in range(len(data)):
                        if isinstance(data[i], torch.Tensor):
                            data[i] = data[i].cuda()
                    (imgs, metas, labels) = data # imgs [1, 1, 30, 3, 480, 640]  labels [1, 30, 60, 60, 36]
                    scenemeta = metas[0]
                    for k, v in scenemeta.items():
                        if k in scenemeta_keys:
                            scenemeta[k] = torch.tensor(v).cuda()
                    K_Frames = len(scenemeta['monometa_list'])
                    monometa_list_cuda = []
                    for i in range(K_Frames):
                        monometa = scenemeta['monometa_list'][i]
                        for k, v in monometa.items():
                            if not (k in metas_tensor_keys_inv):
                                monometa[k] = torch.tensor(v).cuda()
                        monometa['img_depthbranch'] = monometa['img_depthbranch'].cuda()
                        monometa_list_cuda.append(monometa)

                    # 每个验证场景都创建独立 memory，防止上一场景状态泄漏。
                    model.scene_init(scenemeta)

                    for i in range(K_Frames):
                        img = imgs[:, :, i, :, :, :].unsqueeze(2)
                        label = labels[:, i, :, :, :].unsqueeze(1)
                        meta = [monometa_list_cuda[i]]
                        with torch.cuda.amp.autocast(enabled=amp):
                            result_dict, my_occ, predtoreturn, gaussianstensor_to_return, instance_feature_toreturn, gaussian_to_vis = my_model(scenemeta=scenemeta, imgs=img, metas=meta, points=None, label=label, grad_frames=None, test_mode=True)
                        # 仅在线更新场景状态；由于外层 no_grad，绝不会更新模型权重。
                        model.scene_update(scenemeta, gaussianstensor_to_return, instance_feature_toreturn, meta[0]['mask_in_global_from_this'])

                        loss, loss_dict = loss_func(result_dict)
                        loss_record.update(loss=loss.item(), loss_dict=loss_dict)

                        voxel_predict = result_dict['ce_input'].argmax(dim=1).long() # [1, 60, 60, 36]
                        voxel_label = result_dict['ce_label'].long() # [1, 60, 60, 36]

                        voxel_predict[voxel_predict == 0] = 255
                        voxel_predict[voxel_predict == 12] = 0
                        voxel_label[voxel_label == 0] = 255
                        voxel_label[voxel_label == 12] = 0
                        voxel_predict = voxel_predict.cpu()
                        voxel_label = voxel_label.cpu()
                        #* single 指标：评估每一帧完整的 60×60×36 局部体素网格，没有使用 fov_mask。
                        #* 各验证帧的 TP/FP/FN 会持续累积，最后统一计算 IoU，并非逐帧 IoU 的平均值。
                        CalMeanIou.add_batch(voxel_predict, voxel_label)

                        voxel_predict = result_dict['ce_input'].argmax(dim=1).long() # [1, 60, 60, 36]
                        voxel_label = result_dict['ce_label'].long() # [1, 60, 60, 36]
                        this_fov_mask = meta[0]['fov_mask'].unsqueeze(0)
                        voxel_predict = voxel_predict[this_fov_mask].unsqueeze(0)
                        voxel_label = voxel_label[this_fov_mask].unsqueeze(0)

                        voxel_predict[voxel_predict == 0] = 255
                        voxel_predict[voxel_predict == 12] = 0
                        voxel_label[voxel_label == 0] = 255
                        voxel_label[voxel_label == 12] = 0
                        voxel_predict = voxel_predict.cpu()
                        voxel_label = voxel_label.cpu()

                        #* fov 指标：仍是当前帧 Local Occ Head 的预测，但只评估 fov_mask=True 的
                        #* 相机视野内体素；当前局部体积中位于相机视野外的体素不参与该指标。
                        CalMeanIou_Fov.add_batch(voxel_predict, voxel_label)

                        if (i == K_Frames - 1):
                            # myfix
                            model.globalhead.empty_scalar = model.head.empty_scalar
                            model.globalhead.empty_scale = model.head.empty_scale
                            model.globalhead.empty_rot = model.head.empty_rot
                            model.globalhead.empty_sem = model.head.empty_sem
                            model.globalhead.empty_opa = model.head.empty_opa
                            # endfix

                            scene_result_dict = model.get_global_occ(scenemeta, meta[0]['vox_origin'], meta[0]['scene_size'])

                            global_valid_mask = scene_result_dict['mask']
                            global_label = scene_result_dict['label'][global_valid_mask].unsqueeze(0)
                            global_predict = scene_result_dict['predict'][global_valid_mask].unsqueeze(0)

                            global_predict[global_predict == 0] = 255
                            global_predict[global_predict == 12] = 0
                            global_label[global_label == 0] = 255
                            global_label[global_label == 12] = 0
                            global_predict = global_predict.cpu()
                            global_label = global_label.cpu()

                            #* global 指标：每个场景只在序列最后一帧统计一次，评估历史帧在线融合后的
                            #* 场景级 Occupancy；global_valid_mask 是所有历史帧累计观测区域的并集。
                            CalMeanIou_Global.add_batch(global_predict, global_label)

                            model.scene_init(scenemeta)

                    if i_iter_val % print_freq == 0 and is_main_process():
                        loss_info = loss_record.loss_info()
                        logger.info('[EVAL] ' + scenemeta['scene_name'])
                        logger.info('[EVAL] Iter %5d/%d   '%(i_iter_val, len(val_dataset_loader)) + loss_info)

                    gc.collect()
                    torch.cuda.empty_cache()

            global_status = CalMeanIou_Global.get_stats()
            #* global_sem_cls：全局累计观测区域内，每个语义类别各自的 IoU 数组。
            global_sem_cls = global_status["iou_ssc"]
            #* global_sem：排除 empty 类（类别 0）后，其余语义类别 IoU 的平均值，即语义 mIoU。
            global_sem = global_status["iou_ssc_mean"]
            #* global_geo：忽略具体语义类别，只区分 occupied/free 的场景级几何 IoU。
            global_geo = global_status["iou"]
            logger.info(f'Current global iou of sem is {global_sem_cls}')
            logger.info(f'Current global iou of sem is {global_sem}')
            logger.info(f'Current global iou of geo is {global_geo}')

            if is_main_process():
                my_writer.add_scalar('val/global_sem', global_sem, epoch)
                my_writer.add_scalar('val/global_geo', global_geo, epoch)

            #* single：汇总所有验证帧“完整局部体素网格”的指标，不限制在相机 FOV 内。
            stats = CalMeanIou.get_stats()
            #* info_sem_cls：逐类别语义 IoU；info_sem：排除 empty 后的语义 mIoU；
            #* info_geo：将所有非空语义类合并为 occupied 后计算的 occupied/free 几何 IoU。
            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]

            logger.info(f'Current single val iou of sem_cls is {info_sem_cls}')
            logger.info(f'Current single val iou of sem is {info_sem}')
            logger.info(f'Current single val iou of geo is {info_geo}')

            #* fov：汇总所有验证帧中 fov_mask=True 的相机视野内体素指标。
            stats_fov = CalMeanIou_Fov.get_stats()
            #* info_sem_cls_fov：FOV 内逐类别语义 IoU；info_sem_fov：FOV 内语义 mIoU；
            #* info_geo_fov：FOV 内 occupied/free 几何 IoU。
            info_sem_cls_fov = stats_fov["iou_ssc"]
            info_sem_fov = stats_fov["iou_ssc_mean"]
            info_geo_fov = stats_fov["iou"]

            logger.info(f'Current fov val iou of sem_cls is {info_sem_cls_fov}')
            logger.info(f'Current fov val iou of sem is {info_sem_fov}')
            logger.info(f'Current fov val iou of geo is {info_geo_fov}')

            if is_main_process():
                my_writer.add_scalar('val/sem_fov', info_sem_fov, epoch)
                my_writer.add_scalar('val/geo_fov', info_geo_fov, epoch)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/train_embodied_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/train_embodied')
    parser.add_argument('--resume-from', type=str, default='')

    args, _ = parser.parse_known_args()
    main(args)
