import os, time, argparse, os.path as osp, numpy as np
import torch
import gc
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torchrun --nproc_per_node=8 train_mono.py
from utils.iou_eval import IOUEvalBatch
from utils.iou_as_iso import SSCMetrics
from utils.loss_record import LossRecord
from utils.load_save_util import revise_ckpt, revise_ckpt_2

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")
import sys
# sys.path.append('/data1/code/wyq/gaussianindoor/EmbodiedOcc')
# sys.path.append('/data1/code/wyq/gaussianindoor/EmbodiedOcc/Depth-Anything-V2/metric_depth')
# sys.path.append("/vepfs-mlp2/c20250502/haoce/wangyushen/EmbodiedOcc")
# sys.path.append("/vepfs-mlp2/c20250502/haoce/wangyushen/EmbodiedOcc/Depth-Anything-V2/metric_depth")
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


class DistributedEvalSampler(Sampler):
    """将验证集无重复地划分到各进程，不像 DistributedSampler 那样补齐样本。"""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.world_size + 1


def reduce_eval_metrics(metric, device):
    """先汇总所有进程的 TP/FP/FN，再由全局计数计算指标。"""
    completion = torch.tensor(
        [metric.completion_tp, metric.completion_fp, metric.completion_fn],
        dtype=torch.float64,
        device=device,
    )
    semantic = torch.as_tensor(
        np.stack([metric.tps, metric.fps, metric.fns]),
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(completion, op=dist.ReduceOp.SUM)
    dist.all_reduce(semantic, op=dist.ReduceOp.SUM)
    metric.completion_tp, metric.completion_fp, metric.completion_fn = completion.cpu().tolist()
    metric.tps, metric.fps, metric.fns = semantic.cpu().numpy()

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
    save_freq = cfg.get("save_freq", 1)  # 每隔几个epoch保存一次model

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
    logger = MMLogger(
        name='indoor_nyu',
        log_file=log_file if is_main_process() else None,
        log_level='INFO' if is_main_process() else 'ERROR',
    )
    if is_main_process():
        logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from model import build_model
    my_model = build_model(cfg.model)

    if cfg.flag_depthanything_as_gt: # True
        my_model.depthanything.requires_grad_(False)

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

    if distributed:
        # 仅支持训练用途的原实现：DistributedSampler(drop_last=False) 会在验证集
        # 长度不能整除进程数时补入重复样本，导致最终指标有偏差。
        # val_sampler = DistributedSampler(val_wrapper, shuffle=False, drop_last=False)
        # 支持严格多卡评估：每个验证样本只分配给一个进程，不做补齐。
        val_dataset_loader = DataLoader(
            dataset=val_dataset_loader.dataset,
            batch_size=cfg.val_loader_config["batch_size"],
            collate_fn=val_dataset_loader.collate_fn,
            shuffle=False,
            sampler=DistributedEvalSampler(val_dataset_loader.dataset, rank, world_size),
            num_workers=cfg.val_loader_config["num_workers"],
            pin_memory=True,
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
        warmup_t=1000, # FIXME
        warmup_lr_init=1e-6,
        t_in_epochs=False
    )

    CalMeanIou = SSCMetrics(n_classes=12)
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
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))

    metas_tensor_keys_inv = ['depth_gt_np_valid', 'depth_gt_np', 'name', 'cam2img', 'world2img', 'rgb_path', 'depth_path','num_depth', 'occ_mask_valid', 'occ_mask_valid_fov', 'img_shape', 'img_aug_matrix']

    # training
    while epoch < max_num_epochs:
        my_model.train()
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
            (imgs, metas, label) = data

            # 仅支持 bs=1 的原实现：只搬运 metas[0] 到 GPU。
            # for k, v in metas[0].items():
            #     if not (k in metas_tensor_keys_inv):
            #         metas[0][k] = torch.tensor(v).cuda()
            # metas[0]['img_depthbranch'] = metas[0]['img_depthbranch'].cuda()
            # 支持 bs>1：遍历 batch 内所有 metadata，并搬到当前图像所在设备。
            device = imgs.device
            for meta in metas:
                for k, v in meta.items():
                    if k not in metas_tensor_keys_inv:
                        if isinstance(v, torch.Tensor):
                            meta[k] = v.to(device)
                        else:
                            meta[k] = torch.as_tensor(v, device=device)
                meta["img_depthbranch"] = meta["img_depthbranch"].to(device)

            # forward + backward + optimize
            data_time_e = time.time()

            with torch.cuda.amp.autocast(enabled=amp):
                result_dict, my_occ, predtoreturn = my_model(imgs=imgs, metas=metas, points=None, label=label, grad_frames=cfg.grad_frames, test_mode=False)

            loss, loss_dict = loss_func(result_dict)
            loss_record.update(loss=loss.item(), loss_dict=loss_dict)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)

            valid_grad = True

            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(global_iter)
            time_e = time.time()
            if not valid_grad and is_main_process():
                logger.info('[Nan Grad] Epoch %d Iter %5d' % (epoch+1, i_iter))
                params, grads = [], []
                for name, param in my_model.named_parameters():
                    if param.requires_grad:
                        params.append(param.abs().mean().item())
                        grads.append(param.grad.abs().mean().item())
                logger.info('%.5f     %.5f     %.5f' % (loss.item(), torch.mean(torch.tensor(params)).item(), torch.mean(torch.tensor(grads)).item()))

            global_iter += 1
            if i_iter % print_freq == 0 and is_main_process():
                lr = optimizer.param_groups[0]['lr']
                loss_info = loss_record.loss_info()
                
                # logger.info('[TRAIN] Epoch %d Iter %5d/%d   ' % (epoch+1, i_iter, len(train_dataset_loader)) + loss_info +
                #             'GradNorm: %.3f,   lr: %.7f,   time: %.3f (%.3f)' % (grad_norm, lr, time_e - time_s, data_time_e - data_time_s))
                logger.info(
                    "[TRAIN] Epoch %d Iter %5d/%d   "
                    % (epoch + 1, i_iter, len(train_dataset_loader))
                    + loss_info
                    + "GradNorm: %.3f,   lr: %.7f,   memory: %.2f GB,   time: %.3f (%.3f)"
                    % (
                        grad_norm,
                        lr,
                        torch.cuda.memory_allocated() / 1024**3,
                        time_e - time_s,
                        data_time_e - data_time_s,
                    )
                )
                loss_record.reset()
            data_time_s = time.time()
            time_s = time.time()

            gc.collect()
            torch.cuda.empty_cache()

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

            dst_file = osp.join(args.work_dir, "latest.pth")
            torch.save(dict_to_save, dst_file)

            if (epoch + 1) % save_freq == 0:
                save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
                torch.save(dict_to_save, save_file_name)

        epoch += 1

        # eval
        if epoch % eval_freq == 0:
            my_model.eval()
            # 无补齐验证 sampler 可能使各进程迭代次数不同。绕过 DDP 包装层进行
            # 纯前向，避免 DDP forward 中的 buffer 同步让先结束的进程发生死锁。
            eval_model = my_model.module if distributed else my_model
            CalMeanIou.reset()
            loss_record = LossRecord(loss_func=loss_func)
            val_loss_sum = 0.0
            val_sample_count = 0
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            with torch.no_grad():
                for i_iter_val, data in enumerate(val_dataset_loader):
                    for i in range(len(data)):
                        if isinstance(data[i], torch.Tensor):
                            data[i] = data[i].cuda()
                    (imgs, metas, label) = data

                    # 仅支持 bs=1 的原实现：验证时也只搬运 metas[0] 到 GPU。
                    # for k, v in metas[0].items():
                    #     if not (k in metas_tensor_keys_inv):
                    #         metas[0][k] = torch.tensor(v).cuda()
                    # metas[0]['img_depthbranch'] = metas[0]['img_depthbranch'].cuda()
                    # 支持 bs>1：验证阶段同样处理 batch 内全部 metadata。
                    device = imgs.device

                    for meta in metas:
                        for k, v in meta.items():
                            if k not in metas_tensor_keys_inv:
                                if isinstance(v, torch.Tensor):
                                    meta[k] = v.to(device)
                                else:
                                    meta[k] = torch.as_tensor(v, device=device)

                        meta["img_depthbranch"] = meta["img_depthbranch"].to(device)

                    with torch.cuda.amp.autocast(enabled=amp):
                        result_dict, my_occ, predtoreturn = eval_model(imgs=imgs, metas=metas, points=None, label=label, grad_frames=None, test_mode=True)

                    loss, loss_dict = loss_func(result_dict)
                    loss_record.update(loss=loss.item(), loss_dict=loss_dict)
                    batch_size = imgs.shape[0]
                    val_loss_sum += loss.item() * batch_size
                    val_sample_count += batch_size

                    voxel_predict = result_dict['ce_input'].argmax(dim=1).long() # [1, 60, 60, 36]
                    voxel_label = result_dict['ce_label'].long() # [1, 60, 60, 36]

                    voxel_predict[voxel_predict == 0] = 255
                    voxel_predict[voxel_predict == 12] = 0
                    voxel_label[voxel_label == 0] = 255
                    voxel_label[voxel_label == 12] = 0
                    voxel_predict = voxel_predict.cpu()
                    voxel_label = voxel_label.cpu()

                    CalMeanIou.add_batch(voxel_predict, voxel_label)

                    if i_iter_val % print_freq == 0 and is_main_process():
                        loss_info = loss_record.loss_info()
                        logger.info('[EVAL] Iter %5d/%d   '%(i_iter_val, len(val_dataset_loader)) + loss_info)

                    gc.collect()
                    torch.cuda.empty_cache()

            if distributed:
                reduce_eval_metrics(CalMeanIou, torch.device("cuda", gpu))

                # 仅支持单进程的原实现：各进程分别计算自己的验证 loss。
                # global_val_loss = np.mean(loss_record.total_loss)
                # 支持多卡：按样本数归约 loss，避免不同进程 batch 数不同时产生偏差。
                local_sample_count = torch.tensor(
                    val_sample_count,
                    dtype=torch.long,
                    device=torch.device("cuda", gpu),
                )
                sample_counts_per_rank = [
                    torch.zeros_like(local_sample_count) for _ in range(world_size)
                ]
                dist.all_gather(sample_counts_per_rank, local_sample_count)
                sample_counts_per_rank = [
                    count.item() for count in sample_counts_per_rank
                ]
                loss_and_count = torch.tensor(
                    [val_loss_sum, val_sample_count],
                    dtype=torch.float64,
                    device=torch.device("cuda", gpu),
                )
                dist.all_reduce(loss_and_count, op=dist.ReduceOp.SUM)
                global_val_loss = (
                    loss_and_count[0] / loss_and_count[1].clamp_min(1)
                ).item()
                global_sample_count = int(loss_and_count[1].item())
            else:
                global_val_loss = val_loss_sum / max(val_sample_count, 1)
                sample_counts_per_rank = [val_sample_count]
                global_sample_count = val_sample_count

            stats = CalMeanIou.get_stats()

            info_sem_cls = stats["iou_ssc"]
            info_sem = stats["iou_ssc_mean"]
            info_geo = stats["iou"]

            if is_main_process():
                expected_sample_count = len(val_dataset_loader.dataset)
                logger.info(
                    "[EVAL] Samples used to calculate mIoU/IoU: "
                    f"per-rank={sample_counts_per_rank}, "
                    f"collected={global_sample_count}, "
                    f"expected={expected_sample_count}"
                )
                if global_sample_count != expected_sample_count:
                    logger.warning(
                        "[EVAL] Collected sample count does not match validation "
                        f"dataset size: {global_sample_count} != "
                        f"{expected_sample_count}"
                    )
                logger.info(f'Current val loss is {global_val_loss}')
                logger.info(f'Current val iou of sem_cls is {info_sem_cls}')
                logger.info(f'Current val iou of sem is {info_sem}')
                logger.info(f'Current val iou of geo is {info_geo}')


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/train_mono_config.py')
    parser.add_argument('--work-dir', type=str, default='/home/wyq/WorkSpace/workdir/train_mono')
    parser.add_argument('--resume-from', type=str, default='')

    args, _ = parser.parse_known_args()
    main(args)
