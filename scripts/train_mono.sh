
export PYTHONPATH=`pwd`/src:`pwd`:`pwd`/Depth-Anything-V2/metric_depth:`pwd`/EfficientNet-PyTorch:`pwd`/model/depthbranch:$PYTHONPATH
export TORCH_HOME=/c20250502/wangyushen/.cache/torch

python -m torch.distributed.launch \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --node_rank=$MLP_ROLE_INDEX \
    --master_port=$MLP_WORKER_0_PORT \
    --nnodes=$MLP_WORKER_NUM \
    /vepfs-mlp2/c20250502/haoce/wangyushen/EmbodiedOcc/train_mono.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/EmbodiedOcc/config/customs/train_mono_config_custom_experiment.py \
    --work-dir /c20250502/wangyushen/Outputs/embodiedocc/mono/train
