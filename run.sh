pip install . --no-build-isolation
python setup.py build_ext --inplace 
export PYTHONPATH=`pwd`/src:`pwd`:`pwd`/Depth-Anything-V2/metric_depth:`pwd`/EfficientNet-PyTorch:`pwd`/model/depthbranch:$PYTHONPATH
export TORCH_HOME=/c20250502/wangyushen/.cache/torch
python /vepfs-mlp2/c20250502/haoce/wangyushen/EmbodiedOcc/train_mono.py \
    --py-config /vepfs-mlp2/c20250502/haoce/wangyushen/EmbodiedOcc/config/customs/train_mono_config_custom.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/embodiedocc/outputs/train_mono_debug