import argparse
import copy
import os
import warnings

print(os.getcwd())
import sys
sys.path.append(os.getcwd())
import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval
from easydict import EasyDict
from torchpack.utils.config import Config as Config_torch



def main():
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    configs = Config_torch()
    bev_config_file = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    configs.load(bev_config_file, recursive=True)
    config_bev = Config(recursive_eval(configs), filename=bev_config_file)

    ################################### Change configurations here ###################################
    ################# configs for saving train detections
    # args = {
    #     'eval':'bbox',
    #     'checkpoint':'pretrained/bevfusion-det.pth',
    #     'seed':0,
    #     'deterministic':True,
    #     'fuse_conv_bn':False,
    #     'format_only':False,
    #     'out':False,
    #     'gpu_collect':True,
    #     'tmpdir':'results',
    # }
    # data_split = "train"
    # config_bev.data.test.ann_file = config_bev.data.test.dataset_root + "nuscenes_infos_"+data_split+".pkl"
    ################## End of Configs for saving train detections. 
    ################## configs for saving val detections
    # args = {
    #     'eval':'bbox',
    #     'checkpoint':'pretrained/bevfusion-det.pth',
    #     'seed':0,
    #     'deterministic':True,
    #     'fuse_conv_bn':False,
    #     'format_only':False,
    #     'out':False,
    #     'gpu_collect':True,
    #     'tmpdir':'results',
    # }
    # data_split = "val"
    # config_bev.data.test.ann_file = config_bev.data.test.dataset_root + "nuscenes_infos_"+data_split+".pkl"
    ################## End of Configs for saving val detections. 
    ################## Configs for saving test detections. need to replace mmdet3d.datasets with mmdet3d.datasets_test
    args = {
        'checkpoint':'pretrained/bevfusion-det.pth',
        'seed':0,
        'deterministic':True,
        'fuse_conv_bn':False,
        'format_only':True,
        'eval_options':{"jsonfile_prefix":"results"},
        'out':False,
        'gpu_collect':True,
        'tmpdir':'results',
    }
    config_bev.data.test.type = 'NuScenesDatasetTest'
    config_bev.data.test.pipeline = config_bev.data.test.pipeline[0:3]+config_bev.data.test.pipeline[4:]
    config_bev.data.test.pipeline[8].with_gt=False
    config_bev.data.test.pipeline[8].with_label=False
    data_split = "test"
    config_bev.data.test.ann_file = config_bev.data.test.dataset_root + "nuscenes_infos_"+data_split+".pkl"
    ################## End of Configs for saving test detections. 
    args = EasyDict(args)
    config_bev.update(args)
    cfg = config_bev
    cfg.data.test.save_detection_results = True
    # which data split to use


    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    samples_per_gpu = 1
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))
