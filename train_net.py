# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.

./train_net.py \
  --config-file configs/ade20k-150/maskformer_R50_bs16_160k.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
"""
import copy
import itertools
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask_former import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_mask_former_config,
)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.  根据DETR调整的Trainer类的扩展。
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None): # 构建评估器
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:  # 当输出文件夹为空的时候
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")  # 创建输出文件夹，路径为cfg.OUTPUT_DIR/inference
        evaluator_list = []  # 创建评估器列表
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type  # 通过访问数据集的元数据，获取评估器的类型
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:  # 当验证类型为sem_seg或者ade20k_panoptic_seg的时候，会使用SemSegEvaluator，并传入数据集名称，输出文件夹，以及是否分布式
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)  # 会根据传入的评估器列表，创建DatasetEvaluators，再返回

    @classmethod
    def build_train_loader(cls, cfg): # 设置训练集加载器，和上面的评估器类似，会根据数据集的元数据，创建不同的数据集加载器
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # DETR-style dataset mapper for COCO panoptic segmentation
        elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
            mapper = DETRPanopticDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):  # 构建学习率调度器
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):  # 构建优化器
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM  # 获取在norm层的权重衰减参数
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED  # 获取在emmbedding层的权重衰减参数

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR  # 将基础学习率传入defaults
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY  # 将权重衰减参数传入defaults

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )  # norm层的类型传入进去

        params: List[Dict[str, Any]] = []  # 创建参数列表
        memo: Set[torch.nn.parameter.Parameter] = set() # 创建参数集合
        for module_name, module in model.named_modules():  # 遍历模型中的所有模块
            for module_param_name, value in module.named_parameters(recurse=False):  # 遍历模块中的所有参数
                if not value.requires_grad:  # 当参数不需要梯度的时候，会继续进行
                    continue
                # Avoid duplicating parameters
                if value in memo:  # 当参数在参数集合中的时候，会继续进行
                    continue
                memo.add(value)  # 将参数添加到参数集合中

                hyperparams = copy.copy(defaults)  # 将defaults进行拷贝到hyperparams中
                if "backbone" in module_name:  # 当模块名称中包含backbone的时候
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER  # 会将学习率乘以cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):  # 当参数名称中包含relative_position_bias_table或者absolute_pos_embed的时候，会打印出模型的参数名称，并将权重衰减参数设置为0
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):  # 当模块的类型在norm_module_types中的时候
                    hyperparams["weight_decay"] = weight_decay_norm  # 会将权重衰减参数设置为weight_decay_norm
                if isinstance(module, torch.nn.Embedding):  # 当模块的类型为torch.nn.Embedding的时候，会将权重衰减参数设置为weight_decay_embed
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now  # detectron2现在没有完整的模型梯度裁剪，所以这里会返回optim
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE  # 获取梯度裁剪的值
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )  # 当梯度裁剪开启，梯度裁剪类型为full_model，梯度裁剪的值大于0的时候，会将enable设置为True

            class FullModelGradientClippingOptimizer(optim):  #进行梯度裁剪的目的就是了防止梯度爆炸，所以这里会创建一个FullModelGradientClippingOptimizer类，继承自optim 
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])   # 先收集所有的参数
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)  # 再使用clip_grad_norm_进行梯度裁剪，这个函数会将梯度的范数限制在clip_norm_val以内
                    super().step(closure=closure)  # 再调用父类的step函数

            return FullModelGradientClippingOptimizer if enable else optim  # 会根据enable的值，返回FullModelGradientClippingOptimizer或者optim

        optimizer_type = cfg.SOLVER.OPTIMIZER  # 将优化器的类型传入optimizer_type
        if optimizer_type == "SGD":  # 会根据优化器的类型，创建不同的优化器，像SGD，ADAMW等，然后会将参数，像学习率，动量等传入，再返回
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")  # 首先获取日志
        # In the end of training, run an evaluation with TTA.
        # TTA是一种在测试阶段应用数据增强技术以改善模型性能的方法，它通过对输入数据进行多次变换或增强来获得多个不同的预测，然后将这些预测进行平均或投票来得到最终的预测结果。
        # TTA的优点是它可以提高模型的性能，而不需要对模型进行重新训练，因此它可以在训练后应用于任何模型。
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)  # 调用SemanticSegmentorWithTTA类，传入配置文件和模型
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )  # 创建评估器，传入配置文件，数据集名称，输出文件夹
            for name in cfg.DATASETS.TEST  # 遍历测试集
        ]
        res = cls.test(cfg, model, evaluators)  # 调用test函数，传入配置文件，模型，评估器
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})  # 将上面的结果进行整理
        return res


def setup(args):  # 这个函数是用来解析参数的
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()  # 从detectron2中获取配置文件
    # for poly lr schedule
    add_deeplab_config(cfg)  # 添加deeplab的配置文件
    add_mask_former_config(cfg)  # 添加mask_former的配置文件
    cfg.merge_from_file(args.config_file)  # 将配置文件中的参数进行解析
    cfg.merge_from_list(args.opts)  # 将命令行中的参数进行解析
    cfg.freeze()  # 冻结参数
    default_setup(cfg, args)  # 传进来的参数设置成默认的参数
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")  # 设置日志，输出到cfg.OUTPUT_DIR中
    return cfg


def main(args):
    cfg = setup(args)  # 解析参数，获取配置文件

    if args.eval_only: # 当仅进行评估的时候
        model = Trainer.build_model(cfg)  # 构建模型
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )  # 恢复或加载模型，传入模型和权重
        res = Trainer.test(cfg, model)  # 进行评估
        if cfg.TEST.AUG.ENABLED:  # 当cfg.TEST.AUG.ENABLED为True的时候，会调用test_with_TTA函数，传入配置文件和模型，进行评估
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():  # 当是主进程的时候，会打印出评估结果
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)  # 创建Trainer类，传入配置文件
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()  # 进行训练


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )  # 启动训练，并且传入main函数和参数，像num_gpus，num_machines等
