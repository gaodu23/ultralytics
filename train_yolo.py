#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO模型训练脚本
此脚本用于训练YOLO模型，支持多种配置和自动化训练流程
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import yaml
from datetime import datetime
import torch
import torch.cuda

# 添加项目根目录到路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """
    检查训练环境
    """
    logger.info("=" * 60)
    logger.info("🔍 环境检查")
    logger.info("=" * 60)
    
    # Python版本
    python_version = sys.version.split()[0]
    logger.info(f"🐍 Python版本: {python_version}")
    
    # PyTorch版本
    logger.info(f"🔥 PyTorch版本: {torch.__version__}")
    
    # CUDA检查
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
        
        logger.info(f"✅ CUDA可用: {cuda_version}")
        logger.info(f"🎮 GPU数量: {gpu_count}")
        logger.info(f"🎯 当前GPU: {gpu_name}")
        logger.info(f"💾 GPU内存: {gpu_memory:.1f}GB")
    else:
        logger.warning("⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
    
    logger.info("=" * 60)

def create_data_yaml(data_dir, yaml_path, class_names=None):
    """
    创建数据配置YAML文件
    
    Args:
        data_dir (str): 数据集目录路径
        yaml_path (str): YAML文件保存路径
        class_names (list): 类别名称列表
    """
    data_dir = Path(data_dir)
    
    # 默认类别名称
    if class_names is None:
        class_names = ['person', 'vehicle', 'animal']  # 根据你的需求修改
    
    # 创建YAML配置
    yaml_config = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    # 保存YAML文件
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"📝 已创建数据配置文件: {yaml_path}")
    return yaml_path

def validate_dataset(data_yaml):
    """
    验证数据集结构
    
    Args:
        data_yaml (str): 数据集配置文件路径
    """
    logger.info("🔍 验证数据集结构...")
    
    try:
        with open(data_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        data_path = Path(config['path'])
        train_images = data_path / config['train']
        val_images = data_path / config['val']
        train_labels = train_images.parent / 'labels'
        val_labels = val_images.parent / 'labels'
        
        # 检查目录结构
        required_dirs = [train_images, val_images, train_labels, val_labels]
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.warning(f"⚠️  目录不存在: {dir_path}")
            else:
                file_count = len(list(dir_path.glob("*.*")))
                logger.info(f"✅ {dir_path.name}: {file_count} 个文件")
        
        logger.info(f"📊 数据集类别数: {config['nc']}")
        logger.info(f"🏷️  类别名称: {config['names']}")
        
    except Exception as e:
        logger.error(f"❌ 数据集验证失败: {str(e)}")
        return False
    
    return True

def train_model(
    model_name="yolo11n.pt",
    data_yaml="data.yaml",
    epochs=100,
    batch_size=16,
    image_size=640,
    device="",
    project="runs/train",
    name="exp",
    resume=False,
    save_period=10,
    patience=10,
    workers=8,
    lr0=0.01,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4
):
    """
    训练YOLO模型
    
    Args:
        model_name (str): 预训练模型名称或路径
        data_yaml (str): 数据集配置文件路径
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        image_size (int): 图像大小
        device (str): 设备选择
        project (str): 项目保存目录
        name (str): 实验名称
        resume (bool): 是否恢复训练
        save_period (int): 保存间隔
        patience (int): 早停耐心值
        workers (int): 数据加载器工作进程数
        lr0 (float): 初始学习率
        warmup_epochs (int): 预热轮数
        其他参数用于数据增强配置
    """
    try:
        # 环境检查
        check_environment()
        
        # 验证数据集
        if not os.path.exists(data_yaml):
            logger.error(f"❌ 数据集配置文件不存在: {data_yaml}")
            return None
        
        if not validate_dataset(data_yaml):
            logger.error("❌ 数据集验证失败")
            return None
        
        # 加载模型
        logger.info(f"🚀 开始训练模型: {model_name}")
        logger.info(f"📊 数据集配置: {data_yaml}")
        
        model = YOLO(model_name)
        
        # 设备选择
        if not device:
            device = select_device()
        logger.info(f"🎯 使用设备: {device}")
        
        # 训练配置
        train_config = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': image_size,
            'device': device,
            'project': project,
            'name': name,
            'resume': resume,
            'save_period': save_period,
            'patience': patience,
            'workers': workers,
            'lr0': lr0,
            'warmup_epochs': warmup_epochs,
            'mosaic': mosaic,
            'mixup': mixup,
            'copy_paste': copy_paste,
            'degrees': degrees,
            'translate': translate,
            'scale': scale,
            'shear': shear,
            'perspective': perspective,
            'flipud': flipud,
            'fliplr': fliplr,
            'hsv_h': hsv_h,
            'hsv_s': hsv_s,
            'hsv_v': hsv_v,
            'verbose': True,
            'save': True,
            'cache': False,  # 根据内存情况调整
            'rect': False,   # 矩形训练
            'cos_lr': False, # 余弦学习率调度
            'close_mosaic': 10,  # 最后几轮关闭马赛克
            'amp': True,     # 自动混合精度
        }
        
        logger.info("=" * 60)
        logger.info("📋 训练配置:")
        for key, value in train_config.items():
            if key not in ['data']:  # 跳过长路径
                logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
        
        # 开始训练
        start_time = datetime.now()
        logger.info(f"🎬 训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = model.train(**train_config)
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("🎉 训练完成!")
        logger.info(f"⏱️  训练时长: {training_duration}")
        logger.info(f"📁 模型保存在: {model.trainer.save_dir}")
        logger.info("=" * 60)
        
        # 显示训练结果
        if hasattr(results, 'results_dict'):
            logger.info("📊 最终训练指标:")
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
        
        return str(model.trainer.save_dir)
        
    except Exception as e:
        logger.error(f"❌ 训练过程中出错: {str(e)}")
        return None

def validate_model(model_path, data_yaml, device=""):
    """
    验证训练后的模型
    
    Args:
        model_path (str): 模型路径
        data_yaml (str): 数据集配置文件
        device (str): 设备选择
    """
    try:
        logger.info("🔍 开始模型验证...")
        
        model = YOLO(model_path)
        
        # 运行验证
        metrics = model.val(
            data=data_yaml,
            device=device,
            verbose=True
        )
        
        logger.info("=" * 60)
        logger.info("📊 验证结果:")
        logger.info(f"  mAP50: {metrics.box.map50:.4f}")
        logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"  Precision: {metrics.box.mp:.4f}")
        logger.info(f"  Recall: {metrics.box.mr:.4f}")
        logger.info("=" * 60)
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ 模型验证失败: {str(e)}")
        return None

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="YOLO模型训练脚本")
    
    # 基本参数
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="预训练模型路径")
    parser.add_argument("--data", type=str, default="data.yaml", help="数据集配置文件")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图像大小")
    parser.add_argument("--device", type=str, default="", help="设备选择 (cpu, 0, 1, ...)")
    
    # 训练设置
    parser.add_argument("--project", type=str, default="runs/train", help="项目保存目录")
    parser.add_argument("--name", type=str, default="exp", help="实验名称")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--save-period", type=int, default=10, help="保存间隔")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    
    # 优化器参数
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="预热轮数")
    
    # 数据增强参数
    parser.add_argument("--mosaic", type=float, default=1.0, help="马赛克增强概率")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp增强概率")
    parser.add_argument("--degrees", type=float, default=0.0, help="旋转角度")
    parser.add_argument("--translate", type=float, default=0.1, help="平移比例")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放比例")
    parser.add_argument("--fliplr", type=float, default=0.5, help="水平翻转概率")
    
    # 其他参数
    parser.add_argument("--validate", action="store_true", help="训练后验证模型")
    parser.add_argument("--create-data", type=str, help="创建数据配置文件的数据集目录")
    parser.add_argument("--class-names", nargs='+', help="类别名称列表")
    
    args = parser.parse_args()
    
    # 创建数据配置文件
    if args.create_data:
        create_data_yaml(args.create_data, args.data, args.class_names)
        logger.info(f"✅ 数据配置文件已创建: {args.data}")
        return
    
    # 开始训练
    save_dir = train_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        save_period=args.save_period,
        patience=args.patience,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr
    )
    
    if save_dir and args.validate:
        best_model = os.path.join(save_dir, "weights", "best.pt")
        if os.path.exists(best_model):
            validate_model(best_model, args.data, args.device)

if __name__ == "__main__":
    # 直接运行的默认配置
    # 你可以在这里修改默认的训练参数
    
    # 创建示例数据配置
    data_yaml = "custom_data.yaml"
    
    # 如果数据配置文件不存在，创建一个示例
    if not os.path.exists(data_yaml):
        logger.info("📝 创建示例数据配置文件...")
        sample_config = {
            'path': './datasets/custom',
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 80,  # 类别数量
            'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                     'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        }
        
        with open(data_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✅ 已创建示例数据配置文件: {data_yaml}")
        logger.info("⚠️  请根据你的数据集修改配置文件")
    
    # 默认训练参数
    train_model(
        model_name="yolo11n.pt",     # 使用YOLOv11 nano模型
        data_yaml=data_yaml,          # 数据配置文件
        epochs=100,                   # 训练轮数
        batch_size=16,                # 批次大小
        image_size=640,               # 图像大小
        device="",                    # 自动选择设备
        project="runs/train",         # 保存目录
        name="custom_model",          # 实验名称
        lr0=0.01,                     # 初始学习率
        patience=20                   # 早停耐心值
    )
