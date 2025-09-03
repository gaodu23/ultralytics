#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建YOLO训练数据集配置文件
此脚本用于从检测结果生成的dataset目录创建YOLO训练配置文件
"""

import os
import yaml
import argparse
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dataset_yaml(dataset_dir, output_path="dataset.yaml", class_names=None, train_split=0.8):
    """
    创建YOLO训练用的数据集配置文件
    
    Args:
        dataset_dir (str): train_dataset目录路径
        output_path (str): 输出YAML文件路径
        class_names (list): 类别名称列表
        train_split (float): 训练集比例
    """
    try:
        dataset_path = Path(dataset_dir)
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.error(f"train_dataset目录结构不正确: {dataset_path}")
            logger.error(f"需要存在: {images_dir} 和 {labels_dir}")
            return None
        
        # 统计文件数量
        image_files = list(images_dir.glob("*.*"))
        label_files = list(labels_dir.glob("*.txt"))
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        logger.info(f"找到 {len(label_files)} 个标签文件")
        
        # 如果未提供类别名称，尝试从标签文件中提取
        if class_names is None:
            class_ids = set()
            for label_file in label_files:
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                except Exception as e:
                    logger.warning(f"读取标签文件失败 {label_file}: {e}")
            
            # 生成默认类别名称
            class_names = [f"class_{i}" for i in sorted(class_ids)]
            logger.info(f"从标签文件中检测到 {len(class_names)} 个类别: {class_names}")
            logger.warning("请手动编辑YAML文件，将class_0, class_1等替换为实际的类别名称")
        
        # 创建数据集配置
        dataset_config = {
            'path': str(dataset_path.resolve()),
            'train': 'images',
            'val': 'images',  # 可以后续手动分割训练集和验证集
            'test': 'images',
            'nc': len(class_names),
            'names': class_names
        }
        
        # 保存YAML配置文件
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info("=" * 60)
        logger.info("🎯 训练数据集配置文件创建成功!")
        logger.info(f"📁 配置文件路径: {output_file.resolve()}")
        logger.info(f"📊 train_dataset路径: {dataset_path.resolve()}")
        logger.info(f"🏷️  类别数量: {len(class_names)}")
        logger.info(f"📷 图像数量: {len(image_files)}")
        logger.info(f"📝 标签数量: {len(label_files)}")
        logger.info("=" * 60)
        
        # 显示配置文件内容
        logger.info("📋 生成的配置文件内容:")
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        
        logger.info("=" * 60)
        logger.info("📝 使用说明:")
        logger.info("1. 请检查并修改类别名称（将class_0等替换为实际名称）")
        logger.info("2. 如需分割训练/验证集，请创建train和val子目录")
        logger.info(f"3. 使用以下命令开始训练:")
        logger.info(f"   python train_yolo.py --data {output_file} --epochs 100")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"创建配置文件失败: {str(e)}")
        return None

def split_dataset(dataset_dir, train_ratio=0.8, val_ratio=0.15):
    """
    将train_dataset数据集分割为训练集、验证集和测试集
    
    Args:
        dataset_dir (str): train_dataset目录路径
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
    """
    try:
        import random
        import shutil
        
        dataset_path = Path(dataset_dir)
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        # 创建分割后的目录结构
        train_images_dir = dataset_path / "train" / "images"
        train_labels_dir = dataset_path / "train" / "labels"
        val_images_dir = dataset_path / "val" / "images"
        val_labels_dir = dataset_path / "val" / "labels"
        test_images_dir = dataset_path / "test" / "images"
        test_labels_dir = dataset_path / "test" / "labels"
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, 
                        val_labels_dir, test_images_dir, test_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(images_dir.glob("*.*"))
        random.shuffle(image_files)
        
        total_count = len(image_files)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        
        logger.info(f"数据集分割:")
        logger.info(f"  训练集: {train_count} 张 ({train_ratio*100:.1f}%)")
        logger.info(f"  验证集: {val_count} 张 ({val_ratio*100:.1f}%)")
        logger.info(f"  测试集: {total_count - train_count - val_count} 张")
        
        # 分割数据集
        for i, image_file in enumerate(image_files):
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            if i < train_count:
                # 训练集
                shutil.copy(image_file, train_images_dir)
                if label_file.exists():
                    shutil.copy(label_file, train_labels_dir)
            elif i < train_count + val_count:
                # 验证集
                shutil.copy(image_file, val_images_dir)
                if label_file.exists():
                    shutil.copy(label_file, val_labels_dir)
            else:
                # 测试集
                shutil.copy(image_file, test_images_dir)
                if label_file.exists():
                    shutil.copy(label_file, test_labels_dir)
        
        logger.info("✅ 数据集分割完成!")
        return True
        
    except Exception as e:
        logger.error(f"数据集分割失败: {str(e)}")
        return False

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="创建YOLO训练数据集配置文件")
    parser.add_argument("--train_dataset_dir", type=str, required=True, help="train_dataset目录路径")
    parser.add_argument("--output", type=str, default="dataset.yaml", help="输出YAML文件路径")
    parser.add_argument("--class_names", nargs='+', help="类别名称列表")
    parser.add_argument("--split", action="store_true", help="是否分割数据集")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例")
    
    args = parser.parse_args()
    
    # 创建配置文件
    config_path = create_dataset_yaml(
        args.train_dataset_dir, 
        args.output, 
        args.class_names
    )
    
    if config_path and args.split:
        # 分割数据集
        split_dataset(args.train_dataset_dir, args.train_ratio, args.val_ratio)
        
        # 更新配置文件路径
        dataset_path = Path(args.train_dataset_dir)
        updated_config = {
            'path': str(dataset_path.resolve()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(args.class_names) if args.class_names else 1,
            'names': args.class_names if args.class_names else ['class_0']
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(updated_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"✅ 已更新配置文件以支持分割后的数据集结构")

if __name__ == "__main__":
    main()
