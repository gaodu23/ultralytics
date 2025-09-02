#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO检测并保留原始照片EXIF信息
此脚本使用YOLOv5进行目标检测，并将原始照片的EXIF信息（如GPS坐标和拍照时间）
复制到生成的检测结果图片中。
"""

import sys
import os
from pathlib import Path

# Add the YOLOv5 root directory to the path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import cv2
from PIL import Image
import piexif
import logging
import argparse
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_exif_data(img_path):
    """
    获取图片的EXIF数据
    
    Args:
        img_path (str): 图片路径
        
    Returns:
        bytes: EXIF数据，如果没有则返回空字节
    """
    try:
        img = Image.open(img_path)
        if "exif" in img.info:
            return img.info["exif"]
        
        # 如果info中没有exif，尝试用piexif读取
        try:
            exif_dict = piexif.load(img.info.get("exif", b""))
            return piexif.dump(exif_dict)
        except:
            logger.warning(f"无法从 {img_path} 读取EXIF信息")
    except Exception as e:
        logger.error(f"处理 {img_path} 时出错: {str(e)}")
    
    return b""

def copy_exif_to_results(original_files, results_dir):
    """
    将原始图片的EXIF信息复制到检测结果图片
    
    Args:
        original_files (list): 原始图片文件路径列表
        results_dir (str): 检测结果保存目录
    """
    copied_count = 0
    failed_count = 0
    
    for img_path in original_files:
        try:
            # 获取原始图片的EXIF数据
            exif_data = get_exif_data(img_path)
            if not exif_data:
                logger.warning(f"{img_path} 没有EXIF数据")
                continue
                
            # 构建结果图片路径
            img_name = os.path.basename(img_path)
            result_img_path = os.path.join(results_dir, img_name)
            
            # 确保结果图片存在
            if not os.path.exists(result_img_path):
                logger.warning(f"结果图片不存在: {result_img_path}")
                continue
                
            # 将EXIF数据写入结果图片
            result_img = Image.open(result_img_path)
            result_img.save(result_img_path, exif=exif_data)
            
            copied_count += 1
            logger.debug(f"成功复制EXIF信息到 {result_img_path}")
            
        except Exception as e:
            failed_count += 1
            logger.error(f"处理 {img_path} 时出错: {str(e)}")
    
    logger.info(f"成功复制EXIF信息: {copied_count}张图片")
    if failed_count > 0:
        logger.warning(f"处理失败: {failed_count}张图片")

def run_detection(model_path="bestljd.pt", source_path="E:\\FLY\\100MSDCF", conf=0.1, device=""):
    """
    运行检测并保留EXIF信息
    
    Args:
        model_path (str): YOLO模型路径
        source_path (str): 图片源目录或文件
        conf (float): 置信度阈值
        device (str): 设备选择 (cpu, 0, 1, ...)
        
    Returns:
        str: 结果保存目录
    """
    try:
        # 检查源路径是否存在
        if not os.path.exists(source_path):
            logger.error(f"源路径不存在: {source_path}")
            return None
            
        # 加载模型
        logger.info(f"正在加载模型: {model_path}")
        logger.info(f"使用设备: {device if device else '自动选择'}")
        
        model = YOLO(model_path)
        if device:
            model.to(device)
        
        # 创建结果保存目录 - 在输入文件夹内创建results文件夹
        input_dir = Path(source_path)
        if input_dir.is_file():
            # 如果输入是单个文件，使用文件所在目录
            input_dir = input_dir.parent
        
        save_dir = input_dir / "detection_results"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始检测: {source_path}")
        
        detected_count = 0
        total_processed = 0
        exif_success_count = 0
        exif_failed_count = 0
        
        logger.info(f"🚀 开始处理图片，置信度阈值: {conf}")
        
        # 获取所有图片文件
        if os.path.isfile(source_path):
            image_files = [source_path]
        else:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(source_path).glob(f"*{ext}"))
                image_files.extend(Path(source_path).glob(f"*{ext.upper()}"))
        
        # 运行推理
        for img_path in image_files:
            total_processed += 1
            img_path = str(img_path)
            
            try:
                # 使用模型进行预测
                results = model(img_path, conf=conf, device=device)
                
                # 处理检测结果
                result = results[0]  # 获取第一个结果
                
                if len(result.boxes) > 0:
                    detected_count += 1
                    
                    # 统计各类别检测数量
                    class_counts = {}
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        class_name = result.names[cls_id]
                        if class_name not in class_counts:
                            class_counts[class_name] = 0
                        class_counts[class_name] += 1
                    
                    # 打印检测结果详情
                    detection_info = ", ".join([f"{name}: {count}个" for name, count in class_counts.items()])
                    logger.info(f"📷 {Path(img_path).name} - 检测到 {len(result.boxes)} 个目标 ({detection_info})")
                    
                    # 绘制检测框并保存图片
                    annotated_img = result.plot()
                    
                    # 保存结果图像
                    img_name = Path(img_path).name
                    save_path = str(save_dir / img_name)
                    
                    # 使用OpenCV保存图片
                    cv2.imwrite(save_path, annotated_img)
                    
                    logger.info(f"✅ 已保存检测结果图片: {img_name}")
                    
                    # 立即获取并写入EXIF信息
                    try:
                        exif_data = get_exif_data(img_path)
                        if exif_data:
                            # 将EXIF数据写入刚保存的结果图片
                            result_img = Image.open(save_path)
                            result_img.save(save_path, exif=exif_data)
                            exif_success_count += 1
                        else:
                            logger.warning(f"⚠️  {img_name} 没有EXIF数据")
                            exif_failed_count += 1
                    except Exception as e:
                        logger.error(f"❌ 写入EXIF信息失败 {img_name}: {str(e)}")
                        exif_failed_count += 1
                        
                    # 打印每个检测框的详细信息
                    for box in result.boxes:
                        cls_id = int(box.cls)
                        class_name = result.names[cls_id]
                        confidence = float(box.conf)
                        logger.debug(f"  - {class_name}: 置信度 {confidence:.3f}")
                        
                else:
                    logger.debug(f"🔍 {Path(img_path).name} - 未检测到任何目标")
                    
            except Exception as e:
                logger.error(f"处理图片 {img_path} 时出错: {str(e)}")
                continue
            
            # 每处理10张图片打印一次进度
            if total_processed % 10 == 0:
                logger.info(f"⏳ 已处理 {total_processed} 张图片，检测到目标的图片: {detected_count} 张")
        
        
        logger.info("=" * 60)
        logger.info("🎯 检测任务完成统计:")
        logger.info(f"📊 总共处理图片: {total_processed} 张")
        logger.info(f"✅ 检测到目标的图片: {detected_count} 张")
        logger.info(f"❌ 未检测到目标的图片: {total_processed - detected_count} 张")
        logger.info(f"📄 成功写入EXIF信息: {exif_success_count} 张")
        if exif_failed_count > 0:
            logger.info(f"⚠️  EXIF信息处理失败: {exif_failed_count} 张")
        
        if detected_count == 0:
            logger.info("🔍 没有找到符合置信度要求的检测结果")
            return None
            
        detection_rate = (detected_count / total_processed) * 100
        logger.info(f"🎯 目标检测率: {detection_rate:.1f}%")
        logger.info(f"📁 结果保存在: {save_dir}")
        logger.info("=" * 60)
        return str(save_dir)
        
    except Exception as e:
        logger.error(f"检测过程中出错: {str(e)}")
        return None

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="使用YOLO进行目标检测并保留原始EXIF信息")
    parser.add_argument("--model", type=str, default="bestljd.pt", help="YOLO模型路径")
    parser.add_argument("--source", type=str, default="E:\\FLY\\100MSDCF", help="图片源目录或文件")
    parser.add_argument("--conf", type=float, default=0.1, help="置信度阈值")
    parser.add_argument("--device", type=str, default="", help="设备选择 (cpu, 0, 1, ...)")
    args = parser.parse_args()
    
    run_detection(args.model, args.source, args.conf, args.device)


if __name__ == "__main__":
    # 直接运行的默认配置
    # 你可以在这里修改默认的模型路径、图片源目录和置信度阈值
    model_path = "./yolo11n.pt"  # 模型路径
    source_path = "E:\\BaiduNetdiskDownload\\WQM150101_0"  # 图片源目录
    conf = 0.7  # 置信度阈值
    device = ""  # 设备选择，空字符串表示自动选择
    
    run_detection(model_path, source_path, conf, device)


