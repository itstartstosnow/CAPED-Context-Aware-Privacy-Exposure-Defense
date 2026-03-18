"""
UIED CNN PyTorch 版本 - UI 元素分类器

从 TensorFlow 转换而来，用于替代原始的 cnn/CNN.py
解决 TensorFlow/PyTorch CUDA 冲突问题。

用法：
    from cnn.cnn_pytorch import CNN
    classifier = CNN('Elements')
    classifier.predict(imgs, compos)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2

# 添加项目路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_UIED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _UIED_ROOT not in sys.path:
    sys.path.insert(0, _UIED_ROOT)


class CNN:
    """UIED Elements 分类器 - PyTorch 版本

    与原始 TensorFlow 版本 API 兼容。
    """

    CLASS_MAP = [
        'Button', 'CheckBox', 'Chronometer', 'EditText', 'ImageButton',
        'ImageView', 'ProgressBar', 'RadioButton', 'RatingBar', 'SeekBar',
        'Spinner', 'Switch', 'ToggleButton', 'VideoView', 'TextView'
    ]

    def __init__(self, classifier_type='Elements', is_load=True, device=None):
        """
        Args:
            classifier_type: 目前只支持 'Elements'
            is_load: 是否立即加载模型
            device: 'cuda', 'cpu', 或 None (自动选择)
        """
        self.model = None
        self.classifier_type = classifier_type
        self.class_map = self.CLASS_MAP
        self.class_number = len(self.class_map)
        self.image_shape = (64, 64, 3)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if is_load:
            self.load(classifier_type)

    def load(self, classifier_type):
        """加载 PyTorch 模型"""
        if classifier_type != 'Elements':
            raise ValueError(f"目前只支持 'Elements' 分类器，得到: {classifier_type}")

        # 使用 onnx2torch 转换的模型（在 cnn/ 目录下）
        onnx_path = os.path.join(os.path.dirname(__file__), 'cnn_elements.onnx')

        if os.path.exists(onnx_path):
            # 使用 onnx2torch 加载
            from onnx2torch import convert
            self.model = convert(onnx_path)
            self.model.eval()
            self.model.to(self.device)
            print(f'Model Loaded From ONNX: {onnx_path}')
        else:
            raise FileNotFoundError(f"找不到模型文件: {onnx_path}")

    def preprocess_img(self, image):
        """预处理单张图片

        Args:
            image: RGB 图片 (H, W, 3)

        Returns:
            处理后的 tensor (1, H, W, 3) - NHWC 格式，因为 onnx2torch 保持原始格式
        """
        image = cv2.resize(image, self.image_shape[:2])
        x = (image / 255.0).astype('float32')
        x = np.array([x])  # (1, 64, 64, 3)
        return x

    def predict(self, imgs, compos, load=False, show=False):
        """
        对图片列表进行分类预测

        Args:
            imgs: 图片列表，每个是 RGB numpy array
            compos: 组件列表，会被修改添加 category 属性
            load: 是否重新加载模型
            show: 是否显示结果
        """
        if load:
            self.load(self.classifier_type)

        if self.model is None:
            print("*** No model loaded ***")
            return

        for i in range(len(imgs)):
            X = self.preprocess_img(imgs[i])
            X_tensor = torch.from_numpy(X).to(self.device)

            with torch.no_grad():
                output = self.model(X_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_class = self.class_map[pred_idx]

            compos[i].category = pred_class

            if show:
                print(pred_class)
                cv2.imshow('element', imgs[i])
                cv2.waitKey()
