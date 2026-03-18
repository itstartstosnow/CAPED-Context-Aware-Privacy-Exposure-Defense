"""
uiedTools.py — UIED 底层工具：模型单例 + CNN 分类 + UIED 参数

职责：
  - 管理 PaddleOCR、CNN 分类器的全局单例（延迟加载，避免重复初始化）
  - 提供 CNN class → 'icon'|'image' 映射
  - 暴露 UIED 检测参数常量

注意：业务流程（检测、合并、Florence 描述）统一在 core/uiedDetection.py 中实现。
"""

import os
import sys

# ── 路径初始化（UIED 库和项目根都加入 path）─────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_UIED = os.path.join(_HERE, 'UIED')
for _p in [_HERE, _UIED]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── PaddleOCR 单例 ──────────────────────────────────────────────────────────

_PADDLE_OCR = None

def getPaddleOCR():
    """延迟加载 PaddleOCR，全局复用同一实例。"""
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        try:
            from paddleocr import PaddleOCR
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        except Exception as e:
            print(f"[uiedTools] 加载 PaddleOCR 失败: {e}")
    return _PADDLE_OCR


# ── CNN 分类器单例 ───────────────────────────────────────────────────────────

_CNN_CLASSIFIER = None

def getCNNClassifier():
    """延迟加载 UIED CNN 分类器（PyTorch 版本，避免 TensorFlow 冲突）。"""
    global _CNN_CLASSIFIER
    if _CNN_CLASSIFIER is None:
        from cnn.cnn_pytorch import CNN
        _CNN_CLASSIFIER = CNN('Elements', device='cpu')
    return _CNN_CLASSIFIER


# ── CNN class → detection type 映射 ─────────────────────────────────────────

_IMAGE_CLASSES = {'ImageView'}
_ICON_CLASSES = {
    'Button', 'ImageButton', 'CheckBox', 'EditText',
    'RadioButton', 'Switch', 'ToggleButton', 'Spinner',
    'SeekBar', 'RatingBar', 'ProgressBar', 'Chronometer',
    'VideoView', 'TextView',
}

def cnnClassToType(cnnClass: str) -> str:
    """CNN 类名 → 'image' | 'icon'。

    未知类别（'Compo'、'Block' 等）返回 'icon'（中性），
    Big Image Rule 由 core/uiedDetection._customMerge() 在合并后统一处理。
    """
    if cnnClass in _IMAGE_CLASSES:
        return 'image'
    if cnnClass in _ICON_CLASSES:
        return 'icon'
    return 'icon'


# ── UIED 检测参数 ────────────────────────────────────────────────────────────

UIED_PARAMS = {
    'min-grad': 10,
    'ffl-block': 5,
    'min-ele-area': 50,
    'merge-contained-ele': True,
    'remove-bar': True,
    'save-result': True,
}


# ── Internal helper aliases ──────────────────────────────────────────────────
_get_paddle_ocr = getPaddleOCR
_getClassifier  = getCNNClassifier
_cnnClassToType = lambda cls, *_: cnnClassToType(cls)   # 旧接口带 bboxNorm 参数
_UIED_PARAMS    = UIED_PARAMS
