"""api/dependencies.py — V3 依赖注入

V3 变更：
- 移除 SimilarityModule (BGE)、controller.py 中的复杂编排
- 保留 GLiNER NER 模型（通过 anonymization.py）
- 新增 LLM 和 Florence-2 VQA 的预加载接口

get_controller() 仅用于 NER，返回一个轻量级包装器。
"""
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('AGENT_PRIVACY_GPU', '0'))

from core.anonymization import TextDeAnonymizer

# 全局 NER 单例



class _NERController:
    """轻量级 NER 控制器，仅包含 anonymizer。"""
    def __init__(self):
        self.anonymizer = TextDeAnonymizer()


_controller: _NERController | None = None


def get_controller() -> _NERController:
    """获取或初始化全局 NER 控制器单例。"""
    global _controller
    if _controller is None:
        print("[API] Initializing NER controller...")
        _controller = _NERController()
    return _controller
