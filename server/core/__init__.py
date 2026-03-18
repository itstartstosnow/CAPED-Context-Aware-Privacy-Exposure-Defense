"""核心模块 — V3

V3 变更：移除 SimilarityModule (BGE)、render、controller 自动导入。
仅保留 TextDeAnonymizer（NER）和辅助工具函数。
"""

from .anonymization import TextDeAnonymizer


def image_to_base64(image_path: str) -> str:
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def read_json(file_path: str) -> dict:
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


__all__ = ['TextDeAnonymizer', 'image_to_base64', 'read_json']
