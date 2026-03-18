"""工具函数"""
import base64
import json
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """将图片文件转换为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def base64_to_image(base64_str: str) -> bytes:
    """将 base64 字符串转换为图片字节"""
    return base64.b64decode(base64_str)


def read_json(file_path: str) -> dict:
    """读取 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
