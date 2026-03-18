import torch
import json
from typing import List, Dict, Tuple, Any
from gliner import GLiNER
import re, os
from collections import defaultdict
from copy import deepcopy


# 全局 NER 模型缓存
GLOBAL_NER_MODEL = None

# 配置：是否允许对可交互的文本/TextIcon进行 PII 脱敏检查
CHECK_INTERACTIVE_TEXT = True
CHECK_TEXTICON = True 

# NER 置信度阈值
NER_CONFIDENCE_THRESHOLD = 0.3  # 降低阈值以提升中文召回

# 预设白名单：UI 功能性词汇，不视为 PII（中英文）
# 参考 Available-but-Invisible 论文的白名单设计
whitelistClasses = [
    # --- 英文 UI 操作词 ---
    "Search", "Type", "Google", "URL", "Home", "Back", "Menu", "More",
    "Share", "Edit", "Delete", "Add", "Settings", "Close", "Send", "Like",
    "Save", "Cancel", "OK", "Yes", "No", "Done", "Submit", "Continue",
    "Next", "Previous", "Skip", "Retry", "Refresh", "Reload", "Open",
    "Copy", "Paste", "Cut", "Select", "All", "Clear", "Filter", "Sort",
    "Upload", "Download", "Import", "Export", "Login", "Logout",
    "Sign in", "Sign out", "Register", "Profile", "Help", "About",
    "Notification", "Message", "Call", "Chat", "Photo", "Camera",
    "Gallery", "Video", "Audio", "File", "Folder", "Confirm", "Apply",
    # --- 中文 UI 操作词 ---
    "确认", "取消", "发送", "返回", "搜索", "设置", "关闭", "删除",
    "编辑", "添加", "分享", "更多", "保存", "提交", "登录", "退出",
    "首页", "发现", "消息", "联系人", "拨号", "通话", "短信",
    "相机", "相册", "文件", "视频", "音乐", "下载", "上传",
    "确定", "下一步", "上一步", "跳过", "重试", "刷新", "全选",
    "复制", "粘贴", "剪切", "清除", "筛选", "排序", "注册",
    "个人", "帮助", "关于", "通知", "主页", "返回键",
]

# 短字符精确白名单（≤2字），只做整体精确匹配，避免误白名单长文本
# 主要覆盖：底部导航栏人称代词（GLiNER 容易将"我"识别为 PERSON NAME）
SHORT_EXACT_WHITELIST = {
    # 人称代词 - 不是人名
    "我", "你", "他", "她", "它",
    "我们", "你们", "他们", "她们",
    # 底部导航/Tab 常见短标签
    "我的", "主页", "发现", "消息",
}

# Regex 兜底层：对高熵结构化 PII 进行精确匹配（GLiNER 容易漏检）
# 格式: {归一化类型: [(pattern, flags), ...]}
REGEX_PII_PATTERNS = {
    'PHONE_NUMBER': [
        # 中国大陆手机号
        (r'(?<!\d)(1[3-9]\d{9})(?!\d)', 0),
        (r'(?<!\d)(\+86[\s\-]?1[3-9]\d{9})(?!\d)', 0),
        # 美国/加拿大 +1 或 (区号) 格式
        (r'(?<!\d)(\+1[\s.\-]?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})(?!\d)', 0),
        (r'(?<!\d)(\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4})(?!\d)', 0),
        # 英国 +44
        (r'(?<!\d)(\+44[\s\-]?\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4})(?!\d)', 0),
        # 德国 +49
        (r'(?<!\d)(\+49[\s\-]?\(?\d{2,5}\)?[\s\-]?\d{5,10})(?!\d)', 0),
        # 法国 +33
        (r'(?<!\d)(\+33[\s\-]?[1-9](?:[\s\-]?\d{2}){4})(?!\d)', 0),
    ],
    'EMAIL': [
        (r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', re.IGNORECASE),
    ],
    'ID_CARD': [
        # 中国居民身份证（18位）
        (r'\b(\d{17}[\dXx])\b', 0),
    ],
    'BANK_CARD': [
        # 16位银行卡（Visa/MC/银联等）
        (r'\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b', 0),
        # 美国运通 15位
        (r'\b(3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5})\b', 0),
    ],
}

def is_whitelisted_simple(detection):
    """
    Args:
        detection: 检测结果字典

    Returns:
        是否在白名单中
    """
    content = detection.get('content', '')
    if not content:
        return False
    # 短字符精确白名单：整体匹配，避免误命中长文本
    if content.strip() in SHORT_EXACT_WHITELIST:
        return True
    # 长白名单：子串匹配（UI 操作词）
    for w in whitelistClasses:
        if w in content:
            return True
    return False

class TextDeAnonymizer:
    """
    文本脱敏类：使用 urchade/gliner_multi-v2.1 进行 NER，支持中英文，进行脱敏编号，支持还原
    """
    
    def __init__(self, model_name="urchade/gliner_multi-v2.1", device="cuda", filepath=None, check_text_icon=True):
        """
        初始化脱敏器
        
        Args:
            model_name: 使用的 NER 模型名称
            device: 计算设备 ('cuda' 或 'cpu')
            check_text_icon: 是否对 textIcon 类型进行 PII 检查
        """
        self.device = device
        self.model_name = model_name
        self.check_text_icon = check_text_icon
        self.filepath = filepath
        
        if filepath: 
            self.load_mapping(filepath)
        
        global GLOBAL_NER_MODEL
        if GLOBAL_NER_MODEL is None:
            print(f"[INFO] Loading GLiNER model: {model_name}")
            # GLiNER 内部会自动处理 device
            GLOBAL_NER_MODEL = GLiNER.from_pretrained(model_name)
            if "cuda" in device:
                GLOBAL_NER_MODEL.to(device)
        
        self.model = GLOBAL_NER_MODEL
        # GLiNER zero-shot 标签：自然语言短语效果优于缩写
        # ORG 误报率高已验证禁用；password/username 误报风险也高，置信度阈值会过滤
        self.labels = [
            "person name",        # 人名（中英文）
            "phone number",       # 手机/电话号码
            "email address",      # 邮箱
            "home address",       # 家庭/通讯地址
            "id card number",     # 身份证号
            "passport number",    # 护照号
            "bank account number", # 银行账号
            "credit card number", # 信用卡号
            "social security number", # 社保/SSN
            "password",           # 密码
            "username",           # 用户名/账号
        ]
        
        # 维护的映射表：{类型: {原始值: 生成的标签}}
        self.mapping_table = defaultdict(dict)
        
        # 维护的计数器：各类型的当前编号
        self.entity_counters = defaultdict(int)

    def _identify_and_extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        识别和提取阶段：GLiNER NER + Regex 兜底，合并去重
        """
        # 1. GLiNER 预测
        ner_raw = self.model.predict_entities(text, self.labels, threshold=0.1)
        final_entities = []
        for ent in ner_raw:
            score = float(ent['score'])
            if score < NER_CONFIDENCE_THRESHOLD:
                continue
            final_entities.append({
                'type': ent['label'].upper(),
                'word': ent['text'],
                'start': ent['start'],
                'end': ent['end'],
                'score': score,
                'source': 'gliner',
            })

        # 2. Regex 兜底：覆盖 GLiNER 容易漏的结构化 PII
        # 记录已被 GLiNER 覆盖的区间，避免重复
        covered = {(e['start'], e['end']) for e in final_entities}
        for pii_type, patterns in REGEX_PII_PATTERNS.items():
            for pattern, flags in patterns:
                for m in re.finditer(pattern, text, flags):
                    span = (m.start(), m.end())
                    # 跳过与已有实体重叠的区间
                    if any(s <= m.start() < e or s < m.end() <= e
                           for s, e in covered):
                        continue
                    covered.add(span)
                    final_entities.append({
                        'type': pii_type,
                        'word': m.group(0),
                        'start': m.start(),
                        'end': m.end(),
                        'score': 1.0,  # regex 是确定性的
                        'source': 'regex',
                    })

        # 按位置排序
        final_entities.sort(key=lambda x: x['start'])
        return final_entities
    
    def _normalize_entity_type(self, ner_type: str) -> str:
        """
        将 NER 模型标签和 Regex 类型归一化为统一的短标签（用于 placeholder 前缀）
        """
        type_mapping = {
            # GLiNER 自然语言标签 → 短标签
            'PERSON NAME': 'PERSON',
            'PERSON': 'PERSON',
            '人名': 'PERSON',
            '姓名': 'PERSON',
            'PHONE NUMBER': 'PHONE',
            'PHONE': 'PHONE',
            'PHONE_NUMBER': 'PHONE',
            'EMAIL ADDRESS': 'EMAIL',
            'EMAIL': 'EMAIL',
            'HOME ADDRESS': 'ADDRESS',
            'ADDRESS': 'ADDRESS',
            'ID CARD NUMBER': 'ID_CARD',
            'ID_CARD': 'ID_CARD',
            'PASSPORT NUMBER': 'PASSPORT',
            'PASSPORT': 'PASSPORT',
            'BANK ACCOUNT NUMBER': 'BANK_ACCT',
            'BANK_ACCOUNT': 'BANK_ACCT',
            'BANK_CARD': 'BANK_CARD',
            'CREDIT CARD NUMBER': 'BANK_CARD',
            'SOCIAL SECURITY NUMBER': 'SSN',
            'SSN': 'SSN',
            'PASSWORD': 'PASSWORD',
            'USERNAME': 'USERNAME',
            # 禁用的 ORG 仍做归一保留兼容
            'ORGANIZATION': 'ORG',
            'ORG': 'ORG',
        }
        return type_mapping.get(ner_type.upper(), ner_type.upper())
    
    def _get_or_create_label(self, entity_type: str, value: str) -> str:
        """
        映射阶段：获取或创建实体的脱敏标签
        """
        normalized_type = self._normalize_entity_type(entity_type)
        
        # 检查是否已在映射表中
        if value in self.mapping_table[normalized_type]:
            return self.mapping_table[normalized_type][value]
        
        # 生成新标签
        self.entity_counters[normalized_type] += 1
        label = f"{normalized_type}_{self.entity_counters[normalized_type]}"
        
        # 存入映射表
        self.mapping_table[normalized_type][value] = label
        
        return label
    
    def anonymize_text(self, text: str) -> Tuple[str, Dict[str, List[Dict[str, Any]]]]:
        """
        脱敏阶段：将文本中的敏感信息替换为脱敏标签
        """
        entities = self._identify_and_extract_entities(text)
        anonymized_text = text
        replacements = []
        
        # 按位置反向遍历，避免索引改变
        for entity in reversed(entities):
            entity_type = entity['type']
            value = entity['word']
            start = entity['start']
            end = entity['end']
            
            # 获取或创建标签
            label = self._get_or_create_label(entity_type, value)
            
            # 替换（从后往前替换保证索引正确）
            anonymized_text = anonymized_text[:start] + label + anonymized_text[end:]
            
            replacements.append({
                'original': value,
                'label': label,
                'type': self._normalize_entity_type(entity_type),
                'original_position': (start, end)
            })
        
        return anonymized_text, {
            'entities': entities,
            'replacements': list(reversed(replacements))  # 正序输出
        }
    
    def deanonymize_text(self, anonymized_text: str, reverse_mapping: Dict[str, str]) -> str:
        """
        还原阶段：将脱敏标签还原为原始值
        """
        deanonymized_text = anonymized_text
        for label, original_value in reverse_mapping.items():
            deanonymized_text = deanonymized_text.replace(label, original_value)
        return deanonymized_text
    
    def anonymize_detections(
        self,
        detections: List[Dict[str, Any]],
        inplace: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        对检测结果进行脱敏
        """
        if inplace:
            anonymized_detections = detections
        else:
            anonymized_detections = deepcopy(detections)
            
        anonymization_info = {
            'mapping_table': {},
            'text_replacements': []
        }
        
        for i, detection in enumerate(anonymized_detections):
            det_type = detection.get('type')
            if det_type != 'text': 
                if not CHECK_TEXTICON or det_type != 'textIcon':
                    continue
                
            if is_whitelisted_simple(detection):
                continue
            
            original_text = detection['content']
            anonymized_text, meta = self.anonymize_text(original_text)

            if len(meta['entities']) > 0:
                detection['content'] = anonymized_text
                detection['original_content'] = original_text
                detection['anonymized'] = True
                detection['has_pii'] = meta 
                
                anonymization_info['text_replacements'].append({
                    'detection_id': i,
                    'original': original_text,
                    'anonymized': anonymized_text,
                    'replacements': meta['replacements']
                })
        
        for entity_type, mapping in self.mapping_table.items():
            anonymization_info['mapping_table'][entity_type] = mapping
        
        return anonymized_detections, anonymization_info
    
    def get_reverse_mapping(self) -> Dict[str, str]:
        """
        获得反向映射（标签 -> 原始值）
        """
        reverse_mapping = {}
        for entity_type, mapping in self.mapping_table.items():
            for original, label in mapping.items():
                reverse_mapping[label] = original
        return reverse_mapping
    
    def deanonymize_detections(
        self,
        anonymized_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        对脱敏的检测结果进行还原
        """
        deanonymized_detections = deepcopy(anonymized_detections)
        for detection in deanonymized_detections:
            if detection.get('anonymized') and 'original_content' in detection:
                detection['content'] = detection['original_content']
                del detection['original_content']
                del detection['anonymized']
        return deanonymized_detections
    
    def save_mapping(self):
        """
        保存映射表到文件
        """
        # 如果 filepath 为 None，跳过保存（API 模式不需要保存到文件）
        if self.filepath is None:
            return

        mapping_data = {
            'mapping_table': dict(self.mapping_table),
            'entity_counters': dict(self.entity_counters),
            'reverse_mapping': self.get_reverse_mapping()
        }
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Mapping table saved to {self.filepath}")
    
    def load_mapping(self, filepath: str):
        """
        从文件加载映射表
        """
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            self.mapping_table = defaultdict(dict, mapping_data.get('mapping_table', {}))
            self.entity_counters = defaultdict(int, mapping_data.get('entity_counters', {}))
            print(f"[INFO] Mapping table loaded from {filepath}")

    def load_mapping_from_dict(self, mapping_data: dict):
        """
        从字典加载映射表（用于 API 直接传入）

        Args:
            mapping_data: 映射表字典，包含 mapping_table 和 entity_counters
        """
        if mapping_data:
            self.mapping_table = defaultdict(dict, mapping_data.get('mapping_table', {}))
            self.entity_counters = defaultdict(int, mapping_data.get('entity_counters', {}))
            print(f"[INFO] Mapping table loaded from dict")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取脱敏统计信息
        """
        stats = {}
        for entity_type, mapping in self.mapping_table.items():
            stats[entity_type] = {
                'count': len(mapping),
                'examples': list(mapping.items())[:5]
            }
        return stats

    def processDetections(
        self,
        detections: List[Dict[str, Any]],
        filterTypes: List[str] = None,
        excludeTypes: List[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        处理检测结果：脱敏文本内容
        """
        return self.anonymize_detections(detections, inplace=True)
