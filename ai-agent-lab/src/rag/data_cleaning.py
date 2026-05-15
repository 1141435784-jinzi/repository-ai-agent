"""
=== 企业级文档数据清洗流水线 ===

【核心功能】实现7步文档处理流水线，将原始文档转换为高质量知识库内容：

1. 文档解析：PDF/Word/Excel/图片/OCR/表格提取
2. 基础清洗：去页眉页脚、去水印、去乱码、合并空行
3. 结构还原：标题层级、列表、表格转Markdown/JSON
4. 内容净化：删除模板、重复段落、低信息密度文本
5. 安全脱敏：手机号、身份证、涉密内容过滤/打码
6. 分块优化：按语义/标题切分，防止截断
7. 入库校验：重复检测、质量评分、版本管理

【设计原则】
- 模块化：每步都是独立的处理单元，可单独调用或组合
- 可配置：支持自定义清洗规则和参数
- 可扩展：支持新增文档类型和清洗规则
- 可观测：每步处理都有日志和质量指标
"""

import os
import re
import hashlib
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from collections import Counter

import logging

logger = logging.getLogger(__name__)

# 从配置文件导入分块参数
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentType(Enum):
    """支持的文档类型"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    XLS = "xls"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class CleanResult:
    """清洗结果数据结构"""
    success: bool
    content: str
    metadata: Dict[str, Any]
    errors: List[str] = None
    quality_score: float = 0.0
    duplicate_detected: bool = False
    duplicate_hash: str = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ChunkInfo:
    """文档块信息"""
    content: str
    chunk_index: int
    total_chunks: int
    heading: str = None
    section: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataCleaner:
    """
    企业级文档数据清洗器
    
    实现完整的7步文档处理流水线：
    1. 文档解析 → 2. 基础清洗 → 3. 结构还原 → 4. 内容净化 → 5. 安全脱敏 → 6. 分块优化 → 7. 入库校验
    """

    def __init__(self, knowledge_base_dir: str = None):
        """
        初始化数据清洗器
        
        Args:
            knowledge_base_dir: 知识库目录路径，用于重复检测和版本管理
        """
        self.knowledge_base_dir = knowledge_base_dir
        self._setup_patterns()
        logger.info("数据清洗器初始化完成")

    def _setup_patterns(self):
        """设置正则表达式模式"""
        # 页眉页脚模式
        self.header_patterns = [
            r'^\s*[第页]\s*\d+\s*[页章]\s*$',  # "第1页", "第3章"
            r'^\s*\d+\s*/\s*\d+\s*$',         # "1/10", "5/20"
            r'^\s*-\s*\d+\s*-\s*$',           # "- 5 -"
            r'^\s*[\u4e00-\u9fff]+有限公司\s*$',  # 公司名称作为页眉
            r'^\s*[\u4e00-\u9fff]+股份\s*$',    # 股份公司名称
        ]

        # 水印模式
        self.watermark_patterns = [
            r'[\u4e00-\u9fff]{0,2}水印[\u4e00-\u9fff]{0,2}',
            r'CONFIDENTIAL',
            r'INTERNAL USE ONLY',
            r'DRAFT',
            r'草稿',
            r'内部资料',
            r'保密',
        ]

        # 敏感信息模式
        self.sensitive_patterns = {
            'phone': re.compile(r'1[3-9]\d{9}'),  # 手机号
            'id_card': re.compile(r'\d{17}[\dXx]|\d{15}'),  # 身份证号
            'bank_card': re.compile(r'\d{16,19}'),  # 银行卡号
            'email': re.compile(r'[\w.-]+@[\w.-]+\.\w+'),  # 邮箱
            'ip': re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'),  # IP地址
        }

        # 低信息密度模式
        self.low_info_patterns = [
            r'^\s*[一-九]、\s*$',           # 空的章节标题
            r'^\s*[（\(]\s*[\d\u4e00-\u9fff]+\s*[）\)]\s*$',  # 空的列表项
            r'^\s*[-*•·]\s*$',              # 空的项目符号
            r'^\s*=\s*$',                   # 分隔线
            r'^\s*[-]{3,}\s*$',             # 短横线分隔
            r'^\s*[*]{3,}\s*$',             # 星号分隔
            r'^\s*#{1,6}\s*$',              # 空的Markdown标题
        ]

    # ============================================================
    # 第一步：文档解析
    # ============================================================
    def parse_document(self, file_path: str) -> CleanResult:
        """
        解析文档，提取文本内容
        
        支持的格式：PDF、Word(docx/doc)、Excel(xlsx/xls)、TXT、MD、HTML、图片(OCR)
        
        Args:
            file_path: 文件路径
            
        Returns:
            CleanResult: 解析结果
        """
        logger.info(f"开始解析文档: {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            return CleanResult(
                success=False,
                content="",
                metadata={"file_name": file_path.name},
                errors=["文件不存在"]
            )

        doc_type = self._detect_document_type(file_path)
        content = ""
        metadata = {
            "file_name": file_path.name,
            "file_type": doc_type.value,
            "file_size": file_path.stat().st_size,
            "parse_time": datetime.now().isoformat(),
        }

        try:
            if doc_type == DocumentType.PDF:
                content = self._parse_pdf(file_path)
            elif doc_type in (DocumentType.DOCX, DocumentType.DOC):
                content = self._parse_word(file_path)
            elif doc_type in (DocumentType.XLSX, DocumentType.XLS):
                content = self._parse_excel(file_path)
            elif doc_type == DocumentType.TXT:
                content = self._parse_txt(file_path)
            elif doc_type == DocumentType.MD:
                content = self._parse_md(file_path)
            elif doc_type == DocumentType.HTML:
                content = self._parse_html(file_path)
            elif doc_type == DocumentType.IMAGE:
                content = self._parse_image(file_path)
            else:
                content = self._parse_txt(file_path)  # 未知类型尝试按文本解析

            if content:
                metadata["original_length"] = len(content)
                logger.info(f"文档解析成功，原始文本长度: {len(content)}")
                return CleanResult(success=True, content=content, metadata=metadata)
            else:
                return CleanResult(
                    success=False,
                    content="",
                    metadata=metadata,
                    errors=["无法提取文档内容"]
                )

        except Exception as e:
            logger.error(f"文档解析失败: {e}")
            return CleanResult(
                success=False,
                content="",
                metadata=metadata,
                errors=[str(e)]
            )

    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """检测文档类型"""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return DocumentType.PDF
        elif ext == ".docx":
            return DocumentType.DOCX
        elif ext == ".doc":
            return DocumentType.DOC
        elif ext == ".xlsx":
            return DocumentType.XLSX
        elif ext == ".xls":
            return DocumentType.XLS
        elif ext == ".txt":
            return DocumentType.TXT
        elif ext == ".md":
            return DocumentType.MD
        elif ext == ".html":
            return DocumentType.HTML
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            return DocumentType.IMAGE
        return DocumentType.UNKNOWN

    def _parse_pdf(self, file_path: Path) -> str:
        """解析PDF文档"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            content = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n\n"
            return content.strip()
        except ImportError:
            # 备用方案：使用unstructured
            return self._parse_with_unstructured(file_path)
        except Exception as e:
            logger.warning(f"pypdf解析失败，尝试unstructured: {e}")
            return self._parse_with_unstructured(file_path)

    def _parse_word(self, file_path: Path) -> str:
        """解析Word文档"""
        try:
            from docx import Document
            doc = Document(str(file_path))
            content = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return content.strip()
        except ImportError:
            return self._parse_with_unstructured(file_path)
        except Exception as e:
            logger.warning(f"python-docx解析失败，尝试unstructured: {e}")
            return self._parse_with_unstructured(file_path)

    def _parse_excel(self, file_path: Path) -> str:
        """解析Excel文档"""
        try:
            import pandas as pd
            df = pd.read_excel(str(file_path))
            # 将表格转为Markdown格式
            return df.to_markdown(index=False)
        except ImportError:
            return self._parse_with_unstructured(file_path)
        except Exception as e:
            logger.warning(f"pandas解析失败，尝试unstructured: {e}")
            return self._parse_with_unstructured(file_path)

    def _parse_txt(self, file_path: Path) -> str:
        """解析文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                return f.read()

    def _parse_md(self, file_path: Path) -> str:
        """解析Markdown文件"""
        return self._parse_txt(file_path)

    def _parse_html(self, file_path: Path) -> str:
        """解析HTML文件"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text(separator='\n\n').strip()
        except ImportError:
            return self._parse_txt(file_path)

    def _parse_image(self, file_path: Path) -> str:
        """解析图片文件（OCR）"""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(file_path)
            return pytesseract.image_to_string(img, lang='chi_sim')
        except ImportError:
            logger.warning("OCR依赖未安装，无法解析图片")
            return ""
        except Exception as e:
            logger.error(f"OCR解析失败: {e}")
            return ""

    def _parse_with_unstructured(self, file_path: Path) -> str:
        """使用unstructured解析文档（通用方案）"""
        try:
            from unstructured.partition.auto import partition
            elements = partition(str(file_path))
            return "\n\n".join([str(el) for el in elements]).strip()
        except Exception as e:
            logger.error(f"unstructured解析失败: {e}")
            return ""

    # ============================================================
    # 第二步：基础清洗
    # ============================================================
    def basic_clean(self, result: CleanResult) -> CleanResult:
        """
        基础清洗：去页眉页脚、去水印、去乱码、合并空行
        
        Args:
            result: 前一步的处理结果
            
        Returns:
            CleanResult: 清洗后的结果
        """
        if not result.success:
            return result

        logger.info("开始基础清洗")
        content = result.content

        # 1. 去除页眉页脚
        content = self._remove_headers_footers(content)

        # 2. 去除水印
        content = self._remove_watermarks(content)

        # 3. 去除乱码和特殊字符
        content = self._remove_gibberish(content)

        # 4. 合并空行
        content = self._merge_empty_lines(content)

        # 5. 去除多余空白字符
        content = self._clean_whitespace(content)

        result.content = content
        result.metadata["cleaned_length"] = len(content)
        logger.info(f"基础清洗完成，清洗后长度: {len(content)}")
        return result

    def _remove_headers_footers(self, content: str) -> str:
        """去除页眉页脚"""
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                cleaned_lines.append(line)
                continue
            
            # 检测是否为页眉页脚
            is_header_footer = False
            for pattern in self.header_patterns:
                if re.match(pattern, line):
                    is_header_footer = True
                    break
            
            # 检测短行（通常页眉页脚较短）
            if len(line) < 10 and re.match(r'^\s*\d+\s*$', line):
                is_header_footer = True
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _remove_watermarks(self, content: str) -> str:
        """去除水印文字"""
        for pattern in self.watermark_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        return content

    def _remove_gibberish(self, content: str) -> str:
        """去除乱码和不可见字符"""
        # 保留中文、英文、数字、常用标点
        allowed_chars = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s，。！？；：、""\'\'（）{}【】《》<>/\-._@#$%^&*+=|\\~`·…]'
        content = re.sub(f'[^{allowed_chars}]', '', content)
        
        # 去除连续的特殊字符
        content = re.sub(r'([，。！？；：、]){3,}', r'\1', content)
        content = re.sub(r'([a-zA-Z]){20,}', '', content)  # 过长的英文字母串（可能是乱码）
        
        return content

    def _merge_empty_lines(self, content: str) -> str:
        """合并连续的空行"""
        return re.sub(r'\n{3,}', '\n\n', content)

    def _clean_whitespace(self, content: str) -> str:
        """清理多余空白字符"""
        # 去除行首行尾空格
        lines = [line.strip() for line in content.split('\n')]
        # 去除全角空格（保留单个作为分隔）
        content = '\n'.join(lines)
        content = re.sub(r'　{2,}', '　', content)  # 全角空格
        content = re.sub(r' {2,}', ' ', content)    # 半角空格
        return content.strip()

    # ============================================================
    # 第三步：结构还原
    # ============================================================
    def structure_restore(self, result: CleanResult) -> CleanResult:
        """
        结构还原：标题层级、列表、表格转Markdown/JSON
        
        Args:
            result: 前一步的处理结果
            
        Returns:
            CleanResult: 结构还原后的结果
        """
        if not result.success:
            return result

        logger.info("开始结构还原")
        content = result.content

        # 1. 识别并标记标题层级
        content = self._identify_headings(content)

        # 2. 识别列表并转换为Markdown格式
        content = self._convert_lists(content)

        # 3. 识别表格结构
        content = self._convert_tables(content)

        # 4. 修复Markdown格式
        content = self._fix_markdown_format(content)

        result.content = content
        logger.info("结构还原完成")
        return result

    def _identify_headings(self, content: str) -> str:
        """识别标题层级并转换为Markdown格式"""
        lines = content.split('\n')
        result_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                result_lines.append(line)
                continue
            
            # 检测标题模式
            # 模式1："第一章 xxx"、"第1章 xxx"
            chapter_match = re.match(r'^(第[\d\u4e00-\u9fff]+[章节篇])\s+(.+)', line)
            if chapter_match:
                result_lines.append(f'# {chapter_match.group(2)}')  # 一级标题
                continue
            
            # 模式2："1. xxx"、"1.1 xxx"、"1.1.1 xxx" - 多级标题
            num_match = re.match(r'^(\d+(\.\d+)*)\s+(.+)', line)
            if num_match:
                level = len(num_match.group(1).split('.'))
                if level <= 6:
                    result_lines.append(f"{'#' * level} {num_match.group(3)}")
                    continue
            
            # 模式3："一、xxx"、"（一）xxx"、"1. xxx" - 中文数字列表/标题
            cn_num_patterns = [
                (r'^([\u4e00-\u9fff]+)、\s+(.+)', 2),  # "一、xxx" - 二级标题
                (r'^（[\u4e00-\u9fff]+）\s+(.+)', 3),  # "（一）xxx" - 三级标题
                (r'^\([\u4e00-\u9fff]+\)\s+(.+)', 3),  # "(一)xxx" - 三级标题
            ]
            matched = False
            for pattern, level in cn_num_patterns:
                match = re.match(pattern, line)
                if match:
                    result_lines.append(f"{'#' * level} {match.group(len(match.groups()))}")
                    matched = True
                    break
            if matched:
                continue
            
            # 模式4：纯数字标题 "1"、"2" 且上下有空白行
            if re.match(r'^\d+$', line):
                if (i > 0 and not lines[i-1].strip()) and (i < len(lines)-1 and not lines[i+1].strip()):
                    # 前后都是空行，可能是章节号
                    result_lines.append(f'## {line}')
                    continue
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)

    def _convert_lists(self, content: str) -> str:
        """识别列表并转换为Markdown格式"""
        lines = content.split('\n')
        result_lines = []
        in_list = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                in_list = False
                result_lines.append(line)
                continue
            
            # 检测列表项
            list_match = re.match(r'^([-*•·●○◇◆■□▸▶➤→])\s+(.+)', stripped)
            if list_match:
                result_lines.append(f'- {list_match.group(2)}')
                in_list = True
                continue
            
            # 检测数字列表
            num_list_match = re.match(r'^(\d+)\s*[、.．]\s+(.+)', stripped)
            if num_list_match:
                result_lines.append(f'{num_list_match.group(1)}. {num_list_match.group(2)}')
                in_list = True
                continue
            
            # 检测嵌套列表（缩进的）
            if in_list and line.startswith('    '):
                nested = line.strip()
                nested_match = re.match(r'^([-*•·●])?\s*(.+)', nested)
                if nested_match:
                    result_lines.append(f'  - {nested_match.group(len(nested_match.groups()))}')
                    continue
            
            in_list = False
            result_lines.append(line)
        
        return '\n'.join(result_lines)

    def _convert_tables(self, content: str) -> str:
        """识别表格结构并转换为Markdown格式"""
        lines = content.split('\n')
        result_lines = []
        table_buffer = []
        in_table = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检测表格行（包含多个连续的分隔符）
            if re.search(r'[-=]{3,}', stripped) and len(table_buffer) > 0:
                # 可能是表格分隔线
                table_buffer.append(line)
                in_table = True
                continue
            
            # 如果在表格模式中，检查是否为数据行
            if in_table:
                # 检测是否有列分隔模式
                if re.search(r'[\t|]', stripped):
                    table_buffer.append(line)
                    continue
                else:
                    # 表格结束，处理并输出
                    markdown_table = self._convert_to_markdown_table(table_buffer)
                    result_lines.append(markdown_table)
                    table_buffer = []
                    in_table = False
            
            # 检测可能的表格开始（包含竖线分隔）
            if '|' in stripped and not in_table:
                table_buffer = [line]
                continue
            
            result_lines.append(line)
        
        # 处理剩余的表格
        if table_buffer:
            markdown_table = self._convert_to_markdown_table(table_buffer)
            result_lines.append(markdown_table)
        
        return '\n'.join(result_lines)

    def _convert_to_markdown_table(self, lines: List[str]) -> str:
        """将表格行转换为Markdown表格格式"""
        if len(lines) < 2:
            return '\n'.join(lines)
        
        # 统一使用竖线分隔
        result = []
        for i, line in enumerate(lines):
            # 替换制表符为竖线
            line = line.replace('\t', '|')
            # 确保每行以竖线开头和结尾
            if not line.startswith('|'):
                line = '| ' + line
            if not line.endswith('|'):
                line = line + ' |'
            
            # 添加表头分隔行
            if i == 1 and not re.search(r'[-=]{3,}', line):
                # 获取第一行的列数
                col_count = len(line.split('|')) - 2
                result.append(line)
                result.append('|' + '---|' * col_count)
                continue
            
            result.append(line)
        
        return '\n'.join(result)

    def _fix_markdown_format(self, content: str) -> str:
        """修复Markdown格式问题"""
        # 确保标题后有空行
        content = re.sub(r'(#+.+?)\n([^#\n])', r'\1\n\n\2', content)
        # 确保列表后有空行
        content = re.sub(r'([-*]\s+.+?)\n([^-\n*])', r'\1\n\n\2', content)
        return content

    # ============================================================
    # 第四步：内容净化
    # ============================================================
    def content_purify(self, result: CleanResult) -> CleanResult:
        """
        内容净化：删除模板、重复段落、低信息密度文本
        
        Args:
            result: 前一步的处理结果
            
        Returns:
            CleanResult: 净化后的结果
        """
        if not result.success:
            return result

        logger.info("开始内容净化")
        content = result.content

        # 1. 删除模板和重复段落
        content = self._remove_duplicates(content)

        # 2. 删除低信息密度文本
        content = self._remove_low_info(content)

        # 3. 删除常见模板内容
        content = self._remove_templates(content)

        result.content = content
        logger.info(f"内容净化完成，净化后长度: {len(content)}")
        return result

    def _remove_duplicates(self, content: str) -> str:
        """删除重复段落"""
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        seen = set()
        unique_paragraphs = []
        
        for para in paragraphs:
            # 使用哈希检测重复
            para_hash = hashlib.md5(para.strip().encode('utf-8')).hexdigest()
            if para_hash not in seen:
                seen.add(para_hash)
                unique_paragraphs.append(para)
        
        return '\n\n'.join(unique_paragraphs)

    def _remove_low_info(self, content: str) -> str:
        """删除低信息密度文本"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue
            
            # 检测低信息模式
            is_low_info = False
            for pattern in self.low_info_patterns:
                if re.match(pattern, stripped):
                    is_low_info = True
                    break
            
            # 检测过短的行（可能是噪声）
            if len(stripped) < 3 and not re.match(r'^[#*\-]', stripped):
                is_low_info = True
            
            # 检测重复字符（如 "========="）
            if len(set(stripped)) <= 2 and len(stripped) > 5:
                is_low_info = True
            
            if not is_low_info:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _remove_templates(self, content: str) -> str:
        """删除常见模板内容"""
        template_patterns = [
            r'^\s*公司名称\s*$',
            r'^\s*日期\s*$',
            r'^\s*编制\s*$',
            r'^\s*审核\s*$',
            r'^\s*批准\s*$',
            r'^\s*版本号\s*$',
            r'^\s*机密\s*$',
            r'^\s*内部使用\s*$',
            r'^\s*文档编号\s*$',
            r'^\s*修订记录\s*$',
            r'^\s*页\s*/\s*页\s*$',
        ]
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            is_template = False
            
            for pattern in template_patterns:
                if re.match(pattern, stripped):
                    is_template = True
                    break
            
            if not is_template:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    # ============================================================
    # 第五步：安全脱敏
    # ============================================================
    def security_desensitize(self, result: CleanResult) -> CleanResult:
        """
        安全脱敏：手机号、身份证、涉密内容过滤/打码
        
        Args:
            result: 前一步的处理结果
            
        Returns:
            CleanResult: 脱敏后的结果
        """
        if not result.success:
            return result

        logger.info("开始安全脱敏")
        content = result.content
        sensitive_count = {}

        # 对每种敏感信息进行脱敏处理
        for info_type, pattern in self.sensitive_patterns.items():
            matches = pattern.findall(content)
            if matches:
                sensitive_count[info_type] = len(matches)
                # 替换为脱敏标记
                content = pattern.sub(f'[{info_type.upper()}]', content)

        # 添加自定义涉密关键词过滤
        secret_keywords = [
            '机密', '绝密', '秘密', '内部资料', '内部信息',
            'password', 'secret', 'token', 'api_key', '密钥'
        ]
        for keyword in secret_keywords:
            content = re.sub(keyword, '[SECRET]', content, flags=re.IGNORECASE)

        result.content = content
        result.metadata["sensitive_info_found"] = sensitive_count
        logger.info(f"安全脱敏完成，发现敏感信息: {sensitive_count}")
        return result

    # ============================================================
    # 第六步：分块优化
    # ============================================================
    def chunk_optimize(self, result: CleanResult, chunk_size: int = 500, 
                       chunk_overlap: int = 50) -> List[ChunkInfo]:
        """
        分块优化：按语义/标题切分，防止截断
        
        Args:
            result: 前一步的处理结果
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            
        Returns:
            List[ChunkInfo]: 切分后的文档块列表
        """
        if not result.success:
            return []

        logger.info(f"开始分块优化 (chunk_size={chunk_size}, overlap={chunk_overlap})")
        content = result.content
        
        # 使用LangChain的文本分块器
        from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
        
        # 优先使用Markdown分块器（保留结构）
        try:
            splitter = MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            docs = splitter.create_documents([content])
        except Exception:
            # 回退到通用分块器
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            )
            docs = splitter.create_documents([content])

        chunks = []
        for i, doc in enumerate(docs):
            chunk_info = ChunkInfo(
                content=doc.page_content,
                chunk_index=i,
                total_chunks=len(docs),
                heading=doc.metadata.get('heading'),
                section=doc.metadata.get('section'),
                metadata=doc.metadata
            )
            chunks.append(chunk_info)

        logger.info(f"分块完成，共生成 {len(chunks)} 个块")
        return chunks

    # ============================================================
    # 第七步：入库前校验
    # ============================================================
    def validate_before_storage(self, result: CleanResult, 
                                knowledge_base_path: str = None) -> CleanResult:
        """
        入库前校验：重复检测、质量评分、版本管理
        
        Args:
            result: 前一步的处理结果
            knowledge_base_path: 知识库路径，用于重复检测
            
        Returns:
            CleanResult: 校验后的结果（包含质量评分和重复检测结果）
        """
        if not result.success:
            return result

        logger.info("开始入库前校验")

        # 1. 重复检测
        duplicate_hash = self._check_duplicate(result.content, knowledge_base_path)
        result.duplicate_detected = duplicate_hash is not None
        result.duplicate_hash = duplicate_hash

        # 2. 质量评分
        quality_score = self._calculate_quality(result)
        result.quality_score = quality_score

        # 3. 添加版本信息
        result.metadata["version"] = str(uuid.uuid4())[:8]
        result.metadata["created_at"] = datetime.now().isoformat()
        result.metadata["validated"] = True

        logger.info(f"入库校验完成 - 质量评分: {quality_score:.2f}, 重复检测: {result.duplicate_detected}")
        return result

    def _check_duplicate(self, content: str, knowledge_base_path: str = None) -> Optional[str]:
        """检查是否与知识库中已有文档重复"""
        if not knowledge_base_path:
            knowledge_base_path = self.knowledge_base_dir
        
        if not knowledge_base_path or not os.path.exists(knowledge_base_path):
            return None

        # 计算当前内容的哈希
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        # 遍历知识库中的所有md文件
        kb_path = Path(knowledge_base_path)
        for md_file in kb_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                    existing_hash = hashlib.md5(existing_content.encode('utf-8')).hexdigest()
                    if existing_hash == content_hash:
                        logger.warning(f"检测到重复文档: {md_file.name}")
                        return existing_hash
            except Exception as e:
                logger.debug(f"检查文件失败: {md_file.name} - {e}")

        return None

    def _calculate_quality(self, result: CleanResult) -> float:
        """计算文档质量评分（0-100）"""
        content = result.content
        score = 50.0  # 基础分

        # 长度评分（越长越好，但超过一定长度后收益递减）
        length = len(content)
        if length < 50:
            score -= 20
        elif length < 200:
            score += 5
        elif length < 500:
            score += 10
        elif length < 2000:
            score += 15
        else:
            score += 20

        # 内容丰富度评分
        # 中文字符比例
        chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        chinese_ratio = chinese_chars / max(length, 1)
        if chinese_ratio > 0.3:
            score += 10
        elif chinese_ratio < 0.1:
            score -= 10

        # 标点符号比例（太密或太疏都不好）
        punctuation_count = sum(1 for c in content if c in '，。！？；：、')
        punctuation_ratio = punctuation_count / max(length, 1)
        if 0.02 < punctuation_ratio < 0.1:
            score += 5
        else:
            score -= 5

        # 结构评分（标题数量）
        heading_count = content.count('# ') + content.count('\n#')
        if heading_count >= 2:
            score += 10
        elif heading_count == 1:
            score += 5

        # 表格评分
        if '|' in content and '---' in content:
            score += 5

        # 列表评分
        if '\n-' in content or '\n*' in content:
            score += 5

        # 归一化到0-100
        return max(0, min(100, score))

    # ============================================================
    # 完整流水线执行
    # ============================================================
    def process(self, file_path: str, chunk_size: int = CHUNK_SIZE, 
                chunk_overlap: int = CHUNK_OVERLAP) -> Tuple[CleanResult, List[ChunkInfo]]:
        """
        执行完整的7步文档处理流水线
        
        Args:
            file_path: 文件路径
            chunk_size: 分块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            Tuple[CleanResult, List[ChunkInfo]]: 清洗结果和分块列表
        """
        logger.info(f"开始处理文档: {file_path}")
        
        # 1. 文档解析
        result = self.parse_document(file_path)
        if not result.success:
            logger.error(f"文档解析失败: {result.errors}")
            return result, []
        
        # 2. 基础清洗
        result = self.basic_clean(result)
        
        # 3. 结构还原
        result = self.structure_restore(result)
        
        # 4. 内容净化
        result = self.content_purify(result)
        
        # 5. 安全脱敏
        result = self.security_desensitize(result)
        
        # 6. 分块优化
        chunks = self.chunk_optimize(result, chunk_size, chunk_overlap)
        
        # 7. 入库前校验
        result = self.validate_before_storage(result)
        
        logger.info(f"文档处理完成 - 质量评分: {result.quality_score:.2f}, "
                   f"分块数: {len(chunks)}, 重复: {result.duplicate_detected}")
        
        return result, chunks

    def save_to_knowledge_base(self, result: CleanResult, chunks: List[ChunkInfo],
                               knowledge_base_path: str, doc_name: str = None) -> bool:
        """
        将处理后的文档保存到知识库
        
        Args:
            result: 清洗结果
            chunks: 分块列表
            knowledge_base_path: 知识库路径
            doc_name: 文档名称（不含扩展名）
            
        Returns:
            bool: 是否保存成功
        """
        if not result.success:
            logger.error("无法保存，文档处理失败")
            return False

        if result.duplicate_detected:
            logger.warning("文档重复，跳过保存")
            return False

        try:
            # 确保知识库目录存在
            kb_path = Path(knowledge_base_path)
            kb_path.mkdir(parents=True, exist_ok=True)

            # 生成文件名
            if doc_name:
                base_name = doc_name
            else:
                base_name = result.metadata.get('file_name', 'document').rsplit('.', 1)[0]
            
            # 保存完整文档
            full_path = kb_path / f"{base_name}.md"
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(result.content)
            
            # 如果分块数较多，也保存分块
            if len(chunks) > 1:
                chunks_dir = kb_path / f"{base_name}_chunks"
                chunks_dir.mkdir(exist_ok=True)
                for chunk in chunks:
                    chunk_path = chunks_dir / f"chunk_{chunk.chunk_index:03d}.md"
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        header = f"---\nheading: {chunk.heading or 'N/A'}\n" \
                                f"section: {chunk.section or 'N/A'}\n" \
                                f"chunk_index: {chunk.chunk_index}\n" \
                                f"total_chunks: {chunk.total_chunks}\n---\n\n"
                        f.write(header + chunk.content)

            # 更新元数据文件
            metadata_path = kb_path / f"{base_name}_metadata.json"
            metadata = {
                "file_name": full_path.name,
                "original_file": result.metadata.get('file_name', ''),
                "version": result.metadata.get('version', ''),
                "quality_score": result.quality_score,
                "created_at": result.metadata.get('created_at', ''),
                "chunk_count": len(chunks),
                "sensitive_info_found": result.metadata.get('sensitive_info_found', {}),
                "original_length": result.metadata.get('original_length', 0),
                "cleaned_length": result.metadata.get('cleaned_length', 0),
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"文档已保存到知识库: {full_path}")
            return True

        except Exception as e:
            logger.error(f"保存文档失败: {e}")
            return False