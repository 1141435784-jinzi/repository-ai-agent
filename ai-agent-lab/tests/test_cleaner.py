"""测试文档清洗器"""
from src.rag import DataCleaner

# 创建测试文件
test_content = """第1章 企业级文档处理

一、项目背景
随着企业数字化转型的加速，文档处理成为核心需求。

二、核心功能
- 文档解析：支持PDF、Word、Excel等格式
- 数据清洗：去噪、去重、结构化
- 安全脱敏：保护敏感信息

三、联系方式
联系人：张三
手机号：13812345678
邮箱：zhangsan@company.com

=== 结束 ===
"""

# 写入测试文件
with open('test_doc.txt', 'w', encoding='utf-8') as f:
    f.write(test_content)

# 测试数据清洗器
cleaner = DataCleaner()
result, chunks = cleaner.process('test_doc.txt')

print('=== 处理结果 ===')
print(f'成功: {result.success}')
print(f'质量评分: {result.quality_score}')
print(f'分块数: {len(chunks)}')
print(f'敏感信息: {result.metadata.get("sensitive_info_found", {})}')
print()
print('=== 清洗后的内容 ===')
print(result.content)

# 清理测试文件
import os
os.remove('test_doc.txt')