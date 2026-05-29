"""
=== RAG 增量更新模块 ===

【功能】：
1. 基于 watchdog 文件监听触发增量更新
2. 仅对变化的文件进行增量更新，无需重建整个向量库
3. 支持新增、修改、删除三种操作类型

【设计】：
- watchdog 检测文件变化并通知
- 通过 metadata 中的 source_path 定位需要更新/删除的向量
- 增量更新比全量重建效率提升 10-100 倍（取决于变化文件数量）
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import KNOWLEDGE_BASE_DIR
from src.rag.engine import _split_documents

logger = logging.getLogger(__name__)


class IncrementalUpdater:
    """增量更新器 - 基于 watchdog 文件变化通知的向量库增量更新"""
    
    def __init__(self, vectorstore: Chroma, knowledge_dir: str = None):
        """初始化增量更新器
        
        Args:
            vectorstore: Chroma 向量数据库实例
            knowledge_dir: 知识库目录路径
        """
        self._vectorstore = vectorstore
        self._knowledge_dir = Path(knowledge_dir or KNOWLEDGE_BASE_DIR)
    
    def update_changed_file(self, file_path: str, change_type: str) -> Dict[str, int]:
        """更新单个变化的文件
        
        Args:
            file_path: 变化文件的相对路径
            change_type: 变化类型 ('added', 'modified', 'deleted')
            
        Returns:
            dict: 更新统计 {'added': N, 'deleted': N}
        """
        stats = {'added': 0, 'deleted': 0}
        
        if change_type == 'deleted':
            stats['deleted'] = self._delete_vectors_by_file(file_path)
            logger.info(f"🗑️ 已删除文件 {file_path} 的向量")
        else:
            if change_type == 'modified':
                self._delete_vectors_by_file(file_path)
            stats['added'] = self._add_file_to_vectorstore(file_path)
            logger.info(f"📝 已更新文件 {file_path} 的向量 (新增 {stats['added']} 个)")
        
        self._vectorstore.persist()
        return stats
    
    def update_changed_files(self, changes: Dict[str, List[str]]) -> Dict[str, int]:
        """批量更新变化的文件
        
        Args:
            changes: 变化文件字典 {
                'added': ['file1.md', 'file2.md'],
                'modified': ['file3.md'],
                'deleted': ['file4.md']
            }
            
        Returns:
            dict: 更新统计 {'added': N, 'modified': N, 'deleted': N, 'total_vectors': N}
        """
        stats = {'added': 0, 'modified': 0, 'deleted': 0}
        
        for file_path in changes.get('deleted', []):
            stats['deleted'] += self._delete_vectors_by_file(file_path)
        
        for file_path in changes.get('modified', []):
            self._delete_vectors_by_file(file_path)
            stats['modified'] += self._add_file_to_vectorstore(file_path)
        
        for file_path in changes.get('added', []):
            stats['added'] += self._add_file_to_vectorstore(file_path)
        
        self._vectorstore.persist()
        stats['total_vectors'] = self._vectorstore._collection.count()
        
        logger.info(f"增量更新完成 ✅ 新增:{stats['added']} | 修改:{stats['modified']} | 删除:{stats['deleted']} | 总计:{stats['total_vectors']}")
        return stats
    
    def _delete_vectors_by_file(self, file_path: str) -> int:
        """删除指定文件对应的向量
        
        Args:
            file_path: 要删除的文件相对路径
            
        Returns:
            int: 删除的向量数量
        """
        try:
            result = self._vectorstore.get(
                where={"source_path": file_path},
                include=["metadatas"]
            )
            
            if result["ids"]:
                self._vectorstore.delete(ids=result["ids"])
                logger.debug(f"已删除文件 {file_path} 的 {len(result['ids'])} 个向量")
                return len(result["ids"])
        except Exception as e:
            logger.warning(f"删除文件 {file_path} 的向量失败: {e}")
        
        return 0
    
    def _add_file_to_vectorstore(self, file_path: str) -> int:
        """将指定文件添加到向量库
        
        Args:
            file_path: 要添加的文件相对路径
            
        Returns:
            int: 添加的向量数量
        """
        full_path = self._knowledge_dir / file_path
        
        if not full_path.exists():
            logger.warning(f"文件不存在: {full_path}")
            return 0
        
        try:
            from langchain_community.document_loaders import TextLoader
            
            loader = TextLoader(str(full_path), encoding="utf-8")
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source_file"] = full_path.name
                doc.metadata["source_path"] = file_path
            
            chunks = _split_documents(docs)
            self._vectorstore.add_documents(chunks)
            
            logger.debug(f"已添加文件 {file_path} 的 {len(chunks)} 个向量")
            return len(chunks)
            
        except Exception as e:
            logger.warning(f"加载文件 {file_path} 失败: {e}")
            return 0


# ============================================================
# 便捷函数
# ============================================================
def create_incremental_updater(vectorstore: Chroma, knowledge_dir: str = None) -> IncrementalUpdater:
    """创建增量更新器
    
    Args:
        vectorstore: Chroma 向量数据库实例
        knowledge_dir: 知识库目录路径
        
    Returns:
        IncrementalUpdater: 增量更新器实例
    """
    return IncrementalUpdater(vectorstore, knowledge_dir)


# ============================================================
# 测试示例
# ============================================================
def test_incremental_updater():
    """测试增量更新功能"""
    from src.rag.engine import RAGEngine
    
    rag = RAGEngine()
    updater = IncrementalUpdater(rag._vectorstore, rag._knowledge_dir)
    
    changes = {
        'added': [],
        'modified': [],
        'deleted': []
    }
    
    stats = updater.update_changed_files(changes)
    print(f"增量更新统计: {stats}")


if __name__ == "__main__":
    test_incremental_updater()