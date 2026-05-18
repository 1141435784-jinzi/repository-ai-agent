"""
=== 文件变化监听服务 ===

【功能】：
1. 监听知识库目录的文件变化
2. 自动触发 RAG 增量更新
3. 支持防抖机制，避免频繁更新

【设计】：
- 使用 watchdog 库实现高效的文件系统监听
- 防抖机制：文件变化后等待一段时间再执行更新，避免频繁触发
- 支持启动/停止控制
- 同步执行，不阻塞主线程
"""

import logging
import time
import threading
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from watchdog.events import (
    FileSystemEventHandler,
    FileSystemEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
)
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# 类型检查时的导入（避免循环引用）
if TYPE_CHECKING:
    from src.rag.incremental_update import IncrementalUpdater


class KnowledgeBaseWatcher:
    """知识库文件监听服务"""

    def __init__(self, knowledge_dir: str, callback: Callable, debounce_delay: float = 3.0):
        """初始化文件监听器

        Args:
            knowledge_dir: 要监听的知识库目录
            callback: 文件变化时的回调函数，签名为 callback(file_path: str, change_type: str)
            debounce_delay: 防抖延迟（秒），默认 3 秒
        """
        self._knowledge_dir = Path(knowledge_dir)
        self._callback = callback
        self._debounce_delay = debounce_delay
        self._observer = None
        self._event_handler = None
        self._last_event_time = 0
        self._debounce_task = None
        self._is_running = False
        self._pending_events: list[tuple] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """启动文件监听服务"""
        if self._is_running:
            logger.warning("文件监听服务已在运行中")
            return

        if not self._knowledge_dir.exists():
            logger.error(f"知识库目录不存在: {self._knowledge_dir}")
            return

        self._event_handler = _KnowledgeBaseEventHandler(self._on_file_change)

        self._observer = Observer()
        self._observer.schedule(
            self._event_handler,
            str(self._knowledge_dir),
            recursive=True,
        )

        self._observer.start()
        self._is_running = True

    def stop(self) -> None:
        """停止文件监听服务"""
        if not self._is_running:
            return

        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        if self._debounce_task and self._debounce_task.is_alive():
            # 等待防抖任务完成
            try:
                self._debounce_task.join(timeout=1.0)
            except TimeoutError:
                pass

        self._is_running = False
        logger.info("✅ 文件监听服务已停止")

    def _on_file_change(self, event: FileSystemEvent) -> None:
        """处理文件变化事件"""
        if not event.src_path.endswith(".md"):
            return

        if event.is_directory:
            return

        current_time = time.time()

        with self._lock:
            if current_time - self._last_event_time < self._debounce_delay:
                if self._debounce_task and self._debounce_task.is_alive():
                    self._debounce_task.join(timeout=0.1)

            self._last_event_time = current_time

            file_path = str(Path(event.src_path).relative_to(self._knowledge_dir))
            change_type = self._get_change_type(event)

            self._pending_events.append((file_path, change_type))

            self._debounce_task = threading.Thread(
                target=self._debounced_update,
                args=(file_path, change_type),
                daemon=True,
            )
            self._debounce_task.start()

    def _get_change_type(self, event: FileSystemEvent) -> str:
        """获取变化类型"""
        if isinstance(event, FileCreatedEvent):
            return "added"
        elif isinstance(event, FileDeletedEvent):
            return "deleted"
        elif isinstance(event, FileMovedEvent):
            return "modified"
        elif isinstance(event, FileModifiedEvent):
            return "modified"
        return "modified"

    def _debounced_update(self, file_path: str, change_type: str) -> None:
        """防抖后的更新任务"""
        time.sleep(self._debounce_delay)

        with self._lock:
            if not self._pending_events:
                return

            latest_event = self._pending_events[-1]
            file_path, change_type = latest_event
            self._pending_events.clear()

        try:
            logger.info(f"📁 检测到文件变化: {change_type} - {file_path}")

            if self._callback:
                self._callback(file_path, change_type)
                logger.info(f"🔄 增量更新已执行")

        except Exception as e:
            logger.error(f"增量更新失败: {e}")

    def is_running(self) -> bool:
        """检查服务是否正在运行"""
        return self._is_running


class _KnowledgeBaseEventHandler(FileSystemEventHandler):
    """内部文件系统事件处理器"""

    def __init__(self, callback: Callable):
        self._callback = callback

    def on_created(self, event: FileSystemEvent):
        self._callback(event)

    def on_modified(self, event: FileSystemEvent):
        self._callback(event)

    def on_deleted(self, event: FileSystemEvent):
        self._callback(event)

    def on_moved(self, event: FileSystemEvent):
        self._callback(event)


# ============================================================
# 全局文件监听器管理器
# ============================================================
_global_file_watchers: list[KnowledgeBaseWatcher] = []
_global_updaters: dict[str, "IncrementalUpdater"] = {}


def start_all_file_watchers() -> None:
    """启动所有知识库目录的文件监听服务

    扫描所有配置的知识库目录，为每个目录启动文件监听器。
    当知识库文件发生变化时，自动触发增量更新。

    【使用场景】：
    - 在 FastAPI 服务启动时调用
    - 确保知识库更新时向量库自动同步
    """
    from src.config import KNOWLEDGE_BASES

    global _global_file_watchers

    for name, kb_dir in KNOWLEDGE_BASES.items():
        if kb_dir and kb_dir != "NONE":
            try:
                if Path(kb_dir).exists():
                    watcher = create_knowledge_base_watcher(
                        knowledge_dir=kb_dir,
                        callback=_create_update_callback(name),
                        debounce_delay=3.0,
                    )
                    watcher.start()
                    _global_file_watchers.append(watcher)
                    logger.info(f"✅ 文件监听已启动: {name} ({kb_dir})")
                else:
                    logger.warning(f"⚠️ 知识库目录不存在，跳过监听: {name} ({kb_dir})")
            except Exception as e:
                logger.error(f"❌ 文件监听启动失败: {name} - {e}")


def _create_update_callback(knowledge_base_name: str):
    """为指定知识库创建增量更新回调函数"""

    def update_callback(file_path: str, change_type: str):
        try:
            from src.rag.incremental_update import IncrementalUpdater, create_incremental_updater

            if knowledge_base_name not in _global_updaters:
                from src.agents import get_expert_rag_engine

                rag_engine = get_expert_rag_engine(knowledge_base_name)
                if rag_engine:
                    _global_updaters[knowledge_base_name] = create_incremental_updater(
                        rag_engine._vectorstore,
                        rag_engine._knowledge_dir,
                    )

            updater = _global_updaters.get(knowledge_base_name)
            if updater:
                stats = updater.update_changed_file(file_path, change_type)
                logger.info(
                    f"📚 [{knowledge_base_name}] {change_type}: {file_path} → 新增={stats['added']} | 删除={stats['deleted']}"
                )
                return stats
        except Exception as e:
            logger.error(f"❌ [{knowledge_base_name}] 增量更新失败: {e}")
        return None

    return update_callback


def stop_all_file_watchers() -> None:
    """停止所有文件监听服务

    【使用场景】：
    - 在 FastAPI 服务关闭时调用
    - 确保所有文件监听器正确释放资源
    """
    global _global_file_watchers

    for watcher in _global_file_watchers:
        try:
            watcher.stop()
            logger.info(f"✅ 文件监听已停止")
        except Exception as e:
            logger.error(f"❌ 停止文件监听失败: {e}")

    _global_file_watchers.clear()


# ============================================================
# 便捷函数
# ============================================================
def create_knowledge_base_watcher(
    knowledge_dir: str, callback: Callable, debounce_delay: float = 3.0
) -> KnowledgeBaseWatcher:
    """创建知识库文件监听器

    Args:
        knowledge_dir: 知识库目录路径
        callback: 回调函数，签名为 callback(file_path: str, change_type: str)
        debounce_delay: 防抖延迟（秒）

    Returns:
        KnowledgeBaseWatcher: 文件监听器实例
    """
    return KnowledgeBaseWatcher(knowledge_dir, callback, debounce_delay)


# ============================================================
# 测试示例
# ============================================================
def test_file_watcher():
    """测试文件监听服务"""
    from src.config import KNOWLEDGE_BASE_DIR

    def on_change(file_path: str, change_type: str):
        print(f"检测到文件变化: {change_type} - {file_path}")

    watcher = create_knowledge_base_watcher(KNOWLEDGE_BASE_DIR, on_change)

    watcher.start()

    try:
        print("文件监听器已启动，按 Ctrl+C 退出...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("收到退出信号")
    finally:
        watcher.stop()


if __name__ == "__main__":
    test_file_watcher()