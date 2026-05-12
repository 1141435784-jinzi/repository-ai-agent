"""
=== 配置管理模块 ===

统一导出所有配置模块，提供便捷的导入接口
"""

from .settings import *
from .database import *
from .security import *

# 配置验证函数
def validate_config() -> None:
    """验证所有配置项"""
    from .settings import validate_settings
    from .database import validate_database_settings
    from .security import validate_security_settings
    
    validate_settings()
    validate_database_settings()
    validate_security_settings()