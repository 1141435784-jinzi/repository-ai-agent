"""
=== 安全防护函数 ===

第二道防线：输入校验（sanitize_input）
第三道防线：输出过滤（sanitize_output）
"""

import logging
import re

logger = logging.getLogger(__name__)

# ============================================================
# 第二道防线：输入校验（在用户输入进入 Agent 之前）
# ============================================================

# Prompt 注入的常见模式
_INJECTION_PATTERNS: list[re.Pattern] = [
    # 角色覆盖类
    re.compile(r"忽略.{0,10}(之前|上面|以上|所有).{0,10}(指令|提示|规则|设定)", re.IGNORECASE),
    re.compile(r"(你现在是|你不再是|从现在开始你是|扮演|假装你是)", re.IGNORECASE),
    re.compile(r"(切换|进入|启用).{0,5}(模式|角色|身份)", re.IGNORECASE),
    # 信息泄露类
    re.compile(r"(输出|显示|打印|告诉我).{0,10}(系统提示|system prompt|内部指令|初始指令)", re.IGNORECASE),
    re.compile(r"(你的|系统的).{0,5}(提示词|prompt|指令|配置)", re.IGNORECASE),
    re.compile(r"(api.?key|密码|password|secret|token|密钥)", re.IGNORECASE),
    # 越权操作类
    re.compile(r"(执行|运行|调用).{0,10}(命令|脚本|代码|shell|sql|系统)", re.IGNORECASE),
    re.compile(r"(删除|修改|更新).{0,10}(数据库|表|用户|数据)", re.IGNORECASE),
    # 英文注入模式
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"(reveal|show|print|output)\s+(your\s+)?(system\s+)?(prompt|instructions?)", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]

_MAX_INPUT_LENGTH: int = 10000


def sanitize_input(user_input: str) -> tuple[str, bool, str]:
    """输入校验 — 检测并处理潜在的 Prompt 注入攻击

    Args:
        user_input: 用户原始输入文本

    Returns:
        tuple[str, bool, str]:
            - str: 清洗后的输入文本
            - bool: 是否检测到注入风险
            - str: 风险描述
    """
    logger.info(f"=== sanitize_input函数开始 ===")
    logger.info(f"输入参数user_input: {repr(user_input)}")
    logger.info(f"输入参数类型: {type(user_input)}")
    logger.info(f"输入参数长度: {len(user_input) if user_input else 0}")
    
    if not user_input or not user_input.strip():
        logger.info(f"输入为空，返回空字符串")
        return "", False, ""

    cleaned = user_input.strip()
    
    logger.info(f"清理后: {repr(cleaned)}")

    # 长度检查 — 超长输入截断
    if len(cleaned) > _MAX_INPUT_LENGTH:
        logger.warning(
            f"用户输入超长 ({len(cleaned)} 字符)，已截断至 {_MAX_INPUT_LENGTH}"
        )
        cleaned = cleaned[:_MAX_INPUT_LENGTH]

    # 注入模式检测
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            risk_desc = f"检测到可疑注入模式: '{match.group()}'"
            logger.warning(f"⚠️ Prompt 注入风险 — {risk_desc}，原始输入: {cleaned[:100]}...")
            return cleaned, True, risk_desc

    return cleaned, False, ""


# ============================================================
# 第三道防线：输出过滤（在 Agent 回复返回给用户之前）
# ============================================================

# 系统信息泄露检测关键词
_LEAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"postgresql?://\w+:\w+@[\w.]+:\d+/\w+", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"你是一个智能客服助手.*?名叫.*?小智"),
    re.compile(r"安全规范.*?最高优先级.*?任何情况下不可违反"),
]


def sanitize_output(ai_response: str) -> str:
    """输出过滤 — 脱敏处理 + 信息泄露检测

    Args:
        ai_response: LLM 生成的原始回复文本

    Returns:
        str: 过滤和脱敏后的安全回复文本
    """
    if not ai_response:
        return ai_response

    result = ai_response

    # 敏感信息脱敏
    # 手机号：138****1234
    result = re.sub(
        r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)",
        r"\1****\3",
        result,
    )

    # 身份证号：前 6 位 + **** + 后 4 位
    result = re.sub(
        r"(?<!\d)(\d{6})\d{8}(\d{3}[\dXx])(?!\d)",
        r"\1********\2",
        result,
    )

    # 银行卡号：前 4 位 + **** + 后 4 位
    result = re.sub(
        r"(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)",
        r"\1****\2",
        result,
    )

    # 邮箱：u***@domain.com
    def _mask_email(match: re.Match) -> str:
        email = match.group()
        local, domain = email.split("@", 1)
        if len(local) <= 1:
            return f"*@{domain}"
        return f"{local[0]}***@{domain}"

    result = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", _mask_email, result)

    # API Key 脱敏：sk-abc...xyz → sk-***
    result = re.sub(r"sk-[a-zA-Z0-9]{20,}", "sk-***", result)

    # 系统信息泄露检测
    for pattern in _LEAK_PATTERNS:
        if pattern.search(result):
            logger.error(
                f"🚨 输出安全警报 — 检测到疑似信息泄露: '{pattern.pattern}'，"
                f"回复内容: {result[:200]}..."
            )
            return "抱歉，我无法回答这个问题。如需帮助，请联系人工客服。"

    return result