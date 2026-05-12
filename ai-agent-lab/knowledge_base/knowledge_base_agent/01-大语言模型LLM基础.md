# 0 - 大语言模型 LLM 基础

## 学习目标
- 深入理解 Transformer 架构的完整工作原理（Pre-Norm + RMSNorm + SwiGLU 当前主流架构）
- 掌握 LLM 训练三阶段和关键推理参数
- 了解主流 LLM 的能力对比和选型策略
- 理解 Token 计费模型，能做企业成本估算

> 难度：⭐⭐⭐⭐⭐ | 面试热度：🔥🔥🔥🔥（高频，尤其是 Transformer 原理）

---

## 一、Transformer 架构完整讲解

### 1.1 整体结构

Transformer 由 Encoder（编码器）和 Decoder（解码器）组成，但不同模型使用不同部分：

| 模型 | 使用部分 | 代表 | 适用任务 |
|------|----------|------|----------|
| GPT 系列 | 仅 Decoder | GPT-4o、LLaMA、Qwen | 文本生成、对话 |
| BERT 系列 | 仅 Encoder | BERT、RoBERTa | 文本理解、分类 |
| T5 系列 | Encoder + Decoder | T5、BART | 翻译、摘要 |

**当前 Agent 开发主要使用 Decoder-Only 架构（GPT 系列）。**

### 1.2 完整流程图（Pre-Norm 架构，当前主流）

以翻译 "我热爱编程，它让我快乐" → "I love programming, it makes me happy" 为例：

```
输入: "我热爱编程，它让我快乐"

  ┌─────────────────────────────────────────────────────┐
  │                    ENCODER                           │
  │                                                      │
  │  [Tokenization] → [我][热爱][编程][，][它][让][我][快乐]│
  │         ↓                                            │
  │  [Embedding + 位置编码] → 每个 token 变成向量          │
  │         ↓                                            │
  │  ┌─── Encoder Block × N ───┐                        │
  │  │  RMSNorm → Self-Attention → Add(残差)             │
  │  │  RMSNorm → FFN(SwiGLU)  → Add(残差)              │
  │  └─────────────────────────┘                        │
  │         ↓                                            │
  │  编码器输出（上下文表示）                               │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │                    DECODER                           │
  │                                                      │
  │  [已生成: "I love"] → Embedding + 位置编码             │
  │         ↓                                            │
  │  ┌─── Decoder Block × N ───┐                        │
  │  │  RMSNorm → Masked Self-Attention → Add(残差)      │
  │  │  RMSNorm → Cross-Attention(看Encoder) → Add(残差) │
  │  │  RMSNorm → FFN(SwiGLU) → Add(残差)               │
  │  └─────────────────────────┘                        │
  │         ↓                                            │
  │  [Linear → Softmax] → 预测下一个词: "programming"     │
  └─────────────────────────────────────────────────────┘
```

### 1.3 七大关键组件详细拆解

#### 组件 1：Tokenization（分词）

将文本切分为模型能处理的最小单元（Token）。

```
"我热爱编程" → ["我", "热爱", "编程"]          # 中文：基本按字/词
"I love programming" → ["I", " love", " program", "ming"]  # 英文：BPE 子词
```

主流分词算法：BPE（Byte Pair Encoding），GPT/LLaMA/Qwen 都使用。

💡 **场景举例**：为什么 Token 数和字数不一样？因为分词器会把常见词保持完整，罕见词拆成子词。"programming" 可能被拆成 "program" + "ming"。

#### 组件 2：Embedding（词嵌入）+ 位置编码

**词嵌入**：将每个 Token 映射为一个高维向量（如 4096 维），语义相近的词向量距离近。

**位置编码**：Transformer 没有 RNN 的顺序处理能力，需要额外注入位置信息。

原始论文使用正弦/余弦位置编码：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

当前主流使用 **RoPE（旋转位置编码）**，LLaMA、Qwen、GPT-4 都采用，支持更长的上下文窗口。

#### 组件 3：Self-Attention（自注意力）

**核心思想**：让每个词"看到"句子中的所有其他词，理解词与词之间的关系。

**Q/K/V 矩阵**：
- Q（Query，查询）：我在找什么？
- K（Key，键）：我能提供什么？
- V（Value，值）：我的实际内容是什么

**计算公式**：
```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

**数值示例**（简化为 2 维）：

假设句子 "猫 坐 在 垫子 上"，d_k = 2

```
Q = [[1, 0],    K = [[1, 1],    V = [[1, 0],
     [0, 1],         [0, 1],         [0, 1],
     [1, 1],         [1, 0],         [1, 1],
     [0, 0],         [0, 0],         [0, 0],
     [1, 0]]         [1, 0]]         [1, 0]]

Step 1: Q · K^T（计算注意力分数）
Step 2: / √d_k = / √2 ≈ / 1.414（缩放，防止梯度消失）
Step 3: softmax（归一化为概率分布）
Step 4: × V（加权求和，得到上下文感知的表示）
```

💡 **场景举例**：在 "小明把苹果给了小红，她很开心" 中，Self-Attention 让模型理解"她"指的是"小红"而不是"苹果"。

#### 组件 4：Multi-Head Attention（多头注意力）

**核心思想**：一个注意力头只能关注一种关系模式。多个头并行，每个头关注不同的关系。

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) · W_O

其中 head_i = Attention(Q · W_Q_i, K · W_K_i, V · W_V_i)
```

- 典型配置：d_model=4096, h=32 heads, d_k=d_v=128（每个头的维度）
- 多头拆分：将 4096 维拆成 32 个 128 维的子空间
- 拼接后通过 W_O 映射回 4096 维

💡 **场景举例**：
- Head 1 可能关注语法关系（主语-谓语）
- Head 2 可能关注指代关系（代词-实体）
- Head 3 可能关注位置关系（相邻词）

#### 组件 5：Norm & Add（Pre-Norm 架构）

**LayerNorm 公式**：
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

其中 μ = mean(x), σ² = var(x)
```

数值示例：x = [1, 2, 3, 4]
```
μ = 2.5, σ² = 1.25
LayerNorm = γ · ([1,2,3,4] - 2.5) / √1.25 + β
         = γ · [-1.34, -0.45, 0.45, 1.34] + β
```

**LayerNorm vs BatchNorm**：

| 维度 | LayerNorm | BatchNorm |
|------|-----------|-----------|
| 归一化方向 | 单个样本的所有特征 | 一个 batch 的同一特征 |
| 依赖 batch size | 否 | 是 |
| 适用场景 | NLP（序列长度可变） | CV（固定尺寸图像） |
| 推理时行为 | 与训练一致 | 需要维护 running mean/var |

**Pre-Norm vs Post-Norm**：

```
Pre-Norm（当前主流，LLaMA/Qwen/GPT-4）:
  output = x + SubLayer(RMSNorm(x))
  → 先归一化再计算，梯度直通，训练更稳定

Post-Norm（原始 Transformer 论文）:
  output = LayerNorm(x + SubLayer(x))
  → 先计算再归一化，深层网络训练不稳定
```

Pre-Norm 的优势：归一化在子层之前，残差连接直接连到输出，梯度可以无损地流过残差路径。

**RMSNorm（Root Mean Square Normalization）**：

```
RMSNorm(x) = γ · x / √(mean(x²) + ε)
```

相比 LayerNorm，RMSNorm 去掉了均值中心化（减去 μ）和偏置 β，计算更快，效果相当。LLaMA、Qwen、Gemma 等主流模型都使用 RMSNorm。

**残差连接（Residual Connection）**：

```
output = x + SubLayer(x)
```

**为什么需要残差连接？** 梯度高速公路原理：

```
∂(x + F(x))/∂x = 1 + ∂F(x)/∂x
```

即使 `∂F(x)/∂x` 很小（梯度消失），梯度至少为 1，保证信息能流过深层网络。

**重要**：每个子层（Attention、FFN）都有独立的残差连接，不是跳过所有子层。



#### 组件 6：FFN（SwiGLU 激活）

**原始 ReLU FFN**（原始论文）：
```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
隐藏层维度：4d（d_model 的 4 倍）
```

**当前主流 SwiGLU FFN**（LLaMA/Qwen/Gemma）：
```
FFN(x) = (Swish(x · W₁) ⊙ (x · W₃)) · W₂
隐藏层维度：8d/3（约 2.67d）
```

其中：
- **SwiGLU = Swish + GLU（Gated Linear Unit）**
- Swish(x) = x · σ(x)，σ 是 sigmoid 函数
- ⊙ 是逐元素乘法（门控机制）
- 3 个权重矩阵：W₁（gate）、W₂（down）、W₃（up）
- 隐藏层维度从 4d 变为 8d/3，总参数量基本不变

**激活函数演进**：
```
ReLU → GELU → SwiGLU
简单高效   更平滑   门控机制，效果最好
```

**FFN 充当知识存储**：研究发现，Transformer 的大部分"知识"存储在 FFN 的权重矩阵中。Self-Attention 负责"理解关系"，FFN 负责"存储知识"。

#### 组件 7：Mask 与 Cross-Attention

**Decoder 因果掩码（Causal Mask）**：

在生成文本时，每个位置只能看到它之前的位置，不能"偷看"未来。

```
Mask 矩阵（4个位置）：
     pos1  pos2  pos3  pos4
pos1 [  0   -∞   -∞   -∞  ]
pos2 [  0    0   -∞   -∞  ]
pos3 [  0    0    0   -∞  ]
pos4 [  0    0    0    0  ]

0 = 可以看到，-∞ = 看不到（softmax 后变为 0）
```

**Cross-Attention（Encoder-Decoder 桥梁）**：

在 Encoder-Decoder 架构中，Decoder 通过 Cross-Attention 查看 Encoder 的输出：
- Q 来自 Decoder（我在找什么翻译？）
- K, V 来自 Encoder（源语言的上下文表示）

### 1.4 Encoder 全流程（8 步）

以 "我热爱编程" 为例，d_model=4（简化）：

```
Step 1: Tokenization
  "我热爱编程" → ["我", "热爱", "编程"]  → token_ids = [156, 2847, 1523]

Step 2: Token Embedding
  [156, 2847, 1523] → [[0.2, 0.5, -0.1, 0.8],   # "我"
                        [0.7, -0.3, 0.4, 0.1],    # "热爱"
                        [0.1, 0.6, 0.9, -0.2]]    # "编程"

Step 3: 加位置编码
  X = Token_Embedding + Position_Encoding

Step 4: RMSNorm（Pre-Norm）
  X_norm = RMSNorm(X)

Step 5: Multi-Head Self-Attention
  Q = X_norm · W_Q,  K = X_norm · W_K,  V = X_norm · W_V
  Attn = softmax(Q·K^T / √d_k) · V

Step 6: 残差连接
  X = X + Attn    # 注意：是原始 X，不是 X_norm

Step 7: RMSNorm → SwiGLU FFN
  X_norm2 = RMSNorm(X)
  FFN_out = (Swish(X_norm2 · W₁) ⊙ (X_norm2 · W₃)) · W₂

Step 8: 残差连接
  X = X + FFN_out

→ 重复 Step 4-8 共 N 层（如 32 层）
→ 输出：每个 token 的上下文感知表示
```

### 1.5 Decoder 全流程（9 步）

以已生成 "I love" 预测下一个词为例：

```
Step 1: Tokenization
  "I love" → ["I", " love"] → token_ids = [40, 2141]

Step 2: Token Embedding + 位置编码

Step 3: RMSNorm

Step 4: Masked Self-Attention（因果掩码）
  "I" 只能看到 "I"
  "love" 能看到 "I" 和 "love"

Step 5: 残差连接

Step 6: RMSNorm → Cross-Attention（看 Encoder 输出）
  Q 来自 Decoder，K/V 来自 Encoder
  让 "love" 关注源语言中的 "热爱"

Step 7: 残差连接

Step 8: RMSNorm → SwiGLU FFN → 残差连接

Step 9: Linear → Softmax
  将最后一个位置的向量映射到词表大小
  → P("programming") = 0.85, P("coding") = 0.10, ...
  → 选择 "programming"

→ 重复，直到生成 <EOS> 结束符
```

### 1.6 关键设计总结

| 组件 | 原始论文 | 当前主流 | 代表模型 |
|------|----------|----------|----------|
| 归一化 | Post-Norm + LayerNorm | Pre-Norm + RMSNorm | LLaMA, Qwen, Gemma |
| 激活函数 | ReLU | SwiGLU | LLaMA, Qwen, PaLM |
| FFN 隐藏维度 | 4d | 8d/3 | LLaMA, Qwen |
| 位置编码 | 正弦/余弦 | RoPE | LLaMA, Qwen, GPT-4 |
| 注意力 | MHA | GQA/MQA | LLaMA 2+, Qwen |

---

## 二、LLM 训练三阶段

```
Stage 1: 预训练（Pre-training）
  海量文本 → 学习语言规律 → 基座模型
  数据量：万亿 Token
  目标：Next Token Prediction

Stage 2: 监督微调（SFT - Supervised Fine-Tuning）
  高质量指令数据 → 学会遵循指令 → Chat 模型
  数据量：数万~数十万条
  目标：学会按指令格式回答

Stage 3: 人类偏好对齐（RLHF / DPO）
  人类偏好数据 → 输出更符合人类期望
  RLHF：训练奖励模型 + PPO 强化学习
  DPO：直接偏好优化，更简单高效（当前主流）
```

💡 **场景举例**：
- 预训练后的模型像一个读了所有书的学者——知识渊博但不会聊天
- SFT 后像经过培训的客服——知道怎么回答问题
- RLHF/DPO 后像经过考核的金牌客服——回答更专业、更安全、更符合用户期望

---

## 三、关键推理参数

### 3.1 Temperature（温度）

控制输出的随机性。

```
P(token) = softmax(logits / temperature)

temperature = 0：确定性输出，总是选概率最高的（贪心）
temperature = 0.7：适度随机，平衡创造性和准确性
temperature = 1.5：高度随机，创造性强但可能胡说
```

**企业建议**：
- 客服/风控等需要准确性的场景：temperature = 0
- 创意写作/头脑风暴：temperature = 0.7-1.0

### 3.2 Top-P（核采样）

只从累积概率达到 P 的最小 token 集合中采样。

**详细原理**：

假设下一个词的概率分布：
```
"编程" = 0.40
"代码" = 0.25
"技术" = 0.15
"学习" = 0.10
"工作" = 0.05
"吃饭" = 0.03
"睡觉" = 0.02
```

Top-P = 0.8 时：
```
累积概率：0.40 → 0.65 → 0.80 ✓ 停止
候选集：{"编程", "代码", "技术"}
从这 3 个词中按概率采样
```

**Top-P vs Top-K**：

| 方式 | 机制 | 优点 | 缺点 |
|------|------|------|------|
| Top-K | 固定选前 K 个 | 简单 | 概率分布不均时不灵活 |
| Top-P | 动态选累积概率达 P 的 | 自适应候选集大小 | 略复杂 |

### 3.3 Max Tokens 与 Context Window

- **Max Tokens**：限制单次生成的最大 Token 数
- **Context Window**：模型能处理的最大上下文长度

```
Context Window = Input Tokens + Output Tokens

GPT-4o:     128K context
Claude 3.5: 200K context
Qwen2.5:    128K context
```

---

## 四、主流 LLM 对比（2026 年版本）

### 闭源模型

| 模型 | 厂商 | 上下文 | 特点 |
|------|------|--------|------|
| GPT-4o | OpenAI | 128K | 多模态，综合能力强 |
| GPT-o3 | OpenAI | 200K | 推理能力极强（思维链） |
| Claude 3.5 Sonnet | Anthropic | 200K | 长文本理解，代码能力强 |
| Claude Opus 4 | Anthropic | 200K | 最强综合能力 |
| Gemini 2.0 | Google | 1M+ | 超长上下文，多模态 |

### 开源模型

| 模型 | 厂商 | 参数量 | 特点 |
|------|------|--------|------|
| Qwen2.5/3 | 阿里 | 0.5B-72B | 中文最强开源，工具调用好 |
| LLaMA 3/4 | Meta | 8B-405B | 英文生态最好 |
| DeepSeek-V3 | DeepSeek | 671B(MoE) | 性价比极高 |
| DeepSeek-R1 | DeepSeek | 671B(MoE) | 推理能力强 |
| GLM-4 | 智谱 | 9B-130B | 中英双语 |
| Mistral Large | Mistral | 123B | 欧洲开源代表 |

---

## 五、Token 与计费

### 计费公式

```
费用 = (输入 Token 数 × 输入单价) + (输出 Token 数 × 输出单价)
```

### 企业成本估算场景

💡 **场景举例**：一个智能客服系统，日均处理 5000 次对话

```
假设每次对话：
- 输入：~800 tokens（系统提示 + 历史消息 + 用户问题）
- 输出：~200 tokens（Agent 回答）

使用 GPT-4o（$2.5/1M input, $10/1M output）：
- 日输入费用：5000 × 800 / 1M × $2.5 = $10
- 日输出费用：5000 × 200 / 1M × $10 = $10
- 日总费用：$20
- 月总费用：~$600

使用 GPT-4o-mini（$0.15/1M input, $0.6/1M output）：
- 月总费用：~$36（便宜 16 倍）
```

---

## 六、LLM 选型决策树

```
数据能否出境？
├── 否 → 国产模型
│   ├── 需要最强能力 → Qwen-Max / DeepSeek-V3（API）
│   ├── 需要私有部署 → Qwen2.5-72B / GLM-4-9B
│   └── 成本敏感 → Qwen2.5-7B / DeepSeek-V3（API 极便宜）
└── 是 → 国际模型
    ├── 需要最强能力 → GPT-4o / Claude Opus 4
    ├── 需要长上下文 → Gemini 2.0（1M+）/ Claude 3.5（200K）
    ├── 成本敏感 → GPT-4o-mini / DeepSeek-V3
    └── 需要私有部署 → LLaMA 3-70B / Qwen2.5-72B
```

---

## 七、企业级 LLM 调用封装

```python
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LLMClient:
    """企业级 LLM 调用封装：统一接口、自动重试、Token 计数"""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0,
        max_tokens: int = 4096,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.total_tokens = 0
        self.total_cost = 0.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def invoke(self, messages: list) -> str:
        """调用 LLM，带重试和 Token 统计"""
        response = self.llm.invoke(messages)

        # Token 统计
        usage = response.usage_metadata or {}
        tokens = usage.get("total_tokens", 0)
        self.total_tokens += tokens
        logger.info(f"LLM 调用完成 | tokens={tokens} | total={self.total_tokens}")

        return response.content

    async def ainvoke(self, messages: list) -> str:
        """异步调用"""
        response = await self.llm.ainvoke(messages)
        return response.content

# 使用示例
client = LLMClient(model="gpt-4o", temperature=0)
result = client.invoke([
    SystemMessage(content="你是一个专业的数据分析师"),
    HumanMessage(content="分析一下上个月的销售趋势"),
])
```

---

## 八、LLM 的幻觉问题（Hallucination）

### 8.1 什么是幻觉？

LLM 会"一本正经地胡说八道"——生成看起来合理但实际上错误或虚构的内容。

**幻觉的三种类型**：

| 类型 | 说明 | 示例 |
|------|------|------|
| 事实性幻觉 | 编造不存在的事实 | "爱因斯坦在1920年获得了图灵奖" |
| 忠实性幻觉 | 回答与提供的上下文矛盾 | 文档说产品价格599，LLM回答799 |
| 指令性幻觉 | 不遵循用户指令的格式要求 | 要求JSON格式，输出了纯文本 |

### 8.2 幻觉产生的根本原因

```
1. 训练数据中的噪声和错误 → 模型学到了错误的"知识"
2. 概率生成机制 → 模型选择"最可能的下一个词"，不是"最正确的"
3. 知识截止日期 → 不知道训练数据之后发生的事
4. 过度自信 → 模型不会说"我不知道"，总是尝试给出答案
```

### 8.3 企业级幻觉缓解策略

```
策略 1：RAG（检索增强生成）
  → 让 LLM 基于检索到的真实文档回答，而不是凭记忆

策略 2：Prompt 约束
  → "如果你不确定，请明确说明你不知道，不要编造"

策略 3：答案溯源
  → 要求 LLM 标注回答的来源，方便人工核实

策略 4：Temperature = 0
  → 降低随机性，减少"创造性编造"

策略 5：多模型交叉验证
  → 用多个模型回答同一问题，对比结果
```

💡 **场景举例**：金融风控系统中，如果 Agent 幻觉说"该客户信用评分为 A 级"，可能导致错误放贷。所以金融场景必须用 RAG + 答案溯源 + 人工审核。

---

## 九、Embedding 与向量表示

### 9.1 什么是 Embedding？

Embedding 是将文本（词、句子、段落）映射为固定维度的数值向量，使得语义相近的文本在向量空间中距离更近。

```
"国王" → [0.2, 0.8, -0.1, 0.5, ...]   (768维或1024维)
"皇帝" → [0.3, 0.7, -0.2, 0.6, ...]   ← 与"国王"距离近
"苹果" → [-0.5, 0.1, 0.9, -0.3, ...]   ← 与"国王"距离远
```

### 9.2 Embedding 的核心用途

| 用途 | 说明 | 企业场景 |
|------|------|----------|
| 语义检索 | 根据语义相似度搜索文档 | RAG 知识库检索 |
| 文本分类 | 将文本向量输入分类器 | 工单自动分类 |
| 聚类分析 | 对相似文本自动分组 | 客户反馈归类 |
| 相似度计算 | 计算两段文本的语义相似度 | 重复问题检测 |

### 9.3 相似度计算方法

```
余弦相似度（最常用）：
  cos(A, B) = (A · B) / (||A|| × ||B||)
  范围：[-1, 1]，1 表示完全相同，0 表示无关，-1 表示完全相反

欧氏距离：
  d(A, B) = √(Σ(Ai - Bi)²)
  值越小越相似

点积（内积）：
  dot(A, B) = Σ(Ai × Bi)
  值越大越相似（需要向量归一化后才等价于余弦相似度）
```

### 9.4 主流 Embedding 模型

| 模型 | 厂商 | 维度 | 特点 |
|------|------|------|------|
| text-embedding-3-small | OpenAI | 1536 | 性价比高，英文效果好 |
| text-embedding-3-large | OpenAI | 3072 | 效果最好，成本较高 |
| bge-large-zh-v1.5 | BAAI(智源) | 1024 | 中文效果好，可本地部署 |
| bge-m3 | BAAI(智源) | 1024 | 多语言，支持稠密+稀疏+多向量 |
| paraphrase-multilingual-MiniLM-L12-v2 | Sentence-Transformers | 384 | 轻量多语言，适合快速原型 |

💡 **场景举例**：用户问"怎么退货"，Embedding 模型会把这句话转成向量，然后在知识库中找到"退换货政策"相关的段落——即使知识库里写的是"退换货流程"而不是"怎么退货"，语义检索也能匹配到。

---

## 十、LLM 的 API 调用模式

### 10.1 同步调用 vs 异步调用 vs 流式输出

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="glm-4-flash", temperature=0.7)

# 1. 同步调用 — 等待完整响应后返回
response = llm.invoke("你好")
print(response.content)

# 2. 异步调用 — 适合高并发场景（Web 服务）
import asyncio
async def async_call():
    response = await llm.ainvoke("你好")
    return response.content

# 3. 流式输出 — 逐 Token 返回，用户体验好
for chunk in llm.stream("给我讲个故事"):
    print(chunk.content, end="", flush=True)
```

### 10.2 Function Calling（函数调用）

```
LLM 不直接执行代码，而是告诉你"我想调用哪个函数、传什么参数"

用户: "北京今天天气怎么样？"

LLM 返回（不是文本，而是结构化的工具调用请求）:
{
    "tool_calls": [{
        "function": "get_weather",
        "arguments": {"city": "北京"}
    }]
}

你的代码执行 get_weather("北京") → "晴，25°C"

把结果返回给 LLM → LLM 生成最终回复:
"北京今天天气晴朗，气温25°C，适合出行。"
```

这就是 Agent 能"使用工具"的底层原理。

---

## 面试考点

**Q1：简述 Transformer 的核心组件和工作原理**

答题要点：
- 核心组件：Embedding + 位置编码、Multi-Head Attention、FFN、Norm & Add
- 当前主流架构：Pre-Norm + RMSNorm + SwiGLU + RoPE
- Self-Attention 让每个词看到所有其他词，理解上下文关系
- FFN 充当知识存储，Attention 负责关系理解

**Q2：Pre-Norm 和 Post-Norm 的区别？为什么当前主流用 Pre-Norm？**

答题要点：
- Pre-Norm：`output = x + SubLayer(RMSNorm(x))`，先归一化再计算
- Post-Norm：`output = LayerNorm(x + SubLayer(x))`，先计算再归一化
- Pre-Norm 优势：梯度直通（残差路径不经过归一化），深层网络训练更稳定

**Q3：Temperature 和 Top-P 分别控制什么？企业场景怎么设置？**

答题要点：
- Temperature 控制随机性（0=确定性，1=高随机）
- Top-P 控制候选词范围（核采样）
- 企业场景：需要准确性的（客服、风控）用 temperature=0；需要创造性的用 0.7-1.0

**Q4：什么是 LLM 幻觉？企业中怎么解决？**

答题要点：
- 幻觉 = LLM 生成看似合理但实际错误/虚构的内容
- 三种类型：事实性幻觉、忠实性幻觉、指令性幻觉
- 解决方案：RAG 检索增强 + Prompt 约束 + 答案溯源 + Temperature=0 + 人工审核

**Q5：Embedding 是什么？在 Agent 开发中有什么用？**

答题要点：
- Embedding 将文本映射为数值向量，语义相近的文本向量距离近
- 核心用途：RAG 中的语义检索（把用户问题和知识库文档都转成向量，计算相似度）
- 相似度计算：余弦相似度最常用，范围 [-1, 1]
- 模型选择：中文推荐 bge-large-zh-v1.5，多语言推荐 bge-m3

**Q6：同步调用、异步调用、流式输出分别适用什么场景？**

答题要点：
- 同步调用：简单脚本、批处理任务
- 异步调用：Web 服务、高并发场景（FastAPI + async/await）
- 流式输出：面向用户的对话界面，逐字显示提升体验
