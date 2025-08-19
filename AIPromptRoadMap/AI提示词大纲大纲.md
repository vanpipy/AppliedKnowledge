## 01 - 基础入门
### LLM工作原理
- Tokens (令牌)
- Context Window (上下文窗口)
- 模型参数
- Temperature (温度)
- Top-P (核采样)
- Max Length (最大长度)
- Stop Sequences (停止序列)
- Frequency Penalty (频率惩罚)
- Presence Penalty (存在惩罚)
### 基础提示词设计原则
- 清晰与具体
- 提供上下文
- 平衡简洁与复杂
- 迭代与实验
- 肯定性指令
- 引入目标受众
- 使用分隔符
- 角色扮演
- 少量示例 (Few-Shot Prompting)
- 引导词
- 输出引子
- 避免礼貌用语
- 重复关键词

## 02 - 核心原则
### 提示词范式
- 零样本提示 (Zero-Shot Prompting)
- 少样本提示 (Few-Shot Prompting)
- 多样本提示 (Multi-Shot Prompting)
### 思维链 (Chain-of-Thought, CoT) 提示
- 思维链定义与原理
- 零样本CoT (Zero-Shot CoT)
- 少样本CoT (Few-Shot CoT)
- 自洽性 (Self-Consistency)
- Auto-CoT
### 上下文管理与角色扮演
- 上下文窗口的有效利用
- 系统消息与用户消息
- 角色扮演

## 03 - 高级技巧
### 思维树 (Tree-of-Thought, ToT) 提示
- 定义与原理
- 生成“思想”
- 评估“思想”
- 扩展有前景的思想
- 搜索最佳解决方案
- 优势
### 检索增强生成 (Retrieval-Augmented Generation, RAG)
- 定义与原理
- 朴素RAG (Naive RAG)
- 高级RAG (Advanced RAG)
- 模块化RAG (Modular RAG)
- RAG优势
### 自动化提示词优化 (APO)
- 定义与原理
- 种子提示生成
- 推理评估与反馈
- 数值分数反馈
- LLM反馈
- 人类反馈
- 候选提示词生成
- 启发式编辑
- 通过辅助训练神经网络进行编辑
- 元提示设计
- 基于覆盖的方法
- 程序合成
- 过滤与保留
- 迭代深度
- APO优势
### 自我修正与提示词链
- 自我修正 (Self-Refinement) 过程
- 自我修正优势
- 提示词链 (Prompt Chaining)
- 提示词链优势
### 其他高级技巧
- Maieutic Prompting (产婆式提示)
- Meta-prompting (元提示)
- Output Primers (输出引子)
- Dynamic Prompting (动态提示词)
- Prompt Weighting (提示词加权)
- Multimodal Prompting (多模态提示词)

## 04 - 特定领域应用
### 文本生成
- 摘要
- 翻译
- 创意写作
### 代码生成与调试
- 代码生成
- 先泛后精
- 提供示例
- 分解复杂任务
- 避免歧义
- 指明相关代码
- 迭代与实验
- 保持历史相关性
- 遵循良好编码实践
- 代码调试
### 图像生成与编辑
- 文本到图像生成
- 从简单到复杂
- 指令明确
- 提供示例
- 提示词结构
- 动态提示词与加权
- 指定长宽比
- 图像编辑
- 上下文特异性
- 任务导向
- 处理拒绝
- 图片置于文本前
- 先描述图片再任务
- 定义输出格式

## 05 - 评估与优化
### 评估
- 评估指标
- 评估方法
- 评估框架与工具
### 优化
- 迭代优化
- A/B测试
- 提示词版本控制