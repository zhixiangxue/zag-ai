# LlamaIndex Chat Engine 和 Agent 模块深度分析

> 本文档详细分析 LlamaIndex 中 `chat_engine` 和 `agent` 两个模块的设计与实现细节。

---

## 一、Chat Engine 模块分析

### 1.1 模块职责概述

Chat Engine 的核心职责是**对话编排**，解决"如何在多轮对话中使用 RAG"的问题。

**主要功能：**
1. 对话历史管理（通过 Memory）
2. 问题改写（Question Condensation）
3. RAG 流程编排（Retriever + LLM + Memory）
4. 流式响应处理
5. 上下文注入

### 1.2 实现类型详解

#### 1.2.1 SimpleChatEngine

**定位**：最基础的聊天引擎，不涉及检索，纯 LLM 对话。

*核心流程：**
```python
def chat(self, message: str, chat_history: Optional[List[ChatMessage]] = None):
    # 1. 将用户消息放入 memory
    self._memory.put(ChatMessage(content=message, role="user"))
    
    # 2. 组合 prefix_messages（system prompt）+ memory 中的历史
    all_messages = self._prefix_messages + self._memory.get(...)
    
    # 3. 直接调用 LLM
    chat_response = self._llm.chat(all_messages)
    
    # 4. 将 AI 响应也放入 memory
    self._memory.put(chat_response.message)
    
    return AgentChatResponse(response=str(chat_response.message.content))
```

**关键点：**
- 纯粹的对话历史管理 + LLM 调用
- `prefix_messages` 用于注入 system prompt
- Memory 自动处理 token 限制

#### 1.2.2 ContextChatEngine

**定位**：在每轮对话中先检索，将检索结果作为上下文注入。

**核心流程：**
```python
def chat(self, message: str, ...):  
    # 1. 使用用户消息进行检索
    nodes = self._retriever.retrieve(message)
    
    # 2. 对检索结果进行后处理（rerank、filter 等）
    for postprocessor in self._node_postprocessors:
        nodes = postprocessor.postprocess_nodes(nodes, ...)
    
    # 3. 获取对话历史
    chat_history = self._memory.get(input=message)
    
    # 4. 构建 response synthesizer（动态 prompt）
    synthesizer = self._get_response_synthesizer(chat_history)
    
    # 5. 用 synthesizer 合成最终响应
    response = synthesizer.synthesize(message, nodes)
    
    # 6. 保存对话到 memory
    self._memory.put(user_message)
    self._memory.put(ai_message)
    
    return AgentChatResponse(response=str(response), source_nodes=nodes)
```

**Response Synthesizer 的作用：**
- `CompactAndRefine` 类型：当 context 太长时分批处理
- 动态构建 prompt，将上下文模板、对话历史、system prompt 组合

#### 1.2.3 CondenseQuestionChatEngine

**定位**：先将多轮对话历史 + 当前问题改写为独立问题，再查询。

**核心流程：**
```python
def chat(self, message: str, ...):
    # 1. 获取对话历史
    chat_history = self._memory.get(input=message)
    
    # 2. 问题改写（核心步骤）
    condensed_question = self._condense_question(chat_history, message)
    # 原问题："它有什么功能？"
    # 改写后："LlamaIndex 有什么功能？"（假设之前讨论过 LlamaIndex）
    
    # 3. 使用改写后的问题查询
    query_response = self._query_engine.query(condensed_question)
    
    # 4. 保存对话（注意：保存的是原始问题，而非改写后的）
    self._memory.put(ChatMessage(role=USER, content=message))
    self._memory.put(ChatMessage(role=ASSISTANT, content=str(query_response)))
    
    return AgentChatResponse(response=str(query_response))
```

**问题改写的实现：**
```python
def _condense_question(self, chat_history, last_message: str):
    if not chat_history:
        return last_message  # 第一轮对话，无需改写
    
    # 将对话历史转为字符串
    chat_history_str = messages_to_history_str(chat_history)
    # 输出类似："Human: 什么是 LlamaIndex？\nAssistant: LlamaIndex 是..."
    
    # 使用 LLM 改写
    return self._llm.predict(
        self._condense_question_prompt,
        question=last_message,
        chat_history=chat_history_str
    )
```

**默认 Condense Prompt：**
```
Given a conversation (between Human and Assistant) and a follow up message,
rewrite the message to be a standalone question that captures all relevant context.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
```

**为什么要问题改写？**
- 用户的问题常常依赖上下文："它支持哪些模型？"、"怎么安装？"
- 检索器无法理解"它"指的是什么
- 改写成独立问题后，检索效果更好

#### 1.2.4 CondensePlusContextChatEngine

**定位**：结合 CondenseQuestion 和 Context 两种模式，是最完整的 RAG 对话实现。

**核心流程（C3 = Condense + Context + Chat）：**
```python
def chat(self, message: str, ...):
    # === Phase 1: Condense ===
    chat_history = self._memory.get(input=message)
    condensed_question = self._condense_question(chat_history, message)
    # "它有什么功能？" -> "LlamaIndex 有什么功能？"
    
    # === Phase 2: Retrieve with Condensed Question ===
    context_nodes = self._get_nodes(condensed_question)
    # 用改写后的问题检索，效果更好
    
    for postprocessor in self._node_postprocessors:
        context_nodes = postprocessor.postprocess_nodes(context_nodes, ...)
    
    # === Phase 3: Synthesize with Context ===
    response_synthesizer = self._get_response_synthesizer(chat_history)
    response = synthesizer.synthesize(message, context_nodes)
    # 注意：synthesize 使用的是原始问题，而非改写后的
    
    # === Phase 4: Update Memory ===
    self._memory.put(ChatMessage(content=message, role=USER))
    self._memory.put(ChatMessage(content=str(response), role=ASSISTANT))
    
    return AgentChatResponse(
        response=str(response),
        sources=[context_source],
        source_nodes=context_nodes
    )
```

**为什么是最佳实践？**
- **Condense**：确保检索质量（用独立问题检索）
- **Context**：注入检索结果作为上下文
- **Memory**：保留完整对话历史

### 1.3 核心机制分析

#### 1.3.1 Memory 管理

**ChatMemoryBuffer 实现：**
```python
class ChatMemoryBuffer(BaseMemory):
    def __init__(self, token_limit: int = 3000):
        self.token_limit = token_limit
        self.chat_store = SimpleChatStore()
    
    def get(self, input: str = None, initial_token_count: int = 0):
        # 从最新的消息开始，逆序获取，直到 token 超限
        messages = self.chat_store.get_messages()
        
        token_count = initial_token_count
        cutoff_idx = 0
        
        # 逆序遍历，确保获取最新的对话
        for i in range(len(messages) - 1, -1, -1):
            token_count += len(self.tokenizer_fn(messages[i].content))
            if token_count > self.token_limit:
                cutoff_idx = i + 1
                break
        
        return messages[cutoff_idx:]  # 返回未超限的消息
```

**Token 限制策略：**
- 保留最新的对话，丢弃最早的对话
- `token_limit` 通常设为 `context_window - 256`（为响应预留空间）

#### 1.3.2 Streaming 响应处理

**StreamingAgentChatResponse 的设计：**
```python
@dataclass
class StreamingAgentChatResponse:
    chat_stream: ChatResponseGen = None    # LLM 的 streaming generator
    queue: Queue = field(default_factory=Queue)  # 线程安全的队列
    unformatted_response: str = ""         # 积累的完整响应
    is_done: bool = False                  # 标记是否完成
    
    def write_response_to_history(self, memory: BaseMemory):
        """后台线程执行，写入 memory"""
        final_text = ""
        for chat in self.chat_stream:
            self.put_in_queue(chat.delta)  # 放入队列供用户消费
            final_text += chat.delta or ""
        
        # 完整响应生成后，写入 memory
        memory.put(ChatMessage(content=final_text.strip()))
        self.is_done = True
    
    @property
    def response_gen(self) -> Generator:
        """用户调用此方法获取 streaming 响应"""
        while not self.is_done or not self.queue.empty():
            try:
                delta = self.queue.get(block=False)
                yield delta
            except Empty:
                time.sleep(0)  # 释放 GIL
```

**使用示例：**
```python
response = chat_engine.stream_chat("什么是 LlamaIndex？")

for token in response.response_gen:
    print(token, end="", flush=True)
```

---

## 二、Agent 模块分析

### 2.1 模块职责概述

Agent 模块的核心职责是**工具调用循环** + **推理决策**。

**主要功能：**
1. 工具调用循环：根据 LLM 输出调用工具，获取结果，再次调用 LLM
2. 推理步骤管理：维护 Thought-Action-Observation 的推理链
3. 多 Agent 协作：支持 agent 之间的 handoff（交接）
4. 状态管理：使用 Workflow 的 Context 管理状态
5. 结构化输出：将对话历史转换为结构化数据

**与 Chat Engine 的区别：**
- Chat Engine：对话编排，单次调用 LLM
- Agent：循环调用 LLM + 工具，直到得出答案

### 2.2 实现类型详解

#### 2.2.1 BaseWorkflowAgent（基础架构）

**核心设计：**
- 继承自 `Workflow`（LlamaIndex 的工作流引擎）
- 继承自 `BaseModel`（Pydantic，用于配置管理）
- 继承自 `PromptMixin`（支持 prompt 管理）

**关键字段：**
```python
class BaseWorkflowAgent(Workflow, BaseModel, PromptMixin):
    name: str = "Agent"                    # Agent 名称
    description: str = "An agent that..."   # Agent 描述
    system_prompt: Optional[str] = None     # System prompt
    tools: List[BaseTool] = None            # 可用工具列表
    can_handoff_to: List[str] = None        # 可交接的 agent 名称
    llm: LLM = Field(default_factory=get_default_llm)
    initial_state: Dict[str, Any] = {}      # 初始状态
    output_cls: Optional[Type[BaseModel]] = None  # 结构化输出类型
    streaming: bool = True                  # 是否 streaming
```

**核心抽象方法：**
```python
@abstractmethod
async def take_step(
    self, 
    ctx: Context,                    # Workflow context
    llm_input: List[ChatMessage],    # LLM 输入
    tools: Sequence[AsyncBaseTool],  # 可用工具
    memory: BaseMemory               # 对话记忆
) -> AgentOutput:
    """执行一步推理，返回 AgentOutput（包含 tool_calls）"""

@abstractmethod
async def handle_tool_call_results(
    self, 
    ctx: Context,
    results: List[ToolCallResult],  # 工具调用结果
    memory: BaseMemory
) -> None:
    """处理工具调用结果，更新状态"""

@abstractmethod
async def finalize(
    self, 
    ctx: Context,
    output: AgentOutput, 
    memory: BaseMemory
) -> AgentOutput:
    """完成推理，清理状态，返回最终输出"""
```

#### 2.2.2 ReActAgent（推理-行动循环）

**ReAct 模式：**
```
Thought: I need to search for information about X
Action: search
Action Input: {"query": "X"}
Observation: [search result]

Thought: Now I need to find more details about Y
Action: lookup
Action Input: {"term": "Y"}
Observation: [lookup result]

Thought: I can answer without using any more tools
Answer: Based on the information, ...
```

**核心组件：**
1. **ReActChatFormatter**：将工具列表 + 对话历史 + 推理步骤格式化为 prompt
2. **ReActOutputParser**：解析 LLM 输出，提取 Thought、Action、Action Input
3. **Reasoning Steps**：存储推理链

**take_step 实现逻辑：**
```python
async def take_step(self, ctx, llm_input, tools, memory):
    # 1. 获取当前推理历史
    current_reasoning = await ctx.store.get(self.reasoning_key, default=[])
    
    # 2. 格式化 LLM 输入
    input_chat = self.formatter.format(
        tools,
        chat_history=llm_input,
        current_reasoning=current_reasoning
    )
    
    # 3. 调用 LLM
    last_chat_response = await self._get_response(input_chat)
    
    # 4. 解析输出
    reasoning_step = self.output_parser.parse(last_chat_response.message.content)
    # reasoning_step 可能是：
    # - ActionReasoningStep（需要调用工具）
    # - ResponseReasoningStep（最终答案）
    
    # 5. 保存推理步骤
    current_reasoning.append(reasoning_step)
    await ctx.store.set(self.reasoning_key, current_reasoning)
    
    # 6. 如果是 ActionReasoningStep，生成 tool_calls
    if isinstance(reasoning_step, ActionReasoningStep):
        tool_calls = [
            ToolSelection(
                tool_id=str(uuid.uuid4()),
                tool_name=reasoning_step.action,
                tool_kwargs=reasoning_step.action_input
            )
        ]
    else:
        tool_calls = []
    
    return AgentOutput(response=last_chat_response.message, tool_calls=tool_calls)
```

**ReActOutputParser 解析逻辑：**
```python
def parse(self, output: str):
    # 尝试匹配不同的格式
    thought_match = re.search(r"Thought:", output)
    action_match = re.search(r"Action:", output)
    answer_match = re.search(r"Answer:", output)
    
    # Case 1: Thought + Action (需要调用工具)
    if action_match:
        thought, action, action_input = extract_tool_use(output)
        action_input_dict = json.loads(action_input)
        return ActionReasoningStep(
            thought=thought, 
            action=action, 
            action_input=action_input_dict
        )
    
    # Case 2: Thought + Answer (最终答案)
    if answer_match:
        thought, answer = extract_final_response(output)
        return ResponseReasoningStep(thought=thought, response=answer)
    
    # Case 3: 直接输出答案
    return ResponseReasoningStep(
        thought="(Implicit) I can answer without tools!",
        response=output
    )
```

**完整的 ReAct 循环示例：**
```
# 第 1 轮
User: What is the population of Tokyo plus 100?

LLM Output:
Thought: I need to search for Tokyo's population
Action: search
Action Input: {"query": "Tokyo population"}

[调用 search 工具]

# 第 2 轮
Observation: Tokyo has a population of 14 million

LLM Output:
Thought: Now I need to add 100 to 14 million
Action: calculator
Action Input: {"expression": "14000000 + 100"}

[调用 calculator 工具]

# 第 3 轮
Observation: 14000100

LLM Output:
Thought: I can answer without using any more tools
Answer: The population of Tokyo plus 100 is 14,000,100
```

#### 2.2.3 FunctionAgent（函数调用）

**定位**：基于 OpenAI function calling / tool calling，不需要复杂的 prompt engineering。

**与 ReActAgent 的区别：**
- ReActAgent：通过 prompt 让 LLM 输出特定格式，解析后调用工具
- FunctionAgent：直接使用 LLM 的 function calling 能力

**核心实现：**
```python
async def take_step(self, ctx, llm_input, tools, memory):
    # 1. 获取 scratchpad（临时对话历史）
    scratchpad = await ctx.store.get(self.scratchpad_key, default=[])
    current_llm_input = [*llm_input, *scratchpad]
    
    # 2. 调用 function calling LLM
    last_chat_response = await self.llm.achat_with_tools(
        chat_history=current_llm_input,
        tools=tools,  # 工具会被转换为 OpenAI function schema
        allow_parallel_tool_calls=self.allow_parallel_tool_calls
    )
    
    # 3. 提取 tool calls（LLM 直接返回结构化的 tool calls）
    tool_calls = self.llm.get_tool_calls_from_response(last_chat_response)
    
    # 4. 保存到 scratchpad
    scratchpad.append(last_chat_response.message)
    await ctx.store.set(self.scratchpad_key, scratchpad)
    
    return AgentOutput(response=last_chat_response.message, tool_calls=tool_calls)
```

**Function Calling 的优势：**
- 无需复杂的 prompt engineering
- LLM 直接输出结构化的 tool calls，无需解析
- 支持并行调用多个工具
- 更稳定，不容易出现格式错误

#### 2.2.4 CodeActAgent（代码执行）

**定位**：让 LLM 生成并执行 Python 代码来解决问题，适用于需要精确计算、数据处理的场景。

**核心思想：**
- LLM 生成的代码包裹在 `<execute>...</execute>` 标签中
- Agent 提取代码并执行
- 将执行结果返回给 LLM

**System Prompt 示例：**
```
You are a helpful AI assistant that can write and execute Python code.

## Response Format:
<execute>
import math
result = math.sqrt(16)
print(f"Result: {result}")
</execute>

## Available Functions:
{tool_descriptions}
```

**代码提取与执行：**
```python
def _extract_code_from_response(self, response_text: str):
    execute_pattern = r"<execute>(.*?)</execute>"
    execute_matches = re.findall(execute_pattern, response_text, re.DOTALL)
    
    if execute_matches:
        return "\n\n".join([x.strip() for x in execute_matches])
    return None

# 生成 tool call 来执行代码
tool_calls = [
    ToolSelection(
        tool_name="execute",
        tool_kwargs={"code": code}
    )
]
```

**完整的 CodeAct 循环示例：**
```
User: Calculate the area of a circle with radius 5, then multiply by 2

LLM Output:
<execute>
import math
radius = 5
area = math.pi * radius ** 2
result = area * 2
print(f"Result: {result:.2f}")
</execute>

[执行代码]

Observation: Result: 157.08

LLM Output:
The area of a circle with radius 5 is approximately 78.54 square units.
After multiplying by 2, the result is approximately 157.08 square units.
```

**CodeActAgent 的优势：**
- 精确计算（不依赖 LLM 的数学能力）
- 支持复杂的数据处理（pandas、numpy 等）
- 代码可复用（变量在会话中持久化）

### 2.3 核心机制分析

#### 2.3.1 Workflow 集成

**Agent Workflow 的主要步骤：**
```python
class BaseWorkflowAgent(Workflow):
    @step
    async def setup(self, ctx, ev: AgentWorkflowStartEvent):
        """初始化 agent 状态"""
        await ctx.store.set("memory", ChatMemoryBuffer())
        return AgentSetup(...)
    
    @step
    async def agent_step(self, ctx, ev: AgentSetup):
        """执行一步推理"""
        output = await self.take_step(ctx, ev.input, tools, memory)
        
        if output.tool_calls:
            return ToolCall(...)  # 有工具调用，继续循环
        else:
            return await self.finalize(ctx, output, memory)  # 结束
    
    @step
    async def call_tools(self, ctx, ev: ToolCall):
        """调用工具"""
        tool = self._get_tool_by_name(ev.tool_name)
        result = await tool.acall(**ev.tool_kwargs)
        return ToolCallResult(...)
    
    @step
    async def handle_results(self, ctx, ev: ToolCallResult):
        """处理工具结果，继续下一轮"""
        await self.handle_tool_call_results(ctx, [ev], memory)
        return AgentSetup(...)  # 继续下一轮
```

**Context 的作用：**
- 跨步骤共享状态（`ctx.store.get/set`）
- 事件流（`ctx.write_event_to_stream`）：供外部监听 agent 的执行过程

#### 2.3.2 多 Agent 协作（Handoff）

**Handoff 机制：**
- Agent A 可以将任务交给 Agent B 处理
- 通过 `can_handoff_to` 字段定义允许交接的 agent
- Handoff 本质上是一个特殊的 "tool"

**Handoff 示例：**
```
User: I want to book a flight to Tokyo and get weather info

# Agent 1: Router Agent
Thought: This requires booking and weather info. Hand off to booking agent.
Action: handoff
Action Input: {"target_agent": "booking_agent"}

# Agent 2: Booking Agent
[处理订票...]
Thought: Booking complete. Hand off to weather agent.
Action: handoff
Action Input: {"target_agent": "weather_agent"}

# Agent 3: Weather Agent
[查询天气...]
Answer: Flight booked. Weather in Tokyo: Sunny, 25°C.
```

#### 2.3.3 Scratchpad vs Reasoning

**两种状态管理方式：**

1. **Scratchpad**（FunctionAgent、CodeActAgent 使用）
   - 存储所有中间消息（assistant、tool、user）
   - 每次 take_step 时，将 scratchpad 追加到输入
   - finalize 时，将 scratchpad 保存到 memory，然后清空

2. **Reasoning**（ReActAgent 使用）
   - 存储推理步骤（ActionReasoningStep、ObservationReasoningStep）
   - 每次 take_step 时，由 formatter 将 reasoning 格式化为消息

---

## 三、架构分析与思考

### 3.1 职责边界

#### Chat Engine 的职责
- **对话编排**：管理多轮对话流程
- **问题改写**：处理上下文依赖的问题
- **RAG 集成**：将 retriever 和 LLM 组合起来
- **流式处理**：处理 streaming 响应

**不属于 Chat Engine：**
- 检索本身（由 Retriever 负责）
- 向量化/索引（由 Indexer 负责）
- 后处理（由 Postprocessor 负责）

#### Agent 的职责
- **推理循环**：Thought-Action-Observation
- **工具调用**：根据 LLM 输出调用工具
- **状态管理**：维护推理历史、scratchpad
- **多 Agent 协作**：handoff 机制

**不属于 Agent：**
- 对话历史管理（由 Memory 负责）
- 工具实现本身（由 Tool 负责）

### 3.2 这些模块是否应该在 RAG Core 中？

#### 从纯粹的 RAG 定义看
**RAG = Retrieval + Augmentation + Generation**

- **Core**：Indexer、Retriever、Postprocessor
- **Non-core**：Chat Engine、Agent

#### 从依赖关系看
```
Agent
  └─ depends on → Tool, Memory, LLM
  └─ may use → Retriever (as a tool)

Chat Engine
  └─ depends on → Retriever, Memory, LLM, Postprocessor
  
Retriever
  └─ depends on → VectorStore, Embedder
  └─ no dependency on Chat Engine or Agent
```

**结论：**
- Retriever 是 self-contained 的，不依赖上层
- Chat Engine 和 Agent 是**应用层编排**，依赖 Retriever

#### 从灵活性看
- Chat Engine 的对话流程是**固定**的（condense → retrieve → synthesize）
- 用户可能需要自定义流程（例如：先分类，再决定是否检索）
- Agent 的推理模式是**固定**的（ReAct、Function Calling）
- 用户可能想要自己的 agent 框架（LangChain、AutoGPT 等）

### 3.3 替代方案：Pure RAG Core

**如果 zag-ai 不包含 Chat Engine 和 Agent：**

```python
# 用户自己编排对话流程
from zag import VectorRetriever, TokenCompressor

retriever = VectorRetriever(...)
compressor = TokenCompressor(...)

# 第一轮对话
query1 = "什么是 LlamaIndex？"
nodes = retriever.retrieve(query1)
nodes = compressor.compress(nodes, query1)
context = format_context(nodes)
response1 = llm.chat([
    ChatMessage(role="system", content=context),
    ChatMessage(role="user", content=query1)
])

# 第二轮对话（用户自己管理历史）
query2 = "它有什么功能？"
# 用户自己决定：是否改写问题？
condensed_query2 = llm.complete(
    f"Rewrite: '{query2}' given context: '{response1}'"
)
nodes = retriever.retrieve(condensed_query2)
...
```

**提供工具函数，而非 Engine：**
```python
# zag/utils/conversation.py
def condense_question(chat_history, question, llm):
    """将依赖上下文的问题改写为独立问题"""
    history_str = "\n".join([f"{m.role}: {m.content}" for m in chat_history])
    prompt = f"Chat History:\n{history_str}\n\nQuestion: {question}\n\nStandalone:"
    return llm.complete(prompt)

def format_rag_context(nodes, template="Context:\n{context}"):
    """将检索结果格式化为 context"""
    context = "\n\n".join([n.get_content() for n in nodes])
    return template.format(context=context)
```

**优势：**
- 用户完全控制流程
- 可以自由组合不同的组件
- 易于集成到现有系统（如 LangChain）
- 核心保持简洁

**劣势：**
- 用户需要自己处理很多细节
- 学习曲线较陡
- 缺少开箱即用的解决方案

### 3.4 实用建议

**对于 zag-ai 项目：**

✅ **应该专注于 RAG Core：**
- Retriever（vector、hybrid、fusion）
- Indexer（索引构建）
- Postprocessor（rerank、filter、compress）
- Extractor（从文档提取结构化信息）

✅ **如何处理对话场景：**
- 提供简单的工具函数，而非完整的 Engine 类
- 例如：`condense_question()` 作为独立函数
- 让用户自己决定何时、如何使用

✅ **架构建议：**
```
zag-ai/
  core/
    retriever/     # RAG 核心
    indexer/       # RAG 核心
    postprocessor/ # RAG 核心
  utils/
    conversation.py  # 可选的工具函数
```

**关键原则：**
- **保持核心的纯粹性**：RAG core 不应该包含应用层逻辑
- **提供构建块，而非完整方案**：让用户自由组合
- **避免强制特定的交互模式**：不预设对话流程或 agent 模式

---

## 总结

### Chat Engine
- **本质**：对话编排层，处理多轮对话中的 RAG 流程
- **核心价值**：问题改写（condense）、上下文注入、streaming 处理
- **是否应该在 RAG Core**：否，属于应用层

### Agent
- **本质**：工具调用循环，实现"思考-行动-观察"模式
- **核心价值**：推理循环、状态管理、多 agent 协作
- **是否应该在 RAG Core**：否，属于应用层

### 建议
对于 zag-ai 这样强调架构清晰性的项目，应该：
1. **保持 core 的纯粹性**：只包含 RAG 的核心能力
2. **提供工具函数**：而非完整的 Engine/Agent 框架
3. **让用户自由组合**：不强制特定的应用模式
