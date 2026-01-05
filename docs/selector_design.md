# Selector 架构设计文档

## 目录

- [1. Selector 概述](#1-selector-概述)
- [2. 核心概念](#2-核心概念)
- [3. 架构设计](#3-架构设计)
- [4. 接口定义](#4-接口定义)
- [5. 实现策略](#5-实现策略)
- [6. 使用场景](#6-使用场景)
- [7. 使用示例](#7-使用示例)
- [8. 扩展指南](#8-扩展指南)
- [9. 测试建议](#9-测试建议)
- [10. 与 LlamaIndex 对比](#10-与-llamaindex-对比)

---

## 1. Selector 概述

### 1.1 什么是 Selector？

**Selector（选择器）是一个决策组件**，其核心职责是：

> **根据查询（Query）从多个候选项（Choices）中选择最合适的一个或多个**

### 1.2 为什么需要 Selector？

在 RAG 系统中，经常需要做出选择：

| 场景 | 需要选择的内容 |
|------|--------------|
| **RouterRetriever** | 从多个 Retriever 中选择最合适的 |
| **Agent** | 从多个 Tool 中选择最合适的 |
| **MultiIndexQueryEngine** | 从多个索引中选择最合适的 |
| **MultiModalRetriever** | 选择文本检索还是图像检索 |

**Selector 提供了统一的选择接口**，将选择逻辑与业务逻辑分离。

### 1.3 设计原则

- **职责单一**：Selector 只负责选择，不执行选中的操作
- **策略可替换**：支持多种选择策略（LLM、向量相似度、规则等）
- **接口统一**：所有 Selector 实现相同的接口
- **易于测试**：选择逻辑可以独立测试

---

## 2. 核心概念

### 2.1 输入输出

```
输入 1：Choices（候选项列表）
    - 每个候选项包含描述信息（ToolMetadata）

输入 2：Query（查询）
    - 用户查询文本

输出：SelectorResult（选择结果）
    - 选中的候选项索引
    - 选择理由
```

### 2.2 核心数据结构

```python
class ToolMetadata:
    """候选项元数据"""
    name: str                  # 候选项名称
    description: str           # 候选项描述


class SingleSelection:
    """单个选择"""
    index: int                 # 选中的候选项索引（0-based）
    reason: str                # 选择理由


class SelectorResult:
    """选择结果"""
    selections: List[SingleSelection]
    
    @property
    def ind(self) -> int:
        """单选时，返回索引"""
        return self.selections[0].index
    
    @property
    def inds(self) -> List[int]:
        """多选时，返回索引列表"""
        return [x.index for x in self.selections]
    
    @property
    def reason(self) -> str:
        """单选时，返回理由"""
        return self.selections[0].reason
    
    @property
    def reasons(self) -> List[str]:
        """多选时，返回理由列表"""
        return [x.reason for x in self.selections]
```

---

## 3. 架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    应用层                                │
│  RouterRetriever, Agent, MultiIndexQueryEngine          │
│                                                          │
│  使用 Selector 进行决策                                  │
└─────────────────────────────────────────────────────────┘
                         ↓ 依赖
┌─────────────────────────────────────────────────────────┐
│                  Selector Layer                          │
│                                                          │
│  BaseSelector (抽象接口)                                 │
│    ↓                                                     │
│  LLMSelector        EmbeddingSelector    RuleSelector   │
│  (LLM 选择)         (向量相似度)         (规则选择)      │
│                                                          │
│  特征：统一接口，策略可替换                               │
└─────────────────────────────────────────────────────────┘
```

### 3.2 目录结构

```
selector/
├── __init__.py                    # 统一导出
├── base.py                        # BaseSelector 抽象基类
│
├── llm_selector.py                # LLMSelector（LLM 选择）
├── embedding_selector.py          # EmbeddingSelector（向量相似度）
├── rule_selector.py               # RuleSelector（规则选择）
└── hybrid_selector.py             # HybridSelector（混合策略）
```

---

## 4. 接口定义

### 4.1 `selector/base.py`：抽象基类

```python
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

@dataclass
class ToolMetadata:
    """候选项元数据"""
    name: str                  # 候选项名称
    description: str           # 候选项描述
    metadata: dict = None      # 额外元数据（可选）


@dataclass
class SingleSelection:
    """单个选择"""
    index: int                 # 选中的候选项索引（0-based）
    reason: str                # 选择理由


@dataclass
class SelectorResult:
    """选择结果"""
    selections: List[SingleSelection]
    
    @property
    def ind(self) -> int:
        """单选时，返回索引"""
        if len(self.selections) != 1:
            raise ValueError(
                f"There are {len(self.selections)} selections, use .inds instead"
            )
        return self.selections[0].index
    
    @property
    def inds(self) -> List[int]:
        """多选时，返回索引列表"""
        return [x.index for x in self.selections]
    
    @property
    def reason(self) -> str:
        """单选时，返回理由"""
        if len(self.selections) != 1:
            raise ValueError(
                f"There are {len(self.selections)} selections, use .reasons instead"
            )
        return self.selections[0].reason
    
    @property
    def reasons(self) -> List[str]:
        """多选时，返回理由列表"""
        return [x.reason for x in self.selections]


class BaseSelector(ABC):
    """Selector 抽象基类
    
    所有 Selector 必须实现此接口
    """
    
    @abstractmethod
    def select(
        self, 
        choices: List[ToolMetadata], 
        query: Query
    ) -> SelectorResult:
        """选择最合适的候选项
        
        Args:
            choices: 候选项列表
            query: 查询对象
            
        Returns:
            选择结果（包含索引和理由）
        """
        pass
    
    async def aselect(
        self, 
        choices: List[ToolMetadata], 
        query: Query
    ) -> SelectorResult:
        """异步选择（可选实现）
        
        Args:
            choices: 候选项列表
            query: 查询对象
            
        Returns:
            选择结果
        """
        return self.select(choices, query)
```

---

## 5. 实现策略

### 5.1 `selector/llm_selector.py`：LLM 选择器

**使用 LLM 进行智能选择**

```python
from selector.base import BaseSelector, SelectorResult, SingleSelection, ToolMetadata
from typing import List
import json

class LLMSelector(BaseSelector):
    """LLM 选择器
    
    使用 LLM 根据查询和候选项描述进行智能选择。
    
    优点：
    - 智能理解查询意图
    - 可以处理复杂的选择逻辑
    - 提供详细的选择理由
    
    缺点：
    - 速度较慢
    - 需要调用 LLM API
    - 成本较高
    """
    
    def __init__(
        self, 
        llm: LLM,
        select_multi: bool = False,
        max_outputs: int = 1,
    ):
        """初始化 LLM 选择器
        
        Args:
            llm: LLM 实例
            select_multi: 是否允许多选
            max_outputs: 最多选择多少个（仅在 select_multi=True 时有效）
        """
        self.llm = llm
        self.select_multi = select_multi
        self.max_outputs = max_outputs
    
    def select(
        self, 
        choices: List[ToolMetadata], 
        query: Query
    ) -> SelectorResult:
        """使用 LLM 选择
        
        Args:
            choices: 候选项列表
            query: 查询对象
            
        Returns:
            选择结果
        """
        # 1. 构建候选项描述
        choices_text = self._build_choices_text(choices)
        
        # 2. 构建提示词
        prompt = self._build_prompt(choices_text, query, len(choices))
        
        # 3. 调用 LLM
        response = self.llm.complete(prompt)
        
        # 4. 解析输出
        return self._parse_response(response.text)
    
    def _build_choices_text(self, choices: List[ToolMetadata]) -> str:
        """构建候选项描述文本"""
        texts = []
        for i, choice in enumerate(choices):
            text = f"({i + 1}) {choice.description}"  # 1-based indexing for LLM
            texts.append(text)
        return "\n\n".join(texts)
    
    def _build_prompt(self, choices_text: str, query: Query, num_choices: int) -> str:
        """构建提示词"""
        if self.select_multi:
            return f"""You are an expert at selecting the most appropriate options for a given query.

Query: {query.text}

Available options:
{choices_text}

Select the {self.max_outputs} most suitable options for this query.
Return your answer in JSON format:
{{
  "selections": [
    {{"index": 1, "reason": "explanation"}},
    {{"index": 2, "reason": "explanation"}}
  ]
}}

Important: Use 1-based indexing (1, 2, 3, ..., {num_choices}).
"""
        else:
            return f"""You are an expert at selecting the most appropriate option for a given query.

Query: {query.text}

Available options:
{choices_text}

Which option is most suitable for this query?
Return your answer in JSON format:
{{
  "index": 1,
  "reason": "explanation"
}}

Important: Use 1-based indexing (1, 2, 3, ..., {num_choices}).
"""
    
    def _parse_response(self, response: str) -> SelectorResult:
        """解析 LLM 响应"""
        # 提取 JSON
        try:
            # 尝试提取 JSON 代码块
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            if self.select_multi:
                # 多选
                selections = [
                    SingleSelection(
                        index=sel["index"] - 1,  # 转换为 0-based
                        reason=sel["reason"]
                    )
                    for sel in data["selections"]
                ]
            else:
                # 单选
                selections = [
                    SingleSelection(
                        index=data["index"] - 1,  # 转换为 0-based
                        reason=data["reason"]
                    )
                ]
            
            return SelectorResult(selections=selections)
        
        except Exception as e:
            # 解析失败，返回默认值
            print(f"Failed to parse LLM response: {e}")
            print(f"Response: {response}")
            return SelectorResult(
                selections=[SingleSelection(index=0, reason="Parse error, default to first option")]
            )
```

---

### 5.2 `selector/embedding_selector.py`：向量相似度选择器

**通过向量相似度选择最匹配的候选项**

```python
from selector.base import BaseSelector, SelectorResult, SingleSelection, ToolMetadata
from typing import List
import numpy as np

class EmbeddingSelector(BaseSelector):
    """向量相似度选择器
    
    通过计算查询向量和候选项描述向量的相似度进行选择。
    
    优点：
    - 速度快（不需要调用 LLM）
    - 成本低
    - 适合候选项描述清晰的场景
    
    缺点：
    - 无法理解复杂逻辑
    - 依赖候选项描述的质量
    """
    
    def __init__(
        self, 
        embedder: Embedder,
        top_k: int = 1,
    ):
        """初始化向量相似度选择器
        
        Args:
            embedder: 向量化器
            top_k: 选择相似度最高的 top_k 个候选项
        """
        self.embedder = embedder
        self.top_k = top_k
    
    def select(
        self, 
        choices: List[ToolMetadata], 
        query: Query
    ) -> SelectorResult:
        """使用向量相似度选择
        
        Args:
            choices: 候选项列表
            query: 查询对象
            
        Returns:
            选择结果
        """
        # 1. 计算查询向量
        query_embedding = self.embedder.embed(query.text)
        
        # 2. 计算每个候选项的向量
        choice_embeddings = [
            self.embedder.embed(choice.description)
            for choice in choices
        ]
        
        # 3. 计算相似度
        similarities = [
            self._cosine_similarity(query_embedding, choice_embedding)
            for choice_embedding in choice_embeddings
        ]
        
        # 4. 选择 top_k
        top_indices = np.argsort(similarities)[-self.top_k:][::-1].tolist()
        
        # 5. 构建结果
        selections = [
            SingleSelection(
                index=idx,
                reason=f"Cosine similarity: {similarities[idx]:.4f}"
            )
            for idx in top_indices
        ]
        
        return SelectorResult(selections=selections)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
```

---

### 5.3 `selector/rule_selector.py`：规则选择器

**基于规则进行选择**

```python
from selector.base import BaseSelector, SelectorResult, SingleSelection, ToolMetadata
from typing import List, Callable

class RuleSelector(BaseSelector):
    """规则选择器
    
    基于预定义的规则进行选择。
    
    优点：
    - 确定性强
    - 速度最快
    - 无需调用外部服务
    
    缺点：
    - 灵活性差
    - 需要手动定义规则
    """
    
    def __init__(
        self, 
        rules: List[Callable[[Query, ToolMetadata], float]]
    ):
        """初始化规则选择器
        
        Args:
            rules: 规则列表，每个规则是一个函数 (query, choice) -> score
        """
        self.rules = rules
    
    def select(
        self, 
        choices: List[ToolMetadata], 
        query: Query
    ) -> SelectorResult:
        """使用规则选择
        
        Args:
            choices: 候选项列表
            query: 查询对象
            
        Returns:
            选择结果
        """
        # 1. 计算每个候选项的得分
        scores = []
        for choice in choices:
            score = 0.0
            for rule in self.rules:
                score += rule(query, choice)
            scores.append(score)
        
        # 2. 选择得分最高的
        max_idx = scores.index(max(scores))
        
        # 3. 构建结果
        return SelectorResult(
            selections=[
                SingleSelection(
                    index=max_idx,
                    reason=f"Rule score: {scores[max_idx]:.2f}"
                )
            ]
        )


# 示例规则
def keyword_match_rule(query: Query, choice: ToolMetadata) -> float:
    """关键词匹配规则"""
    keywords = ["exact", "keyword", "term"]
    if any(kw in query.text.lower() for kw in keywords):
        if "keyword" in choice.description.lower():
            return 10.0
    return 0.0


def semantic_query_rule(query: Query, choice: ToolMetadata) -> float:
    """语义查询规则"""
    semantic_keywords = ["what", "how", "why", "explain", "meaning"]
    if any(kw in query.text.lower() for kw in semantic_keywords):
        if "semantic" in choice.description.lower() or "vector" in choice.description.lower():
            return 10.0
    return 0.0
```

---

### 5.4 `selector/hybrid_selector.py`：混合选择器

**结合多种策略进行选择**

```python
from selector.base import BaseSelector, SelectorResult, SingleSelection, ToolMetadata
from typing import List

class HybridSelector(BaseSelector):
    """混合选择器
    
    结合多种策略（如向量相似度 + 规则）进行选择。
    
    优点：
    - 综合多种策略的优点
    - 更准确
    
    缺点：
    - 实现复杂
    - 需要调优权重
    """
    
    def __init__(
        self, 
        embedding_selector: EmbeddingSelector,
        rule_selector: RuleSelector,
        embedding_weight: float = 0.7,
        rule_weight: float = 0.3,
    ):
        """初始化混合选择器
        
        Args:
            embedding_selector: 向量相似度选择器
            rule_selector: 规则选择器
            embedding_weight: 向量相似度权重
            rule_weight: 规则权重
        """
        self.embedding_selector = embedding_selector
        self.rule_selector = rule_selector
        self.embedding_weight = embedding_weight
        self.rule_weight = rule_weight
    
    def select(
        self, 
        choices: List[ToolMetadata], 
        query: Query
    ) -> SelectorResult:
        """使用混合策略选择
        
        Args:
            choices: 候选项列表
            query: 查询对象
            
        Returns:
            选择结果
        """
        # 1. 使用向量相似度选择器
        embedding_result = self.embedding_selector.select(choices, query)
        
        # 2. 使用规则选择器
        rule_result = self.rule_selector.select(choices, query)
        
        # 3. 综合结果
        # 简化实现：如果两者选择相同，则返回；否则使用权重更高的
        if embedding_result.ind == rule_result.ind:
            return embedding_result
        
        if self.embedding_weight > self.rule_weight:
            return embedding_result
        else:
            return rule_result
```

---

## 6. 使用场景

### 6.1 RouterRetriever：选择 Retriever

```python
from retriever import BaseRetriever, VectorRetriever, KeywordRetriever
from selector import LLMSelector

class RouterRetriever(BaseRetriever):
    """路由检索器
    
    根据查询选择最合适的 Retriever
    """
    
    def __init__(
        self, 
        retrievers: List[BaseRetriever],
        descriptions: List[str],
        selector: BaseSelector,
    ):
        """初始化路由检索器
        
        Args:
            retrievers: Retriever 列表
            descriptions: 每个 Retriever 的描述
            selector: 选择器
        """
        self.retrievers = retrievers
        self.descriptions = descriptions
        self.selector = selector
    
    def retrieve(self, query: Query) -> List[Node]:
        """执行路由检索
        
        Args:
            query: 查询对象
            
        Returns:
            检索到的节点列表
        """
        # 1. 构建候选项
        choices = [
            ToolMetadata(
                name=f"retriever_{i}",
                description=desc
            )
            for i, desc in enumerate(self.descriptions)
        ]
        
        # 2. 使用 Selector 选择
        result = self.selector.select(choices, query)
        
        # 3. 调用选中的 Retriever
        selected_idx = result.ind
        selected_retriever = self.retrievers[selected_idx]
        
        print(f"[RouterRetriever] Selected retriever {selected_idx}: {result.reason}")
        
        return selected_retriever.retrieve(query)
```

---

### 6.2 Agent：选择 Tool

```python
class ReActAgent:
    """ReAct Agent
    
    根据任务选择最合适的 Tool
    """
    
    def __init__(
        self, 
        tools: List[Tool],
        selector: BaseSelector,
        llm: LLM,
    ):
        self.tools = tools
        self.selector = selector
        self.llm = llm
    
    def run(self, task: str) -> str:
        """执行任务
        
        Args:
            task: 任务描述
            
        Returns:
            执行结果
        """
        # 1. 构建候选项
        choices = [
            ToolMetadata(
                name=tool.name,
                description=tool.description
            )
            for tool in self.tools
        ]
        
        # 2. 使用 Selector 选择 Tool
        query = Query(text=task)
        result = self.selector.select(choices, query)
        
        # 3. 调用选中的 Tool
        selected_tool = self.tools[result.ind]
        
        print(f"[Agent] Selected tool: {selected_tool.name}")
        print(f"[Agent] Reason: {result.reason}")
        
        return selected_tool.call(task)
```

---

### 6.3 MultiIndexQueryEngine：选择索引

```python
class MultiIndexQueryEngine:
    """多索引查询引擎
    
    根据查询选择最合适的索引
    """
    
    def __init__(
        self, 
        query_engines: List[BaseQueryEngine],
        descriptions: List[str],
        selector: BaseSelector,
    ):
        self.query_engines = query_engines
        self.descriptions = descriptions
        self.selector = selector
    
    def query(self, query_str: str) -> Response:
        """执行查询
        
        Args:
            query_str: 查询字符串
            
        Returns:
            查询响应
        """
        # 1. 构建候选项
        choices = [
            ToolMetadata(
                name=f"index_{i}",
                description=desc
            )
            for i, desc in enumerate(self.descriptions)
        ]
        
        # 2. 使用 Selector 选择索引
        query = Query(text=query_str)
        result = self.selector.select(choices, query)
        
        # 3. 查询选中的索引
        selected_qe = self.query_engines[result.ind]
        
        print(f"[MultiIndexQueryEngine] Selected index {result.ind}: {result.reason}")
        
        return selected_qe.query(query_str)
```

---

## 7. 使用示例

### 7.1 基础用法：LLMSelector

```python
from selector import LLMSelector, ToolMetadata

# 创建 LLM Selector
selector = LLMSelector(llm=llm, select_multi=False)

# 构建候选项
choices = [
    ToolMetadata(
        name="vector_search",
        description="Semantic similarity search using vector embeddings. Best for conceptual queries."
    ),
    ToolMetadata(
        name="keyword_search",
        description="Exact text matching using BM25 algorithm. Best for precise term queries."
    ),
]

# 执行选择
query = Query(text="What is machine learning?")
result = selector.select(choices, query)

print(f"Selected: {result.ind}")
print(f"Reason: {result.reason}")
# 输出：
# Selected: 0
# Reason: Vector search is best for semantic queries like 'What is machine learning?'
```

---

### 7.2 多选：LLMSelector with select_multi=True

```python
# 创建多选 Selector
selector = LLMSelector(llm=llm, select_multi=True, max_outputs=2)

# 构建候选项
choices = [
    ToolMetadata(name="vector_search", description="Semantic similarity search"),
    ToolMetadata(name="keyword_search", description="Exact text matching"),
    ToolMetadata(name="graph_search", description="Graph-based relationship search"),
]

# 执行选择
query = Query(text="Find documents about machine learning and neural networks")
result = selector.select(choices, query)

print(f"Selected: {result.inds}")
print(f"Reasons: {result.reasons}")
# 输出：
# Selected: [0, 1]
# Reasons: ['Vector search for semantic similarity', 'Keyword search for exact terms']
```

---

### 7.3 EmbeddingSelector

```python
from selector import EmbeddingSelector

# 创建 Embedding Selector
selector = EmbeddingSelector(embedder=embedder, top_k=1)

# 构建候选项
choices = [
    ToolMetadata(name="vector_search", description="Semantic similarity search"),
    ToolMetadata(name="keyword_search", description="Exact text matching"),
]

# 执行选择
query = Query(text="What is machine learning?")
result = selector.select(choices, query)

print(f"Selected: {result.ind}")
print(f"Reason: {result.reason}")
# 输出：
# Selected: 0
# Reason: Cosine similarity: 0.8521
```

---

### 7.4 RuleSelector

```python
from selector import RuleSelector, keyword_match_rule, semantic_query_rule

# 创建规则选择器
selector = RuleSelector(
    rules=[keyword_match_rule, semantic_query_rule]
)

# 构建候选项
choices = [
    ToolMetadata(
        name="vector_search",
        description="Semantic similarity search using vector embeddings"
    ),
    ToolMetadata(
        name="keyword_search",
        description="Exact keyword matching using BM25"
    ),
]

# 执行选择
query1 = Query(text="What is machine learning?")
result1 = selector.select(choices, query1)
print(f"Query 1 selected: {result1.ind}")  # 0 (vector search)

query2 = Query(text="Find exact term: neural network")
result2 = selector.select(choices, query2)
print(f"Query 2 selected: {result2.ind}")  # 1 (keyword search)
```

---

### 7.5 集成到 RouterRetriever

```python
from retriever import VectorRetriever, KeywordRetriever, RouterRetriever
from selector import LLMSelector

# 创建 Retriever
vector_retriever = VectorRetriever(
    vector_store=chroma_adapter,
    doc_store=doc_store,
    embedder=embedder,
)

keyword_retriever = KeywordRetriever(
    keyword_store=meilisearch_adapter,
    doc_store=doc_store,
)

# 创建 Selector
selector = LLMSelector(llm=llm)

# 创建 RouterRetriever
router = RouterRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    descriptions=[
        "Semantic similarity search using vector embeddings. Best for conceptual queries.",
        "Exact text matching using BM25 algorithm. Best for precise term queries.",
    ],
    selector=selector,
)

# 执行检索
query = Query(text="What is machine learning?")
nodes = router.retrieve(query)

# 输出：
# [RouterRetriever] Selected retriever 0: Vector search is best for semantic queries
```

---

## 8. 扩展指南

### 8.1 添加新的 Selector

**步骤**：

1. 在 `selector/` 目录下创建新文件
2. 继承 `BaseSelector`
3. 实现 `select()` 方法
4. 在 `selector/__init__.py` 中导出

**示例：添加 RandomSelector**

```python
# selector/random_selector.py
import random
from selector.base import BaseSelector, SelectorResult, SingleSelection

class RandomSelector(BaseSelector):
    """随机选择器（用于测试）"""
    
    def select(self, choices: List[ToolMetadata], query: Query) -> SelectorResult:
        # 随机选择一个
        random_idx = random.randint(0, len(choices) - 1)
        
        return SelectorResult(
            selections=[
                SingleSelection(
                    index=random_idx,
                    reason="Randomly selected"
                )
            ]
        )
```

---

### 8.2 自定义规则

```python
# 示例：基于查询长度的规则
def query_length_rule(query: Query, choice: ToolMetadata) -> float:
    """查询长度规则
    
    短查询（< 5 个词）倾向于关键词检索
    长查询（>= 5 个词）倾向于向量检索
    """
    word_count = len(query.text.split())
    
    if word_count < 5:
        # 短查询：关键词检索得分高
        if "keyword" in choice.description.lower():
            return 5.0
    else:
        # 长查询：向量检索得分高
        if "vector" in choice.description.lower() or "semantic" in choice.description.lower():
            return 5.0
    
    return 0.0


# 使用自定义规则
selector = RuleSelector(
    rules=[
        keyword_match_rule,
        semantic_query_rule,
        query_length_rule,  # 添加自定义规则
    ]
)
```

---

### 8.3 组合多个 Selector

```python
class EnsembleSelector(BaseSelector):
    """集成选择器
    
    组合多个 Selector，通过投票决定最终选择
    """
    
    def __init__(self, selectors: List[BaseSelector]):
        self.selectors = selectors
    
    def select(self, choices: List[ToolMetadata], query: Query) -> SelectorResult:
        # 1. 收集所有 Selector 的选择
        votes = {}
        for selector in self.selectors:
            result = selector.select(choices, query)
            idx = result.ind
            votes[idx] = votes.get(idx, 0) + 1
        
        # 2. 选择票数最多的
        max_idx = max(votes, key=votes.get)
        
        return SelectorResult(
            selections=[
                SingleSelection(
                    index=max_idx,
                    reason=f"Voted by {votes[max_idx]}/{len(self.selectors)} selectors"
                )
            ]
        )
```

---

## 9. 测试建议

### 9.1 单元测试：LLMSelector

```python
import pytest
from unittest.mock import Mock

def test_llm_selector():
    # Mock LLM
    mock_llm = Mock()
    mock_llm.complete.return_value = Mock(text='{"index": 1, "reason": "Best match"}')
    
    # 创建 Selector
    selector = LLMSelector(llm=mock_llm)
    
    # 构建候选项
    choices = [
        ToolMetadata(name="option1", description="First option"),
        ToolMetadata(name="option2", description="Second option"),
    ]
    
    # 执行选择
    query = Query(text="test query")
    result = selector.select(choices, query)
    
    # 验证
    assert result.ind == 0  # 1-based -> 0-based
    assert "Best match" in result.reason
    mock_llm.complete.assert_called_once()
```

---

### 9.2 单元测试：EmbeddingSelector

```python
def test_embedding_selector():
    # Mock Embedder
    mock_embedder = Mock()
    mock_embedder.embed.side_effect = [
        [1.0, 0.0, 0.0],  # query embedding
        [0.9, 0.1, 0.0],  # choice 1 embedding (high similarity)
        [0.1, 0.9, 0.0],  # choice 2 embedding (low similarity)
    ]
    
    # 创建 Selector
    selector = EmbeddingSelector(embedder=mock_embedder)
    
    # 构建候选项
    choices = [
        ToolMetadata(name="option1", description="First option"),
        ToolMetadata(name="option2", description="Second option"),
    ]
    
    # 执行选择
    query = Query(text="test query")
    result = selector.select(choices, query)
    
    # 验证
    assert result.ind == 0  # 选择相似度最高的
    assert "similarity" in result.reason.lower()
```

---

### 9.3 集成测试：RouterRetriever

```python
def test_router_retriever_with_selector():
    # 创建真实的组件
    vector_retriever = VectorRetriever(...)
    keyword_retriever = KeywordRetriever(...)
    
    selector = LLMSelector(llm=llm)
    
    router = RouterRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        descriptions=[
            "Semantic search",
            "Keyword search",
        ],
        selector=selector,
    )
    
    # 执行检索
    query = Query(text="What is machine learning?")
    nodes = router.retrieve(query)
    
    # 验证
    assert len(nodes) > 0
```

---

## 10. 与 LlamaIndex 对比

### 10.1 架构对比

| 维度 | 本设计 | LlamaIndex |
|------|--------|-----------|
| **目录结构** | `selector/` 独立模块 | `core/selectors/` + `base/base_selector.py` |
| **抽象接口** | `BaseSelector` | `BaseSelector` |
| **实现策略** | LLM, Embedding, Rule, Hybrid | LLM, Embedding, Pydantic |
| **返回格式** | `SelectorResult` | `SelectorResult` (相同) |
| **多选支持** | ✅ 支持 | ✅ 支持 |

### 10.2 设计差异

| 特性 | 本设计 | LlamaIndex |
|------|--------|-----------|
| **Pydantic Selector** | ❌ 未实现（可选） | ✅ 支持 Function Calling |
| **Rule Selector** | ✅ 支持 | ❌ 未实现 |
| **Hybrid Selector** | ✅ 支持 | ❌ 未实现 |
| **提示词管理** | 简化（内联） | 复杂（Prompt Mixin） |

### 10.3 本设计的优势

- ✅ **更简洁**：去掉了复杂的 Prompt Mixin 机制
- ✅ **更灵活**：支持 RuleSelector 和 HybridSelector
- ✅ **更易理解**：目录结构清晰，职责分明
- ✅ **更易扩展**：可以轻松添加新的选择策略

---

## 附录：完整的 `selector/__init__.py`

```python
# 导出基类
from .base import (
    BaseSelector,
    ToolMetadata,
    SingleSelection,
    SelectorResult,
)

# 导出实现
from .llm_selector import LLMSelector
from .embedding_selector import EmbeddingSelector
from .rule_selector import RuleSelector, keyword_match_rule, semantic_query_rule
from .hybrid_selector import HybridSelector

__all__ = [
    # 基类
    "BaseSelector",
    "ToolMetadata",
    "SingleSelection",
    "SelectorResult",
    
    # 实现
    "LLMSelector",
    "EmbeddingSelector",
    "RuleSelector",
    "HybridSelector",
    
    # 规则
    "keyword_match_rule",
    "semantic_query_rule",
]
```

---

## 总结

本设计文档提供了一个清晰、可扩展的 Selector 架构：

1. **统一接口**：所有 Selector 实现 `BaseSelector` 接口
2. **多种策略**：LLM、向量相似度、规则、混合
3. **易于集成**：可以轻松集成到 RouterRetriever、Agent、QueryEngine 等组件
4. **易于扩展**：可以轻松添加新的选择策略
5. **职责单一**：Selector 只负责选择，不执行选中的操作

按照本文档实现，你将获得一个灵活、强大的 Selector 模块！
