# Postprocessor 架构设计文档

## 目录

- [1. 设计原则](#1-设计原则)
- [2. 架构概览](#2-架构概览)
- [3. 目录结构](#3-目录结构)
- [4. 基础层 Postprocessor](#4-基础层-postprocessor)
- [5. 组合层 Postprocessor](#5-组合层-postprocessor)
- [6. 使用示例](#6-使用示例)
- [7. 扩展指南](#7-扩展指南)

---

## 1. 设计原则

### 1.1 职责分离

**检索层（Retriever）**：
- 从存储中获取数据
- 负责检索策略（向量检索、关键词检索、混合检索等）
- 返回初步的检索结果

**后处理层（Postprocessor）**：
- 对检索结果进行优化
- 不直接访问存储层
- 通过组合实现复杂的后处理流程

### 1.2 统一抽象

所有后处理器（包括 Reranker）都实现统一的 `BasePostprocessor` 接口：

```python
class BasePostprocessor(ABC):
    @abstractmethod
    def process(
        self, 
        query: str, 
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """后处理检索结果"""
        pass
```

**统一接口的好处**：
- ✅ 可以用相同的方式组合所有后处理器
- ✅ 便于实现链式调用和条件组合
- ✅ 降低学习成本

### 1.3 为什么将 Reranker 归入 Postprocessors？

**核心论点**：Reranker 本质上是一种特殊的后处理器

**理由**：
1. **语义统一**：Reranker 不从存储中检索新数据，而是对已有结果重新排序
2. **职责清晰**：
   - Retriever：从存储中获取数据（retrieve from storage）
   - Postprocessor：处理检索结果（post-process results）
   - Reranker：重新排序结果（reorder results）
3. **接口兼容**：可以使用统一的 `process()` 接口
4. **便于组合**：作为 Postprocessor 的一种，可以和其他处理器自然组合

**设计决策**：
- Reranker 作为 Postprocessor 的子类
- 保留 `rerank()` 方法的语义（对熟悉 reranking 的开发者更友好）
- 通过 `process()` 方法统一接口

---

## 2. 架构概览

### 2.1 模块组织

```
┌─────────────────────────────────────────────────────────┐
│           Composite Layer (组合层)                       │
│                                                          │
│  ChainPostprocessor     ConditionalPostprocessor        │
│  - 链式组合              - 条件选择                      │
│                                                          │
│  特征：组合多个 Postprocessor，实现编排逻辑              │
└─────────────────────────────────────────────────────────┘
                         ↓ 依赖
┌─────────────────────────────────────────────────────────┐
│           Basic Layer (基础层)                           │
│                                                          │
│  Rerankers          Filters         Augmentors          │
│  - 重排序            - 过滤          - 上下文增强        │
│                                                          │
│  Compressors        Transformers                        │
│  - Token压缩         - 格式转换                          │
│                                                          │
│  特征：原子性处理器，实现单一后处理功能                  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心接口

```python
class BasePostprocessor(ABC):
    """所有后处理器的抽象基类"""
    
    @abstractmethod
    def process(
        self, 
        query: str, 
        units: list[BaseUnit]
    ) -> list[BaseUnit]:
        """
        后处理检索结果
        
        Args:
            query: 原始查询文本
            units: 待处理的 units
            
        Returns:
            处理后的 units
        """
        pass
```

---

## 3. 目录结构

```
postprocessors/
├── __init__.py                    # 统一导出所有 Postprocessor
├── base.py                        # BasePostprocessor 抽象基类
│
├── rerankers/                     # 重排序器（子模块）
│   ├── __init__.py
│   ├── base.py                   # BaseReranker
│   ├── cross_encoder.py          # CrossEncoderReranker
│   └── llm_reranker.py           # LLMReranker
│
├── filters/                       # 过滤器（子模块）
│   ├── __init__.py
│   ├── similarity.py             # SimilarityFilter
│   ├── metadata.py               # MetadataFilter
│   └── deduplicator.py           # Deduplicator
│
├── augmentors/                    # 增强器（子模块）
│   ├── __init__.py
│   ├── context.py                # ContextAugmentor
│   └── metadata.py               # MetadataAugmentor
│
├── compressors/                   # 压缩器（子模块）
│   ├── __init__.py
│   ├── token.py                  # TokenCompressor
│   └── redundancy.py             # RedundancyCompressor
│
├── transformers/                  # 转换器（子模块）
│   ├── __init__.py
│   ├── summarizer.py             # Summarizer
│   └── highlighter.py            # KeywordHighlighter
│
└── composite/                     # 组合器（编排层）
    ├── __init__.py
    ├── chain.py                  # ChainPostprocessor
    └── conditional.py            # ConditionalPostprocessor
```

**命名规范**：
- 基础层文件：`{type}.py`（如 `similarity.py`）
- 组合层文件：`{strategy}.py`（如 `chain.py`）
- 类名：使用 PascalCase，以功能名结尾（如 `SimilarityFilter`）

---

## 4. 基础层 Postprocessor

### 4.1 `postprocessors/base.py`：抽象基类

```python
from abc import ABC, abstractmethod
from typing import Any, Optional

class BasePostprocessor(ABC):
    """
    所有后处理器的抽象基类
    
    所有后处理器（包括 rerankers、filters 等）实现统一接口。
    这允许灵活的组合和嵌套。
    
    Design Philosophy:
        - Single responsibility: 每个处理器专注于一个后处理功能
        - Composability: 处理器可以组合和嵌套
        - Uniformity: 所有处理器使用相同的接口
    """
    
    @abstractmethod
    def process(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        后处理检索结果
        
        Args:
            query: 原始查询文本
            units: 待处理的 units
            
        Returns:
            处理后的 units（可能改变数量、顺序、内容）
        """
        pass
    
    async def aprocess(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        异步版本的 process（可选实现）
        
        Args:
            query: 原始查询文本
            units: 待处理的 units
            
        Returns:
            处理后的 units
        """
        # 默认实现：调用同步版本
        return self.process(query, units)
```

---

### 4.2 Rerankers - 重排序器

#### `postprocessors/rerankers/base.py`

```python
from ..base import BasePostprocessor

class BaseReranker(BasePostprocessor):
    """
    重排序器基类（特殊的后处理器）
    
    Reranker 使用更精确的模型重新计算相关性分数并排序。
    
    特点：
        - 改变结果的排序（通过重新计算分数）
        - 通常使用复杂模型（Cross-encoder、LLM等）
        - 计算密集，但能显著提升结果质量
    
    Note:
        Reranker 继承自 BasePostprocessor，既实现了 process() 接口，
        也保留了 rerank() 方法以保持语义清晰。
    """
    
    def process(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        实现 BasePostprocessor 接口
        
        内部调用 rerank() 方法
        """
        return self.rerank(query, units)
    
    @abstractmethod
    def rerank(
        self, 
        query: str,
        units: list['BaseUnit'],
        top_k: Optional[int] = None
    ) -> list['BaseUnit']:
        """
        重新排序 units
        
        Args:
            query: 原始查询文本
            units: 待重排序的 units
            top_k: 返回的最大结果数（None表示全部）
            
        Returns:
            重排序后的 units，按新的相关性分数排序
        """
        pass
```

#### `postprocessors/rerankers/cross_encoder.py`

```python
from typing import Optional
from .base import BaseReranker

class CrossEncoderReranker(BaseReranker):
    """
    基于 Cross-Encoder 的重排序器
    
    使用 Cross-Encoder 模型（如 BERT）同时编码 query 和 document，
    计算更准确的相关性分数。
    
    特点：
        - 准确度高：同时考虑 query 和 document
        - 速度较慢：需要对每个 pair 单独编码
        - 适合精排（top-k较小的场景）
    
    Examples:
        >>> from zag.postprocessors.rerankers import CrossEncoderReranker
        >>> 
        >>> reranker = CrossEncoderReranker(
        ...     model="cross-encoder/ms-marco-MiniLM-L-12-v2"
        ... )
        >>> 
        >>> reranked = reranker.rerank(query, units, top_k=10)
    """
    
    def __init__(
        self, 
        model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
    ):
        """
        初始化 Cross-Encoder 重排序器
        
        Args:
            model: Cross-Encoder 模型名称或路径
            device: 计算设备（"cuda" 或 "cpu"，None 表示自动选择）
        """
        self.model_name = model
        self.device = device
        self._model = None
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
    
    def rerank(
        self, 
        query: str,
        units: list['BaseUnit'],
        top_k: Optional[int] = None
    ) -> list['BaseUnit']:
        """
        使用 Cross-Encoder 重排序
        
        Args:
            query: 原始查询文本
            units: 待重排序的 units
            top_k: 返回的最大结果数
            
        Returns:
            重排序后的 units
        """
        if not units:
            return []
        
        # 加载模型
        self._load_model()
        
        # 构建 query-document pairs
        pairs = [[query, unit.content] for unit in units]
        
        # 计算相关性分数
        scores = self._model.predict(pairs)
        
        # 更新分数并排序
        for unit, score in zip(units, scores):
            unit.score = float(score)
        
        sorted_units = sorted(units, key=lambda x: x.score, reverse=True)
        
        # 返回 top_k 结果
        if top_k is not None:
            return sorted_units[:top_k]
        return sorted_units
```

---

### 4.3 Filters - 过滤器

#### `postprocessors/filters/similarity.py`

```python
from ..base import BasePostprocessor

class SimilarityFilter(BasePostprocessor):
    """
    相似度过滤器
    
    只保留相似度分数高于阈值的 units。
    
    适用场景：
        - 过滤低质量结果
        - 确保最低相关性要求
        - 控制结果质量
    
    Examples:
        >>> from zag.postprocessors.filters import SimilarityFilter
        >>> 
        >>> filter = SimilarityFilter(threshold=0.7)
        >>> filtered = filter.process(query, units)
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        初始化相似度过滤器
        
        Args:
            threshold: 相似度阈值（0-1），低于此值的结果会被过滤
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
    
    def process(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        过滤低相似度结果
        
        Args:
            query: 原始查询文本（本过滤器不使用）
            units: 待过滤的 units
            
        Returns:
            相似度 >= threshold 的 units
        """
        return [
            unit for unit in units 
            if hasattr(unit, 'score') and unit.score is not None 
            and unit.score >= self.threshold
        ]
```

#### `postprocessors/filters/deduplicator.py`

```python
from ..base import BasePostprocessor

class Deduplicator(BasePostprocessor):
    """
    去重器
    
    移除重复或高度相似的 units。
    
    去重策略：
        - exact: 完全相同的内容（基于 unit_id）
        - content: 内容相同（基于 content hash）
        - semantic: 语义相似（基于嵌入相似度）
    
    Examples:
        >>> from zag.postprocessors.filters import Deduplicator
        >>> 
        >>> dedup = Deduplicator(strategy="exact")
        >>> unique = dedup.process(query, units)
    """
    
    def __init__(
        self, 
        strategy: str = "exact",
        similarity_threshold: float = 0.95
    ):
        """
        初始化去重器
        
        Args:
            strategy: 去重策略 ("exact", "content", "semantic")
            similarity_threshold: 语义相似度阈值（仅 semantic 模式使用）
        """
        if strategy not in ["exact", "content", "semantic"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
    
    def process(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        去除重复的 units
        
        Args:
            query: 原始查询文本
            units: 待去重的 units
            
        Returns:
            去重后的 units
        """
        if self.strategy == "exact":
            return self._exact_dedup(units)
        elif self.strategy == "content":
            return self._content_dedup(units)
        else:  # semantic
            return self._semantic_dedup(units)
    
    def _exact_dedup(self, units: list['BaseUnit']) -> list['BaseUnit']:
        """基于 unit_id 去重"""
        seen = set()
        result = []
        for unit in units:
            if unit.unit_id not in seen:
                seen.add(unit.unit_id)
                result.append(unit)
        return result
    
    def _content_dedup(self, units: list['BaseUnit']) -> list['BaseUnit']:
        """基于 content hash 去重"""
        import hashlib
        seen = set()
        result = []
        for unit in units:
            content_hash = hashlib.md5(str(unit.content).encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                result.append(unit)
        return result
    
    def _semantic_dedup(self, units: list['BaseUnit']) -> list['BaseUnit']:
        """基于语义相似度去重"""
        # TODO: 实现语义去重（需要计算 embedding 相似度）
        raise NotImplementedError("Semantic deduplication not yet implemented")
```

---

### 4.4 Augmentors - 增强器

#### `postprocessors/augmentors/context.py`

```python
from ..base import BasePostprocessor

class ContextAugmentor(BasePostprocessor):
    """
    上下文增强器
    
    获取检索结果的相邻 units（prev/next），提供更完整的上下文。
    
    适用场景：
        - 需要更多上下文理解
        - 避免信息碎片化
        - 提升 LLM 理解能力
    
    Examples:
        >>> from zag.postprocessors.augmentors import ContextAugmentor
        >>> 
        >>> augmentor = ContextAugmentor(window_size=1)
        >>> augmented = augmentor.process(query, units)
    """
    
    def __init__(
        self, 
        window_size: int = 1,
        deduplicate: bool = True
    ):
        """
        初始化上下文增强器
        
        Args:
            window_size: 窗口大小（前后各取几个 unit）
            deduplicate: 是否去重
        """
        self.window_size = window_size
        self.deduplicate = deduplicate
    
    def process(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        增强 units 的上下文
        
        Args:
            query: 原始查询文本（本增强器不使用）
            units: 待增强的 units
            
        Returns:
            包含上下文的 units
        """
        augmented = []
        
        for unit in units:
            # 获取前面的 units
            current = unit
            for _ in range(self.window_size):
                if prev := current.get_prev():
                    augmented.append(prev)
                    current = prev
                else:
                    break
            
            # 添加当前 unit
            augmented.append(unit)
            
            # 获取后面的 units
            current = unit
            for _ in range(self.window_size):
                if next_unit := current.get_next():
                    augmented.append(next_unit)
                    current = next_unit
                else:
                    break
        
        # 去重（可选）
        if self.deduplicate:
            seen = set()
            result = []
            for unit in augmented:
                if unit.unit_id not in seen:
                    seen.add(unit.unit_id)
                    result.append(unit)
            return result
        
        return augmented
```

---

### 4.5 Compressors - 压缩器

#### `postprocessors/compressors/token.py`

```python
from ..base import BasePostprocessor

class TokenCompressor(BasePostprocessor):
    """
    Token 压缩器
    
    限制总 token 数量，避免超出 LLM 上下文窗口。
    
    压缩策略：
        - truncate: 截断（保留前 N 个 units）
        - smart: 智能压缩（优先保留高分结果）
    
    Examples:
        >>> from zag.postprocessors.compressors import TokenCompressor
        >>> 
        >>> compressor = TokenCompressor(max_tokens=4000)
        >>> compressed = compressor.process(query, units)
    """
    
    def __init__(
        self, 
        max_tokens: int = 4000,
        strategy: str = "smart",
        chars_per_token: float = 4.0
    ):
        """
        初始化 Token 压缩器
        
        Args:
            max_tokens: 最大 token 数量
            strategy: 压缩策略 ("truncate" 或 "smart")
            chars_per_token: 平均每个 token 的字符数（用于估算）
        """
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.chars_per_token = chars_per_token
    
    def process(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        压缩到指定 token 数量
        
        Args:
            query: 原始查询文本
            units: 待压缩的 units
            
        Returns:
            压缩后的 units
        """
        if self.strategy == "truncate":
            return self._truncate(units)
        else:  # smart
            return self._smart_compress(units)
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数量"""
        return int(len(text) / self.chars_per_token)
    
    def _truncate(self, units: list['BaseUnit']) -> list['BaseUnit']:
        """简单截断"""
        total_tokens = 0
        result = []
        
        for unit in units:
            tokens = self._estimate_tokens(str(unit.content))
            if total_tokens + tokens <= self.max_tokens:
                result.append(unit)
                total_tokens += tokens
            else:
                break
        
        return result
    
    def _smart_compress(self, units: list['BaseUnit']) -> list['BaseUnit']:
        """智能压缩：优先保留高分结果"""
        # 按分数排序（如果有分数）
        if units and hasattr(units[0], 'score') and units[0].score is not None:
            sorted_units = sorted(units, key=lambda x: x.score or 0, reverse=True)
        else:
            sorted_units = units
        
        # 然后截断
        return self._truncate(sorted_units)
```

---

## 5. 组合层 Postprocessor

### 5.1 `postprocessors/composite/chain.py`

```python
from ..base import BasePostprocessor

class ChainPostprocessor(BasePostprocessor):
    """
    链式后处理器
    
    按顺序执行多个后处理器，前一个的输出作为后一个的输入。
    类似于 Unix 管道：processor1 | processor2 | processor3
    
    特点：
        - 顺序执行
        - 短路机制：如果某一步返回空列表，提前结束
        - 灵活组合
    
    Examples:
        >>> from zag.postprocessors import (
        ...     ChainPostprocessor,
        ...     CrossEncoderReranker,
        ...     SimilarityFilter,
        ...     TokenCompressor
        ... )
        >>> 
        >>> chain = ChainPostprocessor([
        ...     CrossEncoderReranker(),
        ...     SimilarityFilter(threshold=0.7),
        ...     TokenCompressor(max_tokens=4000),
        ... ])
        >>> 
        >>> results = chain.process(query, units)
    """
    
    def __init__(self, processors: list[BasePostprocessor]):
        """
        初始化链式后处理器
        
        Args:
            processors: 后处理器列表，按执行顺序
        """
        if not processors:
            raise ValueError("Must provide at least one processor")
        self.processors = processors
    
    def process(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        依次执行所有后处理器
        
        Args:
            query: 原始查询文本
            units: 待处理的 units
            
        Returns:
            经过所有处理器后的 units
        """
        result = units
        
        for i, processor in enumerate(self.processors):
            if not result:
                # 短路：如果某一步返回空，提前结束
                break
            
            result = processor.process(query, result)
        
        return result
    
    async def aprocess(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        异步版本的链式处理
        
        Args:
            query: 原始查询文本
            units: 待处理的 units
            
        Returns:
            经过所有处理器后的 units
        """
        result = units
        
        for processor in self.processors:
            if not result:
                break
            result = await processor.aprocess(query, result)
        
        return result
```

---

### 5.2 `postprocessors/composite/conditional.py`

```python
from typing import Callable, Optional
from ..base import BasePostprocessor

class ConditionalPostprocessor(BasePostprocessor):
    """
    条件后处理器
    
    根据条件选择不同的后处理器执行。
    类似于 if-else 逻辑。
    
    适用场景：
        - 根据结果数量选择不同策略
        - 根据查询类型选择不同处理
        - 动态调整后处理流程
    
    Examples:
        >>> from zag.postprocessors import ConditionalPostprocessor
        >>> 
        >>> def need_reranking(query: str, units: list) -> bool:
        ...     return len(units) > 20
        >>> 
        >>> conditional = ConditionalPostprocessor(
        ...     condition=need_reranking,
        ...     true_processor=CrossEncoderReranker(),
        ...     false_processor=None,  # 不需要时直接返回
        ... )
        >>> 
        >>> results = conditional.process(query, units)
    """
    
    def __init__(
        self,
        condition: Callable[[str, list['BaseUnit']], bool],
        true_processor: BasePostprocessor,
        false_processor: Optional[BasePostprocessor] = None,
    ):
        """
        初始化条件后处理器
        
        Args:
            condition: 条件函数，接收 query 和 units，返回 bool
            true_processor: 条件为 True 时使用的处理器
            false_processor: 条件为 False 时使用的处理器（None 表示直接返回）
        """
        self.condition = condition
        self.true_processor = true_processor
        self.false_processor = false_processor
    
    def process(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """
        根据条件选择处理器
        
        Args:
            query: 原始查询文本
            units: 待处理的 units
            
        Returns:
            处理后的 units
        """
        if self.condition(query, units):
            return self.true_processor.process(query, units)
        elif self.false_processor:
            return self.false_processor.process(query, units)
        else:
            return units
```

---

## 6. 使用示例

### 6.1 基础使用：单个后处理器

```python
from zag.retrievers import VectorRetriever
from zag.postprocessors import CrossEncoderReranker, SimilarityFilter

# 1. 检索
retriever = VectorRetriever(vector_store=store)
units = retriever.retrieve("What is RAG?", top_k=100)

# 2. 重排序
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco")
reranked = reranker.process("What is RAG?", units)

# 3. 过滤
filter = SimilarityFilter(threshold=0.7)
filtered = filter.process("What is RAG?", reranked)
```

---

### 6.2 链式组合：多个后处理器

```python
from zag.postprocessors import (
    ChainPostprocessor,
    CrossEncoderReranker,
    SimilarityFilter,
    ContextAugmentor,
    TokenCompressor,
)

# 创建后处理链
postprocessor = ChainPostprocessor([
    # 步骤1: 重排序（使用更精确的模型）
    CrossEncoderReranker(model="cross-encoder/ms-marco", top_k=20),
    
    # 步骤2: 过滤低分结果
    SimilarityFilter(threshold=0.65),
    
    # 步骤3: 增强上下文（获取相邻 chunks）
    ContextAugmentor(window_size=1),
    
    # 步骤4: 压缩到 4000 tokens
    TokenCompressor(max_tokens=4000, strategy="smart"),
])

# 一步到位
query = "What is RAG?"
units = retriever.retrieve(query, top_k=100)
final_units = postprocessor.process(query, units)
```

---

### 6.3 条件组合：动态选择策略

```python
from zag.postprocessors import ConditionalPostprocessor

# 定义条件
def need_heavy_reranking(query: str, units: list) -> bool:
    """结果数量多时才使用重排序（计算开销大）"""
    return len(units) > 20

# 创建条件后处理器
conditional = ConditionalPostprocessor(
    condition=need_heavy_reranking,
    true_processor=CrossEncoderReranker(),  # 结果多：用重排序
    false_processor=SimilarityFilter(threshold=0.8),  # 结果少：只过滤
)

units = retriever.retrieve(query)
results = conditional.process(query, units)
```

---

### 6.4 嵌套组合：复杂流程

```python
# 场景：根据查询类型使用不同的后处理流程

def is_factual_query(query: str, units: list) -> bool:
    """判断是否是事实性查询"""
    factual_keywords = ["what", "when", "where", "who", "how many"]
    return any(kw in query.lower() for kw in factual_keywords)

# 事实性查询的处理流程
factual_chain = ChainPostprocessor([
    CrossEncoderReranker(top_k=10),  # 精确排序
    SimilarityFilter(threshold=0.8),  # 高阈值
    TokenCompressor(max_tokens=2000),  # 较少 tokens
])

# 开放性查询的处理流程
open_chain = ChainPostprocessor([
    SimilarityFilter(threshold=0.6),  # 低阈值，保留更多结果
    ContextAugmentor(window_size=2),  # 更多上下文
    TokenCompressor(max_tokens=6000),  # 更多 tokens
])

# 根据查询类型选择流程
adaptive_processor = ConditionalPostprocessor(
    condition=is_factual_query,
    true_processor=factual_chain,
    false_processor=open_chain,
)

results = adaptive_processor.process(query, units)
```

---

### 6.5 包装成 Retriever：统一接口

```python
from zag.retrievers.base import BaseRetriever

class PostprocessedRetriever(BaseRetriever):
    """
    带后处理的 Retriever
    
    将 Retriever 和 Postprocessor 组合，对外提供统一的 Retriever 接口。
    """
    
    def __init__(
        self, 
        retriever: BaseRetriever,
        postprocessor: BasePostprocessor,
    ):
        self.retriever = retriever
        self.postprocessor = postprocessor
    
    def retrieve(self, query: str, top_k: int = 10, filters=None):
        # 先检索（取更多结果）
        units = self.retriever.retrieve(query, top_k=top_k*10, filters=filters)
        
        # 后处理
        processed = self.postprocessor.process(query, units)
        
        # 返回 top_k 结果
        return processed[:top_k]

# 使用
retriever = PostprocessedRetriever(
    retriever=VectorRetriever(vector_store=store),
    postprocessor=ChainPostprocessor([
        CrossEncoderReranker(),
        SimilarityFilter(threshold=0.7),
        TokenCompressor(max_tokens=4000),
    ])
)

# 一步到位，外部看起来就是一个普通的 Retriever
results = retriever.retrieve("What is machine learning?", top_k=5)
```

---

## 7. 扩展指南

### 7.1 添加新的基础后处理器

**步骤**：

1. 在对应的子模块目录下创建新文件
2. 继承 `BasePostprocessor`
3. 实现 `process()` 方法
4. 在子模块的 `__init__.py` 中导出
5. 在主 `__init__.py` 中导出

**示例：添加 MetadataFilter**

```python
# postprocessors/filters/metadata.py
from ..base import BasePostprocessor

class MetadataFilter(BasePostprocessor):
    """
    元数据过滤器
    
    根据元数据条件过滤 units。
    """
    
    def __init__(self, conditions: dict):
        """
        Args:
            conditions: 过滤条件字典
                例如: {"source": "wiki", "year": 2023}
        """
        self.conditions = conditions
    
    def process(self, query: str, units: list['BaseUnit']) -> list['BaseUnit']:
        """根据元数据过滤"""
        result = []
        for unit in units:
            if self._match_conditions(unit):
                result.append(unit)
        return result
    
    def _match_conditions(self, unit: 'BaseUnit') -> bool:
        """检查是否满足所有条件"""
        if not unit.metadata or not unit.metadata.custom:
            return False
        
        for key, value in self.conditions.items():
            if unit.metadata.custom.get(key) != value:
                return False
        return True
```

---

### 7.2 添加新的组合后处理器

**示例：添加 ParallelPostprocessor**

```python
# postprocessors/composite/parallel.py
from typing import Callable
from ..base import BasePostprocessor

class ParallelPostprocessor(BasePostprocessor):
    """
    并行后处理器
    
    同时执行多个后处理器，然后合并结果。
    
    适用场景：
        - 多个独立的处理器
        - 需要合并不同处理器的结果
    """
    
    def __init__(
        self,
        processors: list[BasePostprocessor],
        merger: Callable[[list[list['BaseUnit']]], list['BaseUnit']],
    ):
        """
        Args:
            processors: 并行执行的处理器列表
            merger: 合并函数，接收多个结果列表，返回合并后的结果
        """
        self.processors = processors
        self.merger = merger
    
    def process(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """并行执行并合并结果"""
        results = []
        for processor in self.processors:
            results.append(processor.process(query, units))
        return self.merger(results)
    
    async def aprocess(
        self, 
        query: str,
        units: list['BaseUnit']
    ) -> list['BaseUnit']:
        """真正的并行执行（异步）"""
        import asyncio
        
        tasks = [
            processor.aprocess(query, units) 
            for processor in self.processors
        ]
        results = await asyncio.gather(*tasks)
        return self.merger(list(results))

# 使用示例
def merge_by_score(results: list[list['BaseUnit']]) -> list['BaseUnit']:
    """合并多个结果，按分数去重并排序"""
    all_units = {}
    for units in results:
        for unit in units:
            if unit.unit_id not in all_units:
                all_units[unit.unit_id] = unit
            else:
                # 保留更高的分数
                if unit.score > all_units[unit.unit_id].score:
                    all_units[unit.unit_id] = unit
    
    return sorted(all_units.values(), key=lambda x: x.score, reverse=True)

parallel = ParallelPostprocessor(
    processors=[
        CrossEncoderReranker(model="model-a"),
        CrossEncoderReranker(model="model-b"),
    ],
    merger=merge_by_score
)
```

---

## 8. 设计决策总结

### 8.1 为什么 Reranker 归入 Postprocessors？

| 方面 | 独立 Reranker 模块 | Reranker 作为 Postprocessor |
|------|-------------------|----------------------------|
| **语义** | 强调 Reranker 的特殊性 | ✅ 统一的后处理抽象 |
| **职责** | Reranker 单独一层 | ✅ 后处理层统一管理 |
| **组合** | 需要特殊处理 | ✅ 自然组合，接口一致 |
| **学习成本** | 需要理解多个概念 | ✅ 一个概念，多种实现 |
| **扩展性** | 每种处理都要单独一层？ | ✅ 统一扩展点 |

**结论**：Reranker 作为特殊的 Postprocessor，在保持语义清晰的同时，提供了更好的组合性和扩展性。

---

### 8.2 架构分层

```
检索 + 后处理 = 完整的检索流程

┌──────────────┐
│  Retriever   │  从存储中获取候选结果
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Postprocessor│  优化和精炼结果
└──────────────┘
       │
       ↓ rerank, filter, augment, compress...
       │
       ↓
   最终结果
```

**关键点**：
- Retriever：负责检索（retrieve）
- Postprocessor：负责后处理（post-process）
- 两者通过统一的 `list[BaseUnit]` 接口连接

---

### 8.3 与 LlamaIndex 对比

| 维度 | 本设计 | LlamaIndex |
|------|--------|-----------|
| **Reranker 位置** | Postprocessors 的子模块 | 独立的 `postprocessor.rerank` |
| **统一接口** | ✅ BasePostprocessor | ⚠️ 多种接口混用 |
| **组合方式** | ✅ ChainPostprocessor | 手动组合 |
| **目录结构** | ✅ 按功能分类清晰 | ⚠️ 分散在多处 |

---

## 9. 常见问题

### Q1: 为什么不让 Postprocessor 直接访问 Storage？

**A**: 职责分离原则。
- Retriever：负责从存储中检索
- Postprocessor：负责处理已有结果
- 这样职责清晰，便于测试和维护

如果需要访问存储（如 ContextAugmentor 获取相邻 chunks），应该：
1. 通过 Unit 的 relations 获取（已加载到内存）
2. 或者在 Retriever 阶段就获取相关数据

### Q2: ChainPostprocessor 和 Unix 管道有什么区别？

**A**: 概念相似，但实现不同。
- Unix 管道：流式处理，数据边产生边消费
- ChainPostprocessor：批量处理，前一步完成后才开始下一步

如果需要流式处理，可以扩展实现 `StreamingChainPostprocessor`。

### Q3: 异步支持如何实现？

**A**: 
- 默认实现：`aprocess()` 调用同步的 `process()`
- 真正的异步：子类可以重写 `aprocess()` 实现真正的异步 I/O
- 组合器支持：`ChainPostprocessor` 等提供了异步版本

---

## 10. 实现检查清单

开始实现前，请确认：

- [ ] 理解 Retriever 和 Postprocessor 的职责分离
- [ ] 理解为什么 Reranker 归入 Postprocessors
- [ ] 理解基础层和组合层的区别
- [ ] 熟悉 `BasePostprocessor` 接口
- [ ] 了解各种后处理器的适用场景
- [ ] 知道如何使用 ChainPostprocessor 组合
- [ ] 知道如何扩展新的后处理器

---

## 附录：实现优先级建议

### Phase 1: 核心基础（必需）
1. ✅ `base.py` - BasePostprocessor
2. ✅ `rerankers/base.py` - BaseReranker
3. ✅ `composite/chain.py` - ChainPostprocessor

### Phase 2: 常用处理器（推荐）
4. ✅ `rerankers/cross_encoder.py` - CrossEncoderReranker
5. ✅ `filters/similarity.py` - SimilarityFilter
6. ✅ `filters/deduplicator.py` - Deduplicator
7. ✅ `compressors/token.py` - TokenCompressor

### Phase 3: 增强功能（可选）
8. `augmentors/context.py` - ContextAugmentor
9. `composite/conditional.py` - ConditionalPostprocessor
10. `transformers/summarizer.py` - Summarizer

### Phase 4: 高级功能（按需）
11. `rerankers/llm_reranker.py` - LLMReranker
12. `filters/metadata.py` - MetadataFilter
13. `composite/parallel.py` - ParallelPostprocessor

---

## 更新日志

- **2026-01-05**: 初始版本，定义 Postprocessor 架构和设计原则
