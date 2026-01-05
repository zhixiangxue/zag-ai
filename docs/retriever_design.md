# Retriever 架构设计文档

## 目录

- [1. 设计原则](#1-设计原则)
- [2. 架构概览](#2-架构概览)
- [3. 目录结构](#3-目录结构)
- [4. 基础层 Retriever](#4-基础层-retriever)
- [5. 组合层 Retriever](#5-组合层-retriever)
- [6. 使用示例](#6-使用示例)
- [7. 扩展指南](#7-扩展指南)
- [8. 测试建议](#8-测试建议)

---

## 1. 设计原则

### 1.1 职责分离

**基础层（Basic Layer）**：
- 封装 Storage 调用逻辑
- 处理单一数据源的检索
- 负责查询预处理（如向量计算）
- 负责结果后处理（如获取完整文档）

**组合层（Composite Layer）**：
- 组合多个 Retriever 实例
- 实现检索策略和编排逻辑
- 不直接调用 Storage
- 只依赖 BaseRetriever 抽象接口

### 1.2 依赖关系

```
组合层 Retriever
    ↓ 依赖
基础层 Retriever
    ↓ 依赖
Storage Layer (VectorStoreAdapter, KeywordStoreAdapter, DocStoreAdapter)
```

**关键原则**：
- ✅ 基础层 Retriever 可以直接使用
- ✅ 组合层 Retriever 可以组合任意 Retriever（包括基础层和组合层）
- ✅ 所有 Retriever 实现统一的 `BaseRetriever` 接口
- ✅ 通过依赖注入实现灵活组合

### 1.3 为什么需要基础层 Retriever？

虽然 Storage 已经提供了 `query()` 方法，但基础层 Retriever 仍然是必要的：

**职责清晰**：
- Storage：纯粹的数据存储和查询（CRUD）
- 基础层 Retriever：封装完整的检索逻辑（预处理 + Storage 调用 + 后处理）
- 组合层 Retriever：检索策略和编排

**代码复用**：
- 避免在每个组合层 Retriever 中重复写查询向量计算、文档获取等逻辑
- 基础层 Retriever 封装了这些通用逻辑，提高复用性

**接口统一**：
- 组合层 Retriever 可以统一操作 `BaseRetriever` 接口
- 不需要关心底层是向量检索、关键词检索还是外部服务

**灵活组合**：
- 基础层 Retriever 可以独立使用
- 也可以作为组合层 Retriever 的组件
- 支持嵌套组合（组合层 Retriever 可以组合其他组合层 Retriever）

---

## 2. 架构概览

### 2.1 两层架构

```
┌─────────────────────────────────────────────────────────┐
│           组合层 Retriever (Composite Layer)            │
│                                                          │
│  QueryFusionRetriever    RouterRetriever                │
│  - 多路检索融合           - 路由选择                     │
│                                                          │
│  RecursiveRetriever                                     │
│  - 递归检索                                              │
│                                                          │
│  特征：组合多个 Retriever，实现编排逻辑                  │
└─────────────────────────────────────────────────────────┘
                         ↓ 依赖
┌─────────────────────────────────────────────────────────┐
│           基础层 Retriever (Basic Layer)                │
│                                                          │
│  VectorRetriever        KeywordRetriever                │
│  - 向量检索              - 关键词检索                    │
│                                                          │
│  HybridRetriever                                        │
│  - 混合检索（单实例内融合向量和关键词）                  │
│                                                          │
│  特征：封装 Storage 调用，处理单一检索逻辑               │
└─────────────────────────────────────────────────────────┘
                         ↓ 依赖
┌─────────────────────────────────────────────────────────┐
│              Storage Layer (存储层)                      │
│                                                          │
│  VectorStoreAdapter   KeywordStoreAdapter               │
│  DocStoreAdapter                                        │
│                                                          │
│  特征：提供数据存储和查询能力（CRUD）                    │
└─────────────────────────────────────────────────────────┘
```

### 2.2 核心接口

```python
class BaseRetriever(ABC):
    """所有 Retriever 的抽象基类"""
    
    @abstractmethod
    def retrieve(self, query: Query) -> List[Node]:
        """检索节点"""
        pass
    
    async def aretrieve(self, query: Query) -> List[Node]:
        """异步检索（可选）"""
        return self.retrieve(query)
```

---

## 3. 目录结构

```
retriever/
├── __init__.py                    # 统一导出所有 Retriever
├── base.py                        # BaseRetriever 抽象基类
│
├── basic/                         # 基础层（原子层）
│   ├── __init__.py
│   ├── vector_retriever.py        # VectorRetriever
│   ├── keyword_retriever.py       # KeywordRetriever
│   └── hybrid_retriever.py        # HybridRetriever
│
└── composite/                     # 组合层（编排层）
    ├── __init__.py
    ├── fusion_retriever.py        # QueryFusionRetriever
    ├── router_retriever.py        # RouterRetriever
    └── recursive_retriever.py     # RecursiveRetriever
```

**命名规范**：
- 基础层文件：`{type}_retriever.py`（如 `vector_retriever.py`）
- 组合层文件：`{strategy}_retriever.py`（如 `fusion_retriever.py`）
- 类名：使用 PascalCase，以 `Retriever` 结尾

---

## 4. 基础层 Retriever

### 4.1 `retriever/base.py`：抽象基类

```python
from abc import ABC, abstractmethod
from typing import List

class BaseRetriever(ABC):
    """所有 Retriever 的抽象基类"""
    
    @abstractmethod
    def retrieve(self, query: Query) -> List[Node]:
        """检索节点
        
        Args:
            query: 查询对象
            
        Returns:
            检索到的节点列表，按相关性排序
        """
        pass
    
    async def aretrieve(self, query: Query) -> List[Node]:
        """异步检索（可选实现）
        
        Args:
            query: 查询对象
            
        Returns:
            检索到的节点列表，按相关性排序
        """
        return self.retrieve(query)
```

---

### 4.2 `retriever/basic/vector_retriever.py`：向量检索器

```python
from retriever.base import BaseRetriever
from typing import List

class VectorRetriever(BaseRetriever):
    """向量检索器（基础层）
    
    封装向量存储的检索逻辑：
    1. 计算查询向量
    2. 调用 VectorStore.query()
    3. 获取完整文档
    4. 附加相关性分数
    """
    
    def __init__(
        self, 
        vector_store: VectorStoreAdapter,
        doc_store: DocStoreAdapter,
        embedder: Embedder,
        top_k: int = 10,
    ):
        """初始化向量检索器
        
        Args:
            vector_store: 向量存储适配器
            doc_store: 文档存储适配器
            embedder: 向量化器
            top_k: 返回的最大结果数量
        """
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.embedder = embedder
        self.top_k = top_k
    
    def retrieve(self, query: Query) -> List[Node]:
        """执行向量检索
        
        Args:
            query: 查询对象
            
        Returns:
            检索到的节点列表，按相关性分数排序
        """
        # 1. 计算查询向量
        query_vector = self.embedder.embed(query.text)
        
        # 2. 调用 Storage 检索
        raw_results = self.vector_store.query(
            query_vector=query_vector,
            top_k=self.top_k,
            filters=query.filters,
        )
        
        # 3. 获取完整节点
        node_ids = [r['id'] for r in raw_results]
        nodes = self.doc_store.get_by_ids(node_ids)
        
        # 4. 附加分数
        for node, result in zip(nodes, raw_results):
            node.score = result['score']
        
        return nodes
```

---

### 4.3 `retriever/basic/keyword_retriever.py`：关键词检索器

```python
from retriever.base import BaseRetriever
from typing import List

class KeywordRetriever(BaseRetriever):
    """关键词检索器（基础层）
    
    封装全文搜索的检索逻辑：
    1. 调用 KeywordStore.query()
    2. 获取完整文档
    3. 附加相关性分数
    """
    
    def __init__(
        self, 
        keyword_store: KeywordStoreAdapter,
        doc_store: DocStoreAdapter,
        top_k: int = 10,
    ):
        """初始化关键词检索器
        
        Args:
            keyword_store: 关键词存储适配器（如 Meilisearch、Elasticsearch）
            doc_store: 文档存储适配器
            top_k: 返回的最大结果数量
        """
        self.keyword_store = keyword_store
        self.doc_store = doc_store
        self.top_k = top_k
    
    def retrieve(self, query: Query) -> List[Node]:
        """执行关键词检索
        
        Args:
            query: 查询对象
            
        Returns:
            检索到的节点列表，按相关性分数排序
        """
        # 1. 调用 Storage 检索
        raw_results = self.keyword_store.query(
            text=query.text,
            top_k=self.top_k,
            filters=query.filters,
        )
        
        # 2. 获取完整节点
        node_ids = [r['id'] for r in raw_results]
        nodes = self.doc_store.get_by_ids(node_ids)
        
        # 3. 附加分数
        for node, result in zip(nodes, raw_results):
            node.score = result['score']
        
        return nodes
```

---

### 4.4 `retriever/basic/hybrid_retriever.py`：混合检索器

```python
from retriever.base import BaseRetriever
from typing import List

class HybridRetriever(BaseRetriever):
    """混合检索器（基础层）
    
    在单个实例中融合向量检索和关键词检索：
    1. 同时执行向量检索和关键词检索
    2. 使用 RRF 算法融合结果
    3. 返回融合后的节点列表
    
    注意：这是基础层 Retriever，因为它封装的是单个实例内的检索逻辑，
    而非组合多个 Retriever。
    """
    
    def __init__(
        self, 
        vector_store: VectorStoreAdapter,
        keyword_store: KeywordStoreAdapter,
        doc_store: DocStoreAdapter,
        embedder: Embedder,
        top_k: int = 10,
        alpha: float = 0.5,  # 向量检索权重
    ):
        """初始化混合检索器
        
        Args:
            vector_store: 向量存储适配器
            keyword_store: 关键词存储适配器
            doc_store: 文档存储适配器
            embedder: 向量化器
            top_k: 返回的最大结果数量
            alpha: 向量检索权重（0-1，1-alpha 为关键词检索权重）
        """
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.doc_store = doc_store
        self.embedder = embedder
        self.top_k = top_k
        self.alpha = alpha
    
    def retrieve(self, query: Query) -> List[Node]:
        """执行混合检索
        
        Args:
            query: 查询对象
            
        Returns:
            融合后的节点列表，按融合分数排序
        """
        # 1. 向量检索
        query_vector = self.embedder.embed(query.text)
        vector_results = self.vector_store.query(
            query_vector=query_vector,
            top_k=self.top_k * 2,  # 获取更多结果用于融合
            filters=query.filters,
        )
        
        # 2. 关键词检索
        keyword_results = self.keyword_store.query(
            text=query.text,
            top_k=self.top_k * 2,
            filters=query.filters,
        )
        
        # 3. 融合结果（RRF 算法）
        merged_ids = self._reciprocal_rank_fusion(vector_results, keyword_results)
        
        # 4. 获取完整节点
        nodes = self.doc_store.get_by_ids(merged_ids[:self.top_k])
        
        return nodes
    
    def _reciprocal_rank_fusion(
        self, 
        vector_results: List[dict], 
        keyword_results: List[dict]
    ) -> List[str]:
        """Reciprocal Rank Fusion (RRF) 算法
        
        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            
        Returns:
            融合后的节点 ID 列表，按融合分数排序
        """
        k = 60.0  # RRF 参数
        fused_scores = {}
        
        # 向量检索结果
        for rank, result in enumerate(vector_results):
            node_id = result['id']
            fused_scores[node_id] = fused_scores.get(node_id, 0.0) + \
                                    self.alpha * (1.0 / (rank + k))
        
        # 关键词检索结果
        for rank, result in enumerate(keyword_results):
            node_id = result['id']
            fused_scores[node_id] = fused_scores.get(node_id, 0.0) + \
                                    (1 - self.alpha) * (1.0 / (rank + k))
        
        # 排序
        sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in sorted_ids]
```

---

## 5. 组合层 Retriever

### 5.1 `retriever/composite/fusion_retriever.py`：查询融合检索器

```python
from retriever.base import BaseRetriever
from typing import List
from enum import Enum

class FusionMode(Enum):
    """融合模式枚举"""
    SIMPLE = "simple"                    # 简单融合（去重，保留最高分）
    RECIPROCAL_RANK = "reciprocal_rank"  # Reciprocal Rank Fusion
    RELATIVE_SCORE = "relative_score"    # 相对分数融合


class QueryFusionRetriever(BaseRetriever):
    """查询融合检索器（组合层）
    
    组合多个 Retriever，融合它们的检索结果：
    1. 调用多个 Retriever 进行检索
    2. 根据融合模式合并结果
    3. 返回融合后的节点列表
    
    支持的融合模式：
    - SIMPLE: 简单去重，保留每个节点的最高分数
    - RECIPROCAL_RANK: RRF 算法，适合融合不同类型的检索
    - RELATIVE_SCORE: 相对分数融合，适合相同类型的检索
    """
    
    def __init__(
        self, 
        retrievers: List[BaseRetriever],
        mode: FusionMode = FusionMode.SIMPLE,
        top_k: int = 10,
        retriever_weights: List[float] = None,
    ):
        """初始化查询融合检索器
        
        Args:
            retrievers: Retriever 列表
            mode: 融合模式
            top_k: 返回的最大结果数量
            retriever_weights: 每个 Retriever 的权重（仅在 RELATIVE_SCORE 模式下使用）
        """
        self.retrievers = retrievers
        self.mode = mode
        self.top_k = top_k
        
        # 处理权重
        if retriever_weights is None:
            self.retriever_weights = [1.0 / len(retrievers)] * len(retrievers)
        else:
            total_weight = sum(retriever_weights)
            self.retriever_weights = [w / total_weight for w in retriever_weights]
    
    def retrieve(self, query: Query) -> List[Node]:
        """执行融合检索
        
        Args:
            query: 查询对象
            
        Returns:
            融合后的节点列表，按融合分数排序
        """
        # 1. 调用多个 Retriever 检索
        results = {}
        for i, retriever in enumerate(self.retrievers):
            results[i] = retriever.retrieve(query)
        
        # 2. 根据模式融合结果
        if self.mode == FusionMode.RECIPROCAL_RANK:
            merged_nodes = self._reciprocal_rank_fusion(results)
        elif self.mode == FusionMode.RELATIVE_SCORE:
            merged_nodes = self._relative_score_fusion(results)
        else:
            merged_nodes = self._simple_fusion(results)
        
        return merged_nodes[:self.top_k]
    
    def _reciprocal_rank_fusion(self, results: dict) -> List[Node]:
        """RRF 算法融合
        
        适用场景：融合不同类型的检索结果（如向量检索 + 关键词检索）
        """
        k = 60.0
        fused_scores = {}
        hash_to_node = {}
        
        for nodes in results.values():
            for rank, node in enumerate(sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)):
                hash_to_node[node.id] = node
                if node.id not in fused_scores:
                    fused_scores[node.id] = 0.0
                fused_scores[node.id] += 1.0 / (rank + k)
        
        # 排序
        reranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回节点
        reranked_nodes = []
        for node_id, score in reranked_ids:
            node = hash_to_node[node_id]
            node.score = score
            reranked_nodes.append(node)
        
        return reranked_nodes
    
    def _relative_score_fusion(self, results: dict) -> List[Node]:
        """相对分数融合
        
        适用场景：融合相同类型的检索结果（如多个向量数据库）
        """
        # MinMax 归一化每个 Retriever 的分数
        for i, nodes in results.items():
            if not nodes:
                continue
            
            scores = [node.score or 0.0 for node in nodes]
            min_score = min(scores)
            max_score = max(scores)
            
            # 归一化
            for node in nodes:
                if max_score == min_score:
                    node.score = 1.0 if max_score > 0 else 0.0
                else:
                    node.score = (node.score - min_score) / (max_score - min_score)
                
                # 加权
                node.score *= self.retriever_weights[i]
        
        # 合并节点
        all_nodes = {}
        for nodes in results.values():
            for node in nodes:
                if node.id in all_nodes:
                    all_nodes[node.id].score += node.score
                else:
                    all_nodes[node.id] = node
        
        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)
    
    def _simple_fusion(self, results: dict) -> List[Node]:
        """简单融合：去重，保留最高分"""
        all_nodes = {}
        for nodes in results.values():
            for node in nodes:
                if node.id in all_nodes:
                    all_nodes[node.id].score = max(
                        all_nodes[node.id].score or 0.0,
                        node.score or 0.0
                    )
                else:
                    all_nodes[node.id] = node
        
        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)
```

---

### 5.2 `retriever/composite/router_retriever.py`：路由检索器

```python
from retriever.base import BaseRetriever
from typing import List, Optional

class RouterRetriever(BaseRetriever):
    """路由检索器（组合层）
    
    根据查询特征选择合适的 Retriever：
    1. 使用选择器（Selector）分析查询
    2. 选择最适合的 Retriever
    3. 调用选中的 Retriever 进行检索
    
    使用场景：
    - 根据查询语言选择不同的检索器
    - 根据查询类型（事实查询 vs 语义查询）选择不同的策略
    - 根据数据源特征动态选择检索器
    """
    
    def __init__(
        self, 
        retrievers: List[BaseRetriever],
        selector: BaseSelector,
        select_multi: bool = False,
    ):
        """初始化路由检索器
        
        Args:
            retrievers: Retriever 列表
            selector: 选择器，用于选择合适的 Retriever
            select_multi: 是否允许选择多个 Retriever
        """
        self.retrievers = retrievers
        self.selector = selector
        self.select_multi = select_multi
    
    def retrieve(self, query: Query) -> List[Node]:
        """执行路由检索
        
        Args:
            query: 查询对象
            
        Returns:
            检索到的节点列表
        """
        # 1. 选择 Retriever
        if self.select_multi:
            selected_indices = self.selector.select_multi(query, self.retrievers)
        else:
            selected_idx = self.selector.select(query, self.retrievers)
            selected_indices = [selected_idx]
        
        # 2. 调用选中的 Retriever
        all_nodes = {}
        for idx in selected_indices:
            selected_retriever = self.retrievers[idx]
            nodes = selected_retriever.retrieve(query)
            
            # 去重
            for node in nodes:
                if node.id not in all_nodes:
                    all_nodes[node.id] = node
        
        return list(all_nodes.values())
```

---

### 5.3 `retriever/composite/recursive_retriever.py`：递归检索器

```python
from retriever.base import BaseRetriever
from typing import List, Dict

class RecursiveRetriever(BaseRetriever):
    """递归检索器（组合层）
    
    处理多层索引结构，递归检索：
    1. 从根 Retriever 开始检索
    2. 如果检索到 IndexNode，递归查询对应的 Retriever
    3. 如果检索到 TextNode，直接返回
    
    使用场景：
    - 多层索引结构（如摘要索引 → 文档索引）
    - 知识图谱检索
    - 多跳推理
    """
    
    def __init__(
        self, 
        root_id: str,
        retriever_dict: Dict[str, BaseRetriever],
    ):
        """初始化递归检索器
        
        Args:
            root_id: 根 Retriever 的 ID
            retriever_dict: Retriever 字典，key 为 ID，value 为 Retriever 实例
        """
        if root_id not in retriever_dict:
            raise ValueError(f"Root id {root_id} not in retriever_dict")
        
        self.root_id = root_id
        self.retriever_dict = retriever_dict
    
    def retrieve(self, query: Query) -> List[Node]:
        """执行递归检索
        
        Args:
            query: 查询对象
            
        Returns:
            检索到的节点列表
        """
        retrieved_nodes, _ = self._retrieve_recursive(query, self.root_id)
        return retrieved_nodes
    
    def _retrieve_recursive(
        self, 
        query: Query, 
        retriever_id: str,
        cur_similarity: float = 1.0
    ) -> tuple[List[Node], List[Node]]:
        """递归检索实现
        
        Args:
            query: 查询对象
            retriever_id: 当前 Retriever ID
            cur_similarity: 当前相似度
            
        Returns:
            (主要节点列表, 附加节点列表)
        """
        retriever = self.retriever_dict[retriever_id]
        nodes = retriever.retrieve(query)
        
        nodes_to_add = []
        additional_nodes = []
        
        for node in nodes:
            if isinstance(node, IndexNode):
                # 递归查询
                cur_nodes, cur_additional = self._retrieve_recursive(
                    query,
                    node.index_id,
                    node.score or cur_similarity
                )
                nodes_to_add.extend(cur_nodes)
                additional_nodes.extend(cur_additional)
            else:
                # 直接添加
                nodes_to_add.append(node)
        
        return nodes_to_add, additional_nodes
```

---

## 6. 使用示例

### 6.1 基础层 Retriever 使用

```python
from retriever import VectorRetriever, KeywordRetriever, HybridRetriever

# 示例 1：向量检索
vector_retriever = VectorRetriever(
    vector_store=chroma_adapter,
    doc_store=doc_store,
    embedder=embedder,
    top_k=10,
)

query = Query(text="What is RAG?")
nodes = vector_retriever.retrieve(query)

# 示例 2：关键词检索
keyword_retriever = KeywordRetriever(
    keyword_store=meilisearch_adapter,
    doc_store=doc_store,
    top_k=10,
)

nodes = keyword_retriever.retrieve(query)

# 示例 3：混合检索
hybrid_retriever = HybridRetriever(
    vector_store=chroma_adapter,
    keyword_store=meilisearch_adapter,
    doc_store=doc_store,
    embedder=embedder,
    top_k=10,
    alpha=0.7,  # 70% 向量检索，30% 关键词检索
)

nodes = hybrid_retriever.retrieve(query)
```

---

### 6.2 组合层 Retriever 使用

```python
from retriever import QueryFusionRetriever, FusionMode, RouterRetriever

# 示例 1：融合多个向量数据库
fusion_retriever = QueryFusionRetriever(
    retrievers=[
        VectorRetriever(chroma_adapter, doc_store, embedder),
        VectorRetriever(pinecone_adapter, doc_store, embedder),
    ],
    mode=FusionMode.RECIPROCAL_RANK,
    top_k=10,
)

nodes = fusion_retriever.retrieve(query)

# 示例 2：融合向量检索和关键词检索
hybrid_fusion = QueryFusionRetriever(
    retrievers=[
        VectorRetriever(vector_store, doc_store, embedder),
        KeywordRetriever(keyword_store, doc_store),
    ],
    mode=FusionMode.RECIPROCAL_RANK,
    top_k=10,
)

nodes = hybrid_fusion.retrieve(query)

# 示例 3：路由检索
router_retriever = RouterRetriever(
    retrievers=[
        VectorRetriever(vector_store, doc_store, embedder),
        KeywordRetriever(keyword_store, doc_store),
    ],
    selector=LLMSelector(llm),  # 使用 LLM 选择合适的 Retriever
    select_multi=False,
)

nodes = router_retriever.retrieve(query)
```

---

### 6.3 嵌套组合

```python
# 组合层 Retriever 可以嵌套组合

# 第一层：混合检索
hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    keyword_store=keyword_store,
    doc_store=doc_store,
    embedder=embedder,
)

# 第二层：融合多个混合检索
nested_fusion = QueryFusionRetriever(
    retrievers=[
        hybrid_retriever,
        VectorRetriever(another_vector_store, doc_store, embedder),
    ],
    mode=FusionMode.RECIPROCAL_RANK,
)

nodes = nested_fusion.retrieve(query)
```

---

### 6.4 切换数据库

```python
# 基础层 Retriever 通过依赖注入切换数据库

# 使用 Chroma
vector_retriever_chroma = VectorRetriever(
    vector_store=chroma_adapter,
    doc_store=doc_store,
    embedder=embedder,
)

# 切换到 Pinecone
vector_retriever_pinecone = VectorRetriever(
    vector_store=pinecone_adapter,
    doc_store=doc_store,
    embedder=embedder,
)

# 融合两个数据库的结果
fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever_chroma, vector_retriever_pinecone],
    mode=FusionMode.RELATIVE_SCORE,
)
```

---

## 7. 扩展指南

### 7.1 添加新的基础层 Retriever

**步骤**：

1. 在 `retriever/basic/` 目录下创建新文件
2. 继承 `BaseRetriever`
3. 实现 `retrieve()` 方法
4. 在 `retriever/basic/__init__.py` 中导出
5. 在 `retriever/__init__.py` 中导出

**示例：添加 BM25 Retriever**

```python
# retriever/basic/bm25_retriever.py
from retriever.base import BaseRetriever

class BM25Retriever(BaseRetriever):
    """BM25 算法检索器"""
    
    def __init__(self, bm25_index, doc_store, top_k=10):
        self.bm25_index = bm25_index
        self.doc_store = doc_store
        self.top_k = top_k
    
    def retrieve(self, query: Query) -> List[Node]:
        # 1. BM25 检索
        scores = self.bm25_index.get_scores(query.text)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        
        # 2. 获取节点
        node_ids = [self.bm25_index.doc_ids[i] for i in top_indices]
        nodes = self.doc_store.get_by_ids(node_ids)
        
        # 3. 附加分数
        for node, idx in zip(nodes, top_indices):
            node.score = scores[idx]
        
        return nodes
```

---

### 7.2 添加新的组合层 Retriever

**步骤**：

1. 在 `retriever/composite/` 目录下创建新文件
2. 继承 `BaseRetriever`
3. 接收 `List[BaseRetriever]` 参数
4. 实现组合逻辑
5. 在 `retriever/composite/__init__.py` 中导出
6. 在 `retriever/__init__.py` 中导出

**示例：添加 Rerank Retriever**

```python
# retriever/composite/rerank_retriever.py
from retriever.base import BaseRetriever

class RerankRetriever(BaseRetriever):
    """重排序检索器（组合层）
    
    先用基础 Retriever 粗排，再用 Reranker 精排
    """
    
    def __init__(
        self, 
        retriever: BaseRetriever,
        reranker: Reranker,
        top_k: int = 10,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
    
    def retrieve(self, query: Query) -> List[Node]:
        # 1. 粗排
        nodes = self.retriever.retrieve(query)
        
        # 2. 精排
        reranked_nodes = self.reranker.rerank(query, nodes)
        
        return reranked_nodes[:self.top_k]
```

---

### 7.3 实现自定义选择器

```python
# selector/llm_selector.py
class LLMSelector(BaseSelector):
    """使用 LLM 选择合适的 Retriever"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def select(self, query: Query, retrievers: List[BaseRetriever]) -> int:
        # 构建提示词
        prompt = f"""
        Query: {query.text}
        
        Available retrievers:
        0. Vector search (semantic similarity)
        1. Keyword search (exact matching)
        
        Which retriever is most suitable? Answer with the index only.
        """
        
        response = self.llm.complete(prompt)
        return int(response.text.strip())
```

---

## 8. 测试建议

### 8.1 基础层 Retriever 测试

```python
import pytest
from unittest.mock import Mock

def test_vector_retriever():
    # Mock 依赖
    mock_vector_store = Mock()
    mock_doc_store = Mock()
    mock_embedder = Mock()
    
    # 设置返回值
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
    mock_vector_store.query.return_value = [
        {'id': '1', 'score': 0.9},
        {'id': '2', 'score': 0.8},
    ]
    mock_doc_store.get_by_ids.return_value = [
        Node(id='1', text='Node 1'),
        Node(id='2', text='Node 2'),
    ]
    
    # 创建 Retriever
    retriever = VectorRetriever(
        vector_store=mock_vector_store,
        doc_store=mock_doc_store,
        embedder=mock_embedder,
        top_k=2,
    )
    
    # 执行检索
    query = Query(text="test query")
    nodes = retriever.retrieve(query)
    
    # 验证
    assert len(nodes) == 2
    assert nodes[0].id == '1'
    assert nodes[0].score == 0.9
    mock_embedder.embed.assert_called_once_with("test query")
    mock_vector_store.query.assert_called_once()
```

---

### 8.2 组合层 Retriever 测试

```python
def test_query_fusion_retriever():
    # Mock Retriever
    mock_retriever1 = Mock()
    mock_retriever2 = Mock()
    
    mock_retriever1.retrieve.return_value = [
        Node(id='1', score=0.9),
        Node(id='2', score=0.8),
    ]
    mock_retriever2.retrieve.return_value = [
        Node(id='2', score=0.85),
        Node(id='3', score=0.75),
    ]
    
    # 创建 Fusion Retriever
    retriever = QueryFusionRetriever(
        retrievers=[mock_retriever1, mock_retriever2],
        mode=FusionMode.SIMPLE,
        top_k=3,
    )
    
    # 执行检索
    query = Query(text="test query")
    nodes = retriever.retrieve(query)
    
    # 验证
    assert len(nodes) == 3
    # Node 2 应该有最高分（0.85）
    assert nodes[0].id == '2'
    assert nodes[0].score == 0.85
```

---

### 8.3 集成测试

```python
def test_integration():
    # 创建真实的存储适配器
    vector_store = ChromaAdapter(...)
    doc_store = SimpleDocStore(...)
    embedder = OpenAIEmbedder(...)
    
    # 创建基础 Retriever
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        doc_store=doc_store,
        embedder=embedder,
    )
    
    keyword_retriever = KeywordRetriever(
        keyword_store=meilisearch_adapter,
        doc_store=doc_store,
    )
    
    # 创建组合 Retriever
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        mode=FusionMode.RECIPROCAL_RANK,
    )
    
    # 执行检索
    query = Query(text="What is machine learning?")
    nodes = fusion_retriever.retrieve(query)
    
    # 验证
    assert len(nodes) > 0
    assert all(node.score is not None for node in nodes)
```

---

## 附录：与 LlamaIndex 架构对比

| 维度 | 本设计 | LlamaIndex |
|------|--------|-----------|
| **基础检索能力** | Storage Layer 提供 | Integrations Retriever 封装 |
| **基础层 Retriever** | `retriever/basic/` | 部分在 `indices/*/retrievers/` |
| **组合层 Retriever** | `retriever/composite/` | `core/retrievers/` |
| **职责分离** | ✅ 清晰（Storage / Basic / Composite） | ⚠️ 混合（Index 包含 Retriever） |
| **目录结构** | ✅ 按功能分层 | ⚠️ 按数据结构分散 |

**本设计的优势**：
- ✅ 职责分离更清晰
- ✅ 目录结构更直观
- ✅ 易于理解和扩展

---

## 总结

本设计文档提供了一个清晰、可扩展的 Retriever 架构：

1. **两层架构**：基础层（封装 Storage 调用）+ 组合层（编排策略）
2. **职责分离**：Storage、Basic Retriever、Composite Retriever 各司其职
3. **统一接口**：所有 Retriever 实现 `BaseRetriever` 接口
4. **灵活组合**：支持任意组合和嵌套
5. **易于扩展**：新增 Retriever 只需遵循命名规范和目录结构

按照本文档实现，你将获得一个结构清晰、易于维护的 Retriever 模块！
