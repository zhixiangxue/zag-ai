# Indexer 设计文档

## 1. 设计原则

### 1.1 核心理念

**单一职责原则（Single Responsibility Principle）**

- **Indexer 的唯一职责**：将已处理好的节点（Nodes）索引到存储系统中
- **不负责**：文档分块（Chunking）、向量嵌入（Embedding）等预处理操作
- **语义**：Indexer = "索引器"，专注于"索引"操作本身

### 1.2 职责边界

```
预处理层（独立）              索引层（Indexer）              存储层
──────────────────           ─────────────────           ──────────
Chunker  → 分块              Indexer → 索引              Storage
Embedder → 嵌入              （写入操作）                （数据库）
Filter   → 过滤
Transform → 转换
```

---

## 2. 架构设计

### 2.1 完整数据流

```
Documents (原始文档)
    ↓
[预处理层 - 独立工具]
    ↓
Chunker (分块器)
    ↓
Nodes (文本块)
    ↓
Embedder (嵌入器)
    ↓
Nodes with Embeddings (带向量的节点)
    ↓
[索引层 - Indexer]
    ↓
Indexer (索引器) - 只负责存储
    ↓
[存储层 - Storage]
    ↓
Storage (向量数据库 + 文档数据库)
```

### 2.2 层级关系

```python
┌─────────────────────────────────────────────┐
│         Application Layer (应用层)           │
│  - Pipeline (便捷封装，可选)                 │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      Preprocessing Layer (预处理层)          │
│  - Chunker (分块器)                          │
│  - Embedder (嵌入器)                         │
│  - Filter (过滤器)                           │
│  - Transform (转换器)                        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         Indexing Layer (索引层)              │
│  - VectorIndexer (向量索引器)               │
│  - KeywordIndexer (关键词索引器)            │
│  - HybridIndexer (混合索引器)               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│        Storage Layer (存储层)                │
│  - VectorStoreAdapter (向量存储适配器)      │
│  - DocStoreAdapter (文档存储适配器)         │
│  - KeywordStoreAdapter (关键词存储适配器)   │
└─────────────────────────────────────────────┘
```

---

## 3. 接口设计

### 3.1 BaseIndexer（抽象基类）

```python
from abc import ABC, abstractmethod
from typing import List
from core.schema import Node

class BaseIndexer(ABC):
    """
    索引器基类
    
    职责：
    - 将已处理的节点索引到存储系统
    - 提供统一的索引接口（CRUD）
    - 不包含预处理逻辑
    """
    
    @abstractmethod
    def index_nodes(self, nodes: List[Node]) -> None:
        """
        索引节点到存储系统
        
        Args:
            nodes: 已处理完成的节点列表（包含向量、元数据等）
            
        前置条件：
            - nodes 必须包含所有必需字段（id, text, embedding, metadata）
            - embedding 必须已经计算完成
        """
        pass
    
    @abstractmethod
    def delete_nodes(self, node_ids: List[str]) -> None:
        """
        从存储系统中删除节点
        
        Args:
            node_ids: 要删除的节点 ID 列表
        """
        pass
    
    @abstractmethod
    def update_nodes(self, nodes: List[Node]) -> None:
        """
        更新存储系统中的节点
        
        Args:
            nodes: 要更新的节点列表
        """
        pass
    
    @abstractmethod
    def node_exists(self, node_id: str) -> bool:
        """
        检查节点是否存在
        
        Args:
            node_id: 节点 ID
            
        Returns:
            bool: 节点是否存在
        """
        pass
```

### 3.2 VectorIndexer（向量索引器）

```python
from storage.vector_store.base import VectorStoreAdapter
from storage.doc_store.base import DocStoreAdapter
from indexer.base import BaseIndexer

class VectorIndexer(BaseIndexer):
    """
    向量索引器
    
    职责：
    - 将带向量的节点索引到向量数据库和文档数据库
    - 支持所有实现了 VectorStoreAdapter 的向量数据库
    
    设计特点：
    - 只依赖 Storage Adapter，不依赖预处理工具
    - 通用实现，不绑定特定数据库
    """
    
    def __init__(self, 
                 vector_store: VectorStoreAdapter,
                 doc_store: DocStoreAdapter):
        """
        初始化向量索引器
        
        Args:
            vector_store: 向量存储适配器（Pinecone, Chroma, Weaviate, etc.）
            doc_store: 文档存储适配器（MongoDB, Postgres, etc.）
        """
        self.vector_store = vector_store
        self.doc_store = doc_store
    
    def index_nodes(self, nodes: List[Node]) -> None:
        """
        索引节点
        
        Args:
            nodes: 已包含 embedding 的节点列表
            
        Raises:
            ValueError: 如果节点缺少 embedding
        """
        # 验证节点
        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"Node {node.id} missing embedding")
        
        # 提取数据
        vectors = [node.embedding for node in nodes]
        metadata = [node.metadata for node in nodes]
        ids = [node.id for node in nodes]
        
        # 索引到向量存储
        self.vector_store.insert(
            vectors=vectors,
            metadata=metadata,
            ids=ids,
        )
        
        # 索引到文档存储（保存原文）
        self.doc_store.insert(nodes)
    
    def delete_nodes(self, node_ids: List[str]) -> None:
        """删除节点"""
        self.vector_store.delete(node_ids)
        self.doc_store.delete(node_ids)
    
    def update_nodes(self, nodes: List[Node]) -> None:
        """更新节点（先删除再插入）"""
        node_ids = [node.id for node in nodes]
        self.delete_nodes(node_ids)
        self.index_nodes(nodes)
    
    def node_exists(self, node_id: str) -> bool:
        """检查节点是否存在"""
        return self.doc_store.exists(node_id)
```

### 3.3 KeywordIndexer（关键词索引器）

```python
from storage.keyword_store.base import KeywordStoreAdapter
from storage.doc_store.base import DocStoreAdapter

class KeywordIndexer(BaseIndexer):
    """
    关键词索引器
    
    职责：
    - 将节点索引到关键词搜索引擎（Elasticsearch, Meilisearch, etc.）
    """
    
    def __init__(self, 
                 keyword_store: KeywordStoreAdapter,
                 doc_store: DocStoreAdapter):
        self.keyword_store = keyword_store
        self.doc_store = doc_store
    
    def index_nodes(self, nodes: List[Node]) -> None:
        """索引节点到关键词存储"""
        # 提取关键词索引数据
        documents = []
        for node in nodes:
            doc = {
                'id': node.id,
                'text': node.text,
                'metadata': node.metadata,
            }
            documents.append(doc)
        
        # 索引到关键词存储
        self.keyword_store.index_documents(documents)
        
        # 索引到文档存储
        self.doc_store.insert(nodes)
    
    # ... 其他方法实现
```

### 3.4 HybridIndexer（混合索引器）

```python
class HybridIndexer(BaseIndexer):
    """
    混合索引器
    
    职责：
    - 同时索引到多个存储系统（向量 + 关键词）
    """
    
    def __init__(self, 
                 vector_store: VectorStoreAdapter,
                 keyword_store: KeywordStoreAdapter,
                 doc_store: DocStoreAdapter):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.doc_store = doc_store
    
    def index_nodes(self, nodes: List[Node]) -> None:
        """索引节点到多个存储"""
        # 验证
        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"Node {node.id} missing embedding")
        
        # 索引到向量存储
        vectors = [node.embedding for node in nodes]
        metadata = [node.metadata for node in nodes]
        ids = [node.id for node in nodes]
        self.vector_store.insert(vectors, metadata, ids)
        
        # 索引到关键词存储
        documents = [
            {'id': n.id, 'text': n.text, 'metadata': n.metadata}
            for n in nodes
        ]
        self.keyword_store.index_documents(documents)
        
        # 索引到文档存储
        self.doc_store.insert(nodes)
    
    # ... 其他方法实现
```

---

## 4. 使用示例

### 4.1 基础用法

```python
from storage.vector_store import PineconeAdapter
from storage.doc_store import MongoDBAdapter
from chunker import SentenceChunker
from embedder import OpenAIEmbedder
from indexer import VectorIndexer

# 1. 准备存储
vector_store = PineconeAdapter()
vector_store.connect({
    'api_key': 'your-api-key',
    'index_name': 'my-index'
})

doc_store = MongoDBAdapter()
doc_store.connect({'uri': 'mongodb://localhost'})

# 2. 准备预处理工具（独立于 Indexer）
chunker = SentenceChunker(chunk_size=512, overlap=50)
embedder = OpenAIEmbedder(api_key='your-api-key')

# 3. 创建索引器（只依赖 Storage）
indexer = VectorIndexer(
    vector_store=vector_store,
    doc_store=doc_store,
)

# 4. 执行索引流程（用户控制每一步）
# 步骤 1：分块
nodes = chunker.chunk_documents(documents)

# 步骤 2：嵌入
nodes_with_embeddings = embedder.embed_nodes(nodes)

# 步骤 3：索引
indexer.index_nodes(nodes_with_embeddings)
```

### 4.2 自定义预处理流程

```python
# 用户可以完全控制预处理流程

# 1. 使用不同的分块策略
from chunker import SemanticChunker

semantic_chunker = SemanticChunker(
    similarity_threshold=0.7,
    min_chunk_size=100,
)
nodes = semantic_chunker.chunk_documents(documents)

# 2. 添加自定义过滤
nodes = [n for n in nodes if len(n.text) > 50]

# 3. 添加自定义元数据
from datetime import datetime

for node in nodes:
    node.metadata['indexed_at'] = datetime.now().isoformat()
    node.metadata['source'] = 'custom_pipeline'

# 4. 使用不同的嵌入模型
from embedder import LocalEmbedder

local_embedder = LocalEmbedder(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
nodes_with_embeddings = local_embedder.embed_nodes(nodes)

# 5. 索引（Indexer 不关心前面的处理过程）
indexer.index_nodes(nodes_with_embeddings)
```

### 4.3 混合索引

```python
from storage.keyword_store import ElasticsearchAdapter
from indexer import HybridIndexer

# 1. 准备多个存储
vector_store = PineconeAdapter()
keyword_store = ElasticsearchAdapter()
doc_store = MongoDBAdapter()

# 配置连接...

# 2. 创建混合索引器
hybrid_indexer = HybridIndexer(
    vector_store=vector_store,
    keyword_store=keyword_store,
    doc_store=doc_store,
)

# 3. 预处理
nodes = chunker.chunk_documents(documents)
nodes_with_embeddings = embedder.embed_nodes(nodes)

# 4. 索引到多个存储
hybrid_indexer.index_nodes(nodes_with_embeddings)
```

### 4.4 切换数据库

```python
# 开发环境：使用 Chroma（本地）
from storage.vector_store import ChromaAdapter

dev_vector_store = ChromaAdapter()
dev_vector_store.connect({'persist_directory': './chroma_db'})

dev_indexer = VectorIndexer(
    vector_store=dev_vector_store,
    doc_store=doc_store,
)

# 生产环境：切换到 Pinecone（云端）
prod_vector_store = PineconeAdapter()
prod_vector_store.connect({
    'api_key': 'prod-api-key',
    'index_name': 'prod-index'
})

prod_indexer = VectorIndexer(
    vector_store=prod_vector_store,
    doc_store=doc_store,
)

# ✅ 同样的预处理流程，不同的存储
nodes = chunker.chunk_documents(documents)
nodes_with_embeddings = embedder.embed_nodes(nodes)

# 索引到不同的数据库
dev_indexer.index_nodes(nodes_with_embeddings)   # 开发环境
# prod_indexer.index_nodes(nodes_with_embeddings)  # 生产环境
```

---

## 5. 可选：Pipeline 封装

### 5.1 IndexingPipeline（便捷封装）

对于不需要自定义预处理流程的标准场景，可以提供 Pipeline 封装：

```python
from pipeline import IndexingPipeline

class IndexingPipeline:
    """
    索引流水线（便捷封装）
    
    职责：
    - 组合预处理工具和索引器
    - 提供一站式索引接口
    
    适用场景：
    - 标准流程，不需要自定义预处理
    - 快速原型开发
    """
    
    def __init__(self, 
                 chunker: BaseChunker,
                 embedder: BaseEmbedder,
                 indexer: BaseIndexer):
        self.chunker = chunker
        self.embedder = embedder
        self.indexer = indexer
    
    def run(self, documents: List[Document]) -> None:
        """执行完整的索引流程"""
        # 1. 分块
        nodes = self.chunker.chunk_documents(documents)
        
        # 2. 嵌入
        nodes = self.embedder.embed_nodes(nodes)
        
        # 3. 索引
        self.indexer.index_nodes(nodes)
    
    def run_with_progress(self, documents: List[Document]) -> None:
        """带进度显示的索引流程"""
        from tqdm import tqdm
        
        # 1. 分块
        print("Chunking documents...")
        nodes = self.chunker.chunk_documents(documents)
        
        # 2. 嵌入（带进度条）
        print(f"Embedding {len(nodes)} nodes...")
        nodes = self.embedder.embed_nodes(nodes)
        
        # 3. 索引
        print("Indexing nodes...")
        self.indexer.index_nodes(nodes)
        
        print(f"✓ Indexed {len(nodes)} nodes")
```

### 5.2 Pipeline 使用示例

```python
# 方式 1：手动控制每一步（推荐，灵活）
nodes = chunker.chunk_documents(documents)
nodes = custom_filter(nodes)  # 自定义步骤
nodes = embedder.embed_nodes(nodes)
indexer.index_nodes(nodes)

# 方式 2：使用 Pipeline（便捷）
pipeline = IndexingPipeline(
    chunker=chunker,
    embedder=embedder,
    indexer=indexer,
)
pipeline.run(documents)
```

---

## 6. 目录结构

```
your_rag_framework/
├── chunker/              # 分块工具（预处理）
│   ├── base.py          # BaseChunker
│   ├── sentence_chunker.py
│   └── semantic_chunker.py
│
├── embedder/             # 嵌入工具（预处理）
│   ├── base.py          # BaseEmbedder
│   ├── openai_embedder.py
│   └── local_embedder.py
│
├── indexer/              # 索引器（只负责存储）
│   ├── __init__.py
│   ├── base.py          # BaseIndexer
│   ├── vector_indexer.py
│   ├── keyword_indexer.py
│   └── hybrid_indexer.py
│
├── storage/              # 存储适配器（底层数据库）
│   ├── vector_store/
│   │   ├── base.py
│   │   ├── pinecone.py
│   │   ├── chroma.py
│   │   └── weaviate.py
│   ├── doc_store/
│   │   ├── base.py
│   │   ├── mongodb.py
│   │   └── postgres.py
│   └── keyword_store/
│       ├── base.py
│       └── elasticsearch.py
│
├── pipeline/             # 可选：便捷封装
│   └── indexing_pipeline.py
│
└── core/                 # 核心数据结构
    ├── schema.py        # Node, Document, Query
    └── types.py
```

---

## 7. 核心设计要点

### 7.1 职责清晰

```
Chunker   → 职责：分块文档        → 输入：Document    → 输出：Node
Embedder  → 职责：计算向量        → 输入：Node        → 输出：Node (with embedding)
Indexer   → 职责：索引节点        → 输入：Node        → 输出：存储操作
Retriever → 职责：检索节点        → 输入：Query       → 输出：Node
```

### 7.2 语义明确

```python
# ✅ 语义清晰
chunker.chunk_documents(documents)       # 分块器分块文档
embedder.embed_nodes(nodes)              # 嵌入器嵌入节点
indexer.index_nodes(nodes)               # 索引器索引节点
retriever.retrieve(query)                # 检索器检索数据

# ❌ 语义混淆（避免）
indexer.build_from_documents(documents)  # 索引器"构建"？包含了太多职责
```

### 7.3 依赖注入

```python
# ✅ 依赖抽象接口
indexer = VectorIndexer(
    vector_store=vector_store,  # VectorStoreAdapter 接口
    doc_store=doc_store,        # DocStoreAdapter 接口
)

# ✅ 切换数据库只需改配置
pinecone_indexer = VectorIndexer(
    vector_store=PineconeAdapter(),
    doc_store=doc_store,
)

chroma_indexer = VectorIndexer(
    vector_store=ChromaAdapter(),  # 只改这一行
    doc_store=doc_store,
)
```

### 7.4 可组合性

```python
# 用户可以自由组合工具
# 场景 1：标准流程
nodes = chunker.chunk(docs)
nodes = embedder.embed(nodes)
indexer.index(nodes)

# 场景 2：跳过分块
nodes = [Node(text=doc.text) for doc in docs]
nodes = embedder.embed(nodes)
indexer.index(nodes)

# 场景 3：使用缓存的向量
nodes = load_cached_nodes()
indexer.index(nodes)  # 直接索引

# 场景 4：自定义流程
nodes = chunker.chunk(docs)
nodes = custom_filter(nodes)
nodes = custom_transform(nodes)
nodes = embedder.embed(nodes)
indexer.index(nodes)
```

---

## 8. 扩展性

### 8.1 新增数据库

```python
# 只需实现 VectorStoreAdapter 接口
from storage.vector_store.base import VectorStoreAdapter

class QdrantAdapter(VectorStoreAdapter):
    def insert(self, vectors, metadata, ids):
        # Qdrant 特定代码
        pass
    
    def query(self, query_vector, top_k, filters):
        # Qdrant 特定代码
        pass
    
    def delete(self, ids):
        # Qdrant 特定代码
        pass

# ✅ 自动支持所有上层功能
qdrant_indexer = VectorIndexer(
    vector_store=QdrantAdapter(),
    doc_store=doc_store,
)
```

### 8.2 新增索引类型

```python
# 实现新的 Indexer
class GraphIndexer(BaseIndexer):
    """图索引器"""
    
    def __init__(self, graph_store: GraphStoreAdapter, doc_store: DocStoreAdapter):
        self.graph_store = graph_store
        self.doc_store = doc_store
    
    def index_nodes(self, nodes: List[Node]) -> None:
        # 提取图结构
        entities, relations = self._extract_graph(nodes)
        
        # 索引到图数据库
        self.graph_store.add_entities(entities)
        self.graph_store.add_relations(relations)
        
        # 索引到文档存储
        self.doc_store.insert(nodes)
```

---

## 9. 测试建议

### 9.1 单元测试

```python
import pytest
from indexer import VectorIndexer
from storage.vector_store import MockVectorStoreAdapter
from storage.doc_store import MockDocStoreAdapter

def test_vector_indexer_index_nodes():
    # 1. 准备 Mock Storage
    vector_store = MockVectorStoreAdapter()
    doc_store = MockDocStoreAdapter()
    
    # 2. 创建 Indexer
    indexer = VectorIndexer(vector_store, doc_store)
    
    # 3. 准备测试数据
    nodes = [
        Node(id='1', text='test', embedding=[0.1, 0.2, 0.3]),
        Node(id='2', text='test2', embedding=[0.4, 0.5, 0.6]),
    ]
    
    # 4. 执行索引
    indexer.index_nodes(nodes)
    
    # 5. 验证
    assert vector_store.insert_called
    assert doc_store.insert_called
    assert len(vector_store.stored_vectors) == 2

def test_vector_indexer_missing_embedding():
    indexer = VectorIndexer(MockVectorStoreAdapter(), MockDocStoreAdapter())
    
    nodes = [Node(id='1', text='test', embedding=None)]
    
    with pytest.raises(ValueError, match="missing embedding"):
        indexer.index_nodes(nodes)
```

### 9.2 集成测试

```python
def test_full_indexing_pipeline():
    # 1. 准备真实存储（测试环境）
    vector_store = ChromaAdapter()
    vector_store.connect({'persist_directory': './test_chroma'})
    
    doc_store = MongoDBAdapter()
    doc_store.connect({'uri': 'mongodb://localhost/test'})
    
    # 2. 准备预处理工具
    chunker = SentenceChunker(chunk_size=256)
    embedder = OpenAIEmbedder(api_key='test-key')
    
    # 3. 创建 Indexer
    indexer = VectorIndexer(vector_store, doc_store)
    
    # 4. 执行完整流程
    documents = load_test_documents()
    nodes = chunker.chunk_documents(documents)
    nodes = embedder.embed_nodes(nodes)
    indexer.index_nodes(nodes)
    
    # 5. 验证
    assert doc_store.count() == len(nodes)
    assert vector_store.count() == len(nodes)
    
    # 清理
    vector_store.clear()
    doc_store.clear()
```

---

## 10. 总结

### 10.1 核心原则

1. **单一职责**：Indexer 只负责索引（写入 Storage）
2. **职责分离**：预处理（Chunker, Embedder）与索引（Indexer）分离
3. **依赖抽象**：依赖 Storage Adapter 接口，不依赖具体实现
4. **可组合性**：用户可以自由组合预处理工具和索引器

### 10.2 设计优势

- ✅ 职责清晰：每个组件职责明确
- ✅ 易于测试：组件独立，便于单元测试
- ✅ 易于扩展：新增数据库只需实现 Adapter
- ✅ 灵活性高：用户可以完全控制预处理流程
- ✅ 语义明确：API 命名清晰，符合直觉

### 10.3 使用建议

- **标准场景**：使用 Pipeline 快速上手
- **自定义需求**：手动组合工具，完全控制流程
- **生产环境**：建议手动控制，便于监控和调试
- **测试环境**：使用 Mock Storage，快速验证逻辑

---

## 附录：完整代码示例

参见项目中的以下文件：

- `indexer/base.py` - BaseIndexer 抽象基类
- `indexer/vector_indexer.py` - VectorIndexer 实现
- `indexer/keyword_indexer.py` - KeywordIndexer 实现
- `indexer/hybrid_indexer.py` - HybridIndexer 实现
- `storage/vector_store/base.py` - VectorStoreAdapter 接口
- `storage/doc_store/base.py` - DocStoreAdapter 接口
- `pipeline/indexing_pipeline.py` - IndexingPipeline 便捷封装
