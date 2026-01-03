# ZAG-AI RAG 系统实现计划

## 📋 项目概述

ZAG-AI 是一个模块化的 RAG（Retrieval-Augmented Generation）系统，参考 LlamaIndex 和 RAGFlow 架构设计。

## ✅ 已完成组件

### 1. Readers（数据读取）
- [x] BaseReader 基类
- [x] MarkItDownReader（支持 PDF、Markdown、Word 等多种格式）
- [x] SourceUtils 资源验证工具

### 2. Schemas（数据结构）
- [x] BaseDocument 基类
- [x] DocumentMetadata（结构化文档元数据）
- [x] BaseUnit 基类
- [x] UnitMetadata（通用 Unit 元数据，支持 context_path）
- [x] TextUnit、TableUnit、ImageUnit
- [x] Page 和 PageableDocument
- [x] PDF、Markdown 文档类型
- [x] UnitRegistry（全局 Unit 注册）
- [x] RelationType（关系类型枚举）

### 3. Splitters（文本分割）
- [x] BaseSplitter 基类
- [x] MarkdownHeaderSplitter（按标题分割，inspired by LlamaIndex）
- [x] 模块化组织：`splitters/markdown/header_based.py`

### 4. Extractors（信息提取）
- [x] BaseExtractor 基类
- [x] IdentityExtractor（默认实现）

### 5. Utils（工具模块）
- [x] SourceUtils（资源验证）
- [x] FileType、SourceType 枚举

---

## 🚧 待实现核心组件

### 1. Embedders（嵌入模型）**【优先级：高】**

**状态**：✅ 目录已创建，基类已定义

**目录结构**：
```
zag/embedders/
├── __init__.py           ✅ 已创建
├── base.py              ✅ 已创建（BaseEmbedder）
├── openai.py            ⏸️  待实现
├── huggingface.py       ⏸️  待实现
└── local/               ⏸️  待创建
    ├── __init__.py
    └── bge.py           # BGE 系列模型
```

**核心接口**：
```python
class BaseEmbedder(ABC):
    def embed_text(self, text: str) -> list[float]
    def embed_batch(self, texts: list[str]) -> list[list[float]]
    @property
    def dimension(self) -> int
```

**实现计划**：
- [ ] OpenAIEmbedder（调用 OpenAI API）
- [ ] HuggingFaceEmbedder（本地模型）
- [ ] BGEEmbedder（BGE-M3、BGE-Large 等）

---

### 2. Storages（存储层）**【优先级：高】**

**状态**：✅ 目录已创建，基类已定义

**目录结构**：
```
zag/storages/
├── __init__.py                    ✅ 已创建
├── vector/                        ✅ 已创建
│   ├── __init__.py               ✅ 已创建
│   ├── base.py                   ✅ 已创建（BaseVectorStore, VectorSearchResult）
│   ├── chroma.py                 ⏸️  待实现
│   ├── faiss.py                  ⏸️  待实现
│   ├── milvus.py                 ⏸️  待实现
│   └── qdrant.py                 ⏸️  待实现
└── unit/                          ✅ 已创建
    ├── __init__.py               ✅ 已创建
    ├── base.py                   ✅ 已创建（BaseUnitStore）
    ├── memory.py                 ⏸️  待实现（InMemoryUnitStore）
    └── sqlite.py                 ⏸️  待实现（SQLiteUnitStore）
```

**核心接口**：
```python
# VectorStore: 存储向量
class BaseVectorStore(ABC):
    def add(self, ids: list[str], vectors: list[list[float]], metadatas: Optional[list[dict]] = None)
    def search(self, query_vector: list[float], top_k: int = 10) -> list[VectorSearchResult]
    def delete(self, ids: list[str])
    def clear()
    @property
    def dimension(self) -> int

# UnitStore: 存储完整 Unit 对象（类似 LlamaIndex 的 NodeStore）
class BaseUnitStore(ABC):
    def add(self, units: list[BaseUnit])
    def get(self, unit_id: str) -> Optional[BaseUnit]
    def get_batch(self, unit_ids: list[str]) -> list[BaseUnit]
    def delete(self, unit_ids: list[str])
    def list_all(self) -> list[str]
    def clear()
```

**实现计划**：
- [ ] BaseVectorStore ✅ 基类已定义
- [ ] ChromaVectorStore（推荐优先实现）
- [ ] FAISSVectorStore
- [ ] BaseUnitStore ✅ 基类已定义
- [ ] InMemoryUnitStore（内存存储，用于测试）
- [ ] SQLiteUnitStore（轻量级持久化存储）

---

### 3. Indexers（索引器）**【优先级：高】**

**状态**：✅ 目录已创建，基类已定义

**目录结构**：
```
zag/indexers/
├── __init__.py           ✅ 已创建
├── base.py              ✅ 已创建（BaseIndexer）
├── vector.py            ⏸️  待实现（VectorIndexer）
└── hybrid.py            ⏸️  待实现（HybridIndexer）
```

**核心接口**：
```python
class BaseIndexer(ABC):
    def build(self, units: list[BaseUnit]) -> None
    def add(self, units: list[BaseUnit]) -> None
    def delete(self, unit_ids: list[str]) -> None
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

**实现计划**：
- [ ] VectorIndexer（向量索引）
- [ ] HybridIndexer（向量 + BM25 混合索引）

---

### 4. Retrievers（检索器）**【优先级：高】**

**状态**：✅ 目录已创建，基类已定义

**目录结构**：
```
zag/retrievers/
├── __init__.py           ✅ 已创建
├── base.py              ✅ 已创建（BaseRetriever, RetrievalResult）
├── vector.py            ⏸️  待实现（VectorRetriever）
├── bm25.py              ⏸️  待实现（BM25Retriever）
└── hybrid.py            ⏸️  待实现（HybridRetriever）
```

**核心接口**：
```python
class BaseRetriever(ABC):
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> list[RetrievalResult]

class RetrievalResult:
    unit: BaseUnit
    score: float
    metadata: dict[str, Any]
```

**实现计划**：
- [ ] VectorRetriever（向量检索）
- [ ] BM25Retriever（关键词检索）
- [ ] HybridRetriever（混合检索）

---

### 5. Rerankers（重排序）**【优先级：中】**

**状态**：✅ 目录已创建，基类已定义

**目录结构**：
```
zag/rerankers/
├── __init__.py           ✅ 已创建
├── base.py              ✅ 已创建（BaseReranker）
├── cohere.py            ⏸️  待实现（CohereReranker）
├── bge.py               ⏸️  待实现（BGEReranker）
└── cross_encoder.py     ⏸️  待实现（CrossEncoderReranker）
```

**核心接口**：
```python
class BaseReranker(ABC):
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]
```

**实现计划**：
- [ ] BGEReranker（BGE-Reranker-M3）
- [ ] CohereReranker（Cohere API）
- [ ] CrossEncoderReranker（本地 cross-encoder）

---

## 🔮 未来扩展组件（暂不实现）

### 1. Postprocessors（后处理器）
- 过滤器（Filters）
- 去重器（Deduplicators）
- 增强器（Enhancers）

### 2. Synthesizers（响应合成）
- RefineResponseSynthesizer
- CompactResponseSynthesizer

### 3. Evaluators（评估器）
- 检索质量评估
- 生成质量评估

### 4. Pipelines（Pipeline 编排）
- 端到端 RAG Pipeline
- 多步骤编排

### 5. Agents（Agent 系统）
- 智能问答 Agent
- 工具调用 Agent

### 6. Memory（对话记忆）
- 短期记忆
- 长期记忆

---

## 🎯 实现优先级

### **第一阶段：核心 RAG 流程（必需）**
1. ✅ Readers
2. ✅ Schemas
3. ✅ Splitters
4. ✅ Extractors
5. 🔴 **Embedders**（下一步）
6. 🔴 **Storages**（Vector Store 优先）
7. 🔴 **Indexers**
8. 🔴 **Retrievers**
9. 🟡 Rerankers

### **第二阶段：增强功能**
- Postprocessors
- Synthesizers
- Evaluators

### **第三阶段：高级功能**
- Pipelines
- Agents
- Memory

---

## 📝 技术选型建议

### Embedders
- **推荐**：BGE-M3（中文效果好）
- **备选**：OpenAI text-embedding-3-small

### Vector Store
- **推荐**：Chroma（轻量、易用）
- **备选**：FAISS（高性能）、Milvus（生产级）

### Reranker
- **推荐**：BGE-Reranker-M3
- **备选**：Cohere Rerank API

---

## 🔗 核心调用链

```python
# 完整的 RAG 流程
from zag.readers import MarkItDownReader
from zag.splitters.markdown import MarkdownHeaderSplitter
from zag.embedders import BGEEmbedder
from zag.storages.vector import ChromaVectorStore
from zag.indexers import VectorIndexer
from zag.retrievers import VectorRetriever
from zag.rerankers import BGEReranker

# 1. 读取文档
reader = MarkItDownReader()
doc = reader.read("document.pdf")

# 2. 分割文本
splitter = MarkdownHeaderSplitter()
units = doc.split(splitter)

# 3. 向量化
embedder = BGEEmbedder()
vectors = embedder.embed_batch([u.content for u in units])

# 4. 存储
vector_store = ChromaVectorStore()
vector_store.add(units, vectors)

# 5. 构建索引
indexer = VectorIndexer(embedder, vector_store)
indexer.build(units)

# 6. 检索
retriever = VectorRetriever(indexer)
results = retriever.retrieve("query", top_k=10)

# 7. 重排序
reranker = BGEReranker()
final_results = reranker.rerank("query", results, top_k=5)
```

---

## 📅 更新日志

- **2026-01-03**:
  - ✅ 创建核心组件目录结构
  - ✅ 定义 Embedders、Indexers、Retrievers、Rerankers 基类
  - ✅ 完成 Markdown header-based splitter
  - ✅ 实现 UnitMetadata 通用化设计
  - ✅ 重构 splitters 模块为子模块结构
  - ✅ 重构 storages 模块：删除 document/metadata，添加 unit/
  - ✅ 定义 BaseVectorStore 和 BaseUnitStore 基类

---

## 🎯 下一步行动

### **优先级 1：Embedders**
1. 实现 BGEEmbedder（本地 BGE-M3 模型）
2. 实现 OpenAIEmbedder（API 调用）
3. 编写 embedder 测试

### **优先级 2：Storages**
1. 实现 ChromaVectorStore
2. 实现 SQLiteDocumentStore
3. 编写 storage 测试

### **优先级 3：Indexers & Retrievers**
1. 实现 VectorIndexer
2. 实现 VectorRetriever
3. 编写端到端检索测试

---

## 📚 参考资源

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [RAGFlow GitHub](https://github.com/infiniflow/ragflow)
- [Chroma Documentation](https://docs.trychroma.com/)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-m3)

---

**最后更新**：2026-01-03 23:50
