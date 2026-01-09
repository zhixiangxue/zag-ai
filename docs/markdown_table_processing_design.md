# Markdown 文档切分与表格处理设计

## 文档概述

本文档描述了 Markdown 文档切分、表格处理、上下文扩展的完整设计方案。

---

## 1. Markdown 切分策略

### 1.1 切分行为

**当前实现**：`MarkdownHeaderSplitter` 按**所有级别标题**（H1-H6）切分文档。

**示例**：
```markdown
# Chapter 1
Intro text for chapter 1

## Section 1.1
Content of section 1.1

## Section 1.2
Content of section 1.2

# Chapter 2
Intro text for chapter 2
```

**切分结果**：
- Unit 1: `# Chapter 1\nIntro text...` (context_path: `Chapter 1`)
- Unit 2: `## Section 1.1\n...` (context_path: `Chapter 1/Section 1.1`)
- Unit 3: `## Section 1.2\n...` (context_path: `Chapter 1/Section 1.2`)
- Unit 4: `# Chapter 2\n...` (context_path: `Chapter 2`)

### 1.2 链式结构

切分后的 Units 形成**双向链表**：
```python
Unit1.next_unit_id = "unit_2"
Unit2.prev_unit_id = "unit_1"
Unit2.next_unit_id = "unit_3"
# ...
```

**优点**：
- 保留文档顺序信息
- 支持上下文扩展（见第 4 节）

---

## 2. 表格处理方案

### 2.1 核心问题

**问题 1**：Markdown 表格文本会引入噪音
```markdown
| Product | Price | Rating |
|---------|-------|--------|
| iPhone  | $999  | 4.5/5  |
```
- 管道符 `|`、横线 `---` 对 embedding 无意义
- 表格结构丢失
- 数字和文本混杂

**问题 2**：需要保留原始表格给 LLM
- LLM 对 Markdown 表格理解很好
- Summary 可能丢失精确信息（如具体数值）

### 2.2 最终方案：双字段存储

#### 存储结构
```python
TextUnit(
    unit_id="chunk_1",
    content="""# Product Comparison
We compared two phones.

| Product | Price | Rating |
|---------|-------|--------|
| iPhone  | $999  | 4.5/5  |
| Samsung | $899  | 4.3/5  |

Both phones are great.""",  # ← 原文（给 LLM，含原始 Markdown 表格）
    
    embedding_content="""# Product Comparison
We compared two phones.

Product comparison table with 2 entries: iPhone priced at $999 rating 4.5/5, 
Samsung priced at $899 rating 4.3/5. Key columns: Product name, Price, User rating.

Both phones are great."""  # ← 用于 embedding（表格替换为自然语言描述）
)
```

**字段说明**：
- **`content`**: 原始完整文本，包含 Markdown 表格，供 LLM 使用
- **`embedding_content`**: 处理后的文本，表格替换为自然语言描述，用于 embedding
- 当 `embedding_content` 为 `None` 时，使用 `content` 进行 embedding

#### 工作流程

**1. 切分阶段（Splitter）**
```python
# 检测表格
if has_table(section):
    table_md = extract_table(section)
    
    # 生成 summary（用于 embedding）
    table_summary = summarize_table(table_md)
    embedding_content = section.replace(table_md, table_summary)
    
    # 保留原文（给 LLM）
    content = section
    
    return TextUnit(
        content=content,  # 原文
        embedding_content=embedding_content  # 处理后版本
    )
```

**2. Embedding 阶段（Embedder）**
```python
# 优先使用 embedding_content，如果没有则使用 content
text_to_embed = unit.embedding_content if unit.embedding_content else unit.content
embedding = embedder.embed(text_to_embed)
```

**3. 存储阶段（VectorIndexer）**
```python
vector_store.add({
    "unit_id": "chunk_1",
    "content": "...原文...",  # 原始文本（含 Markdown 表格）
    "embedding_content": "...table summary...",  # 用于 embedding
    "embedding": [...]
})
```

**4. 检索阶段（Retriever）**
```python
# 检索时使用 embedding（基于 embedding_content）
results = retriever.retrieve("query")

# 返回的 units 直接使用 content（原文）给 LLM
for unit in results:
    # unit.content 已经是原文，包含完整的 Markdown 表格
    llm_context = unit.content
```

#### 方案优势

| 维度 | 优势 |
|------|------|
| **实现复杂度** | 低，直接存储两个字段 |
| **检索性能** | 高，单次查询即可 |
| **信号完整性** | 好，table_summary 提供完整语义 |
| **LLM 理解准确度** | 高，使用原始 Markdown 表格 |
| **存储开销** | 可接受（只是多存一份文本） |

### 2.3 表格是否需要单独存储？

**结论**：**大部分场景不需要**

#### 不需要单独存储的情况（推荐）
- 表格较小（< 10 行）
- 表格和文本强关联
- 每个表格只出现一次
- table_summary 足够表达表格内容

**方案**：
```python
TextUnit(
    content="...原文（含 Markdown 表格）...",  # 给 LLM
    embedding_content="...table_summary..."  # 用于 embedding
)
# ✅ 简单！不需要 TableUnit，不需要 relation
```

#### 需要单独存储的情况（少数）
- 表格很大（> 20 行）
- 需要单独检索表格
- 表格被多处引用
- 需要特殊的表格处理（如结构化查询）

**方案**：
```python
TextUnit(
    content="...原文（含表格占位符）...",
    embedding_content="...brief summary...",
    related_unit_ids=["table_1"]
)

TableUnit(
    unit_id="table_1",
    content="...markdown table...",  # 原始表格
    embedding_content="...detailed summary..."  # 表格描述
)
```

---

## 3. Unit ID 与向量数据库

### 3.1 ID 映射关系

**关键点**：向量数据库的 ID = 用户指定的 `unit_id`（不是自动生成）

#### Chroma 示例
```python
# 存储时
chroma_collection.add(
    ids=["chunk_1", "chunk_2", "chunk_3"],  # ← 您的 unit_id
    documents=[...],
    embeddings=[...],
    metadatas=[
        {"prev_unit_id": None, "next_unit_id": "chunk_2"},
        {"prev_unit_id": "chunk_1", "next_unit_id": "chunk_3"},
        ...
    ]
)

# 查询时
results = chroma_collection.get(ids=["chunk_1", "chunk_3"])  # ← 用 unit_id 查询
```

#### Qdrant 示例
```python
# 存储时
qdrant.upsert(
    collection_name="chunks",
    points=[
        PointStruct(
            id="chunk_1",  # ← 您的 unit_id
            vector=[...],
            payload={"prev_unit_id": None, "next_unit_id": "chunk_2"}
        ),
        ...
    ]
)

# 查询时
results = qdrant.retrieve(collection_name="chunks", ids=["chunk_1", "chunk_3"])
```

### 3.2 完整工作流程

```python
# 1. Splitter 生成 Unit（带 unit_id）
units = splitter.split(document)
# → [
#     TextUnit(unit_id="doc1_chunk_1", prev_unit_id=None, next_unit_id="doc1_chunk_2"),
#     TextUnit(unit_id="doc1_chunk_2", prev_unit_id="doc1_chunk_1", next_unit_id="doc1_chunk_3"),
#     ...
# ]

# 2. Indexer 存储到向量数据库（使用 unit_id 作为向量数据库 ID）
vector_indexer.add(units)

# 3. Retriever 检索
results = retriever.retrieve("query")
# → [TextUnit(unit_id="doc1_chunk_2", prev_unit_id="doc1_chunk_1", next_unit_id="doc1_chunk_3")]

# 4. 可以通过 prev_unit_id / next_unit_id 获取邻居
neighbor_ids = [results[0].prev_unit_id, results[0].next_unit_id]
neighbors = vector_store.get(ids=neighbor_ids)  # ← 批量查询
```

---

## 4. 上下文扩展（Context Expansion）

### 4.1 需求场景

**问题**：单个 chunk 语义不够完整，需要获取前后 chunks 作为上下文。

**例如**：
- 检索到 `chunk_2`
- 需要获取 `chunk_1` 和 `chunk_3` 来提供完整上下文

### 4.2 解决方案：Postprocessor

#### 实现方式
```python
class ContextExpansionPostprocessor(BasePostprocessor):
    """
    上下文扩展 Postprocessor
    自动获取检索到的 chunk 的前后邻居
    """
    
    def __init__(
        self, 
        vector_store,
        expand_previous: int = 1,  # 向前扩展几个 chunk
        expand_next: int = 1,      # 向后扩展几个 chunk
    ):
        self.vector_store = vector_store
        self.expand_previous = expand_previous
        self.expand_next = expand_next
    
    def process(self, units: list[BaseUnit]) -> list[BaseUnit]:
        """为每个检索到的 unit 扩展上下文"""
        
        # 1. 收集需要获取的 neighbor IDs
        neighbor_ids = []
        for unit in units:
            if unit.prev_unit_id:
                neighbor_ids.append(unit.prev_unit_id)
            if unit.next_unit_id:
                neighbor_ids.append(unit.next_unit_id)
        
        # 2. 批量获取（高效！只需 1 次查询）
        if neighbor_ids:
            neighbors = self.vector_store.get(ids=neighbor_ids)
            
            # 3. 合并并排序
            all_units = units + neighbors
            return self._reorder_by_chain(all_units)
        
        return units
    
    def _reorder_by_chain(self, units: list[BaseUnit]) -> list[BaseUnit]:
        """根据 prev_unit_id / next_unit_id 链式关系排序"""
        # 构建映射
        unit_map = {u.unit_id: u for u in units}
        
        # 找起点（没有 prev 的）
        start_units = [u for u in units if not u.prev_unit_id or u.prev_unit_id not in unit_map]
        
        # 按链排序
        ordered = []
        for start in start_units:
            current = start
            while current:
                if current.unit_id not in [u.unit_id for u in ordered]:
                    ordered.append(current)
                next_id = current.next_unit_id
                current = unit_map.get(next_id) if next_id else None
        
        return ordered
```

#### 使用方式
```python
# 1. 检索
results = retriever.retrieve("query", top_k=5)
# → [chunk_2, chunk_7, chunk_15, ...]

# 2. 扩展上下文
postprocessor = ContextExpansionPostprocessor(
    vector_store=vector_store,
    expand_previous=1,  # 向前 1 个
    expand_next=1       # 向后 1 个
)
expanded_results = postprocessor.process(results)
# → [chunk_1, chunk_2, chunk_3, chunk_6, chunk_7, chunk_8, ...]

# 3. 给 LLM
context = "\n\n".join([unit.content for unit in expanded_results])
```

### 4.3 性能分析

**查询次数**：
1. 主查询：1 次（vector search）
2. 扩展查询：1 次（批量 get by IDs）
3. **总计**：2 次查询 ✅

**vs. 其他方案**：
- ❌ 方案 A（逐个查询）：N 次查询
- ✅ 方案 B（批量查询）：2 次查询（推荐）
- ❌ 方案 C（冗余存储）：存储开销大

---

## 5. 向量数据库接口要求

### 5.1 BaseVectorStore 接口

```python
class BaseVectorStore:
    @abstractmethod
    def add(self, units: list[BaseUnit]) -> None:
        """添加 units（使用 unit.unit_id 作为向量数据库 ID）"""
        pass
    
    @abstractmethod
    def get(self, ids: list[str]) -> list[BaseUnit]:
        """
        通过 unit_id 列表批量获取 units
        
        Args:
            ids: unit_id 列表
        
        Returns:
            对应的 BaseUnit 列表
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[BaseUnit]:
        """向量检索"""
        pass
```

### 5.2 实现示例（Chroma）

```python
class ChromaVectorStore(BaseVectorStore):
    def add(self, units: list[BaseUnit]) -> None:
        self.collection.add(
            ids=[u.unit_id for u in units],  # ← 使用 unit_id
            documents=[u.content for u in units],
            embeddings=[u.embedding for u in units],
            metadatas=[{
                "prev_unit_id": u.prev_unit_id,
                "next_unit_id": u.next_unit_id,
                "content": u.content,  # ← 保存原文（给 LLM）
                **u.metadata.custom
            } for u in units]
        )
    
    def get(self, ids: list[str]) -> list[BaseUnit]:
        """批量获取（高效）"""
        results = self.collection.get(
            ids=ids,
            include=["metadatas", "embeddings"]
        )
        
        units = []
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            unit = self._to_unit(
                unit_id=doc_id,
                content=metadata.get('content'),  # 从 metadata 恢复原文
                embedding_content=None,  # embedding_content 不需要恢复
                metadata=metadata
            )
            units.append(unit)
        
        return units
```

---

## 6. 设计总结

### 6.1 关键决策

| 设计点 | 决策 | 理由 |
|--------|------|------|
| **表格处理** | 双字段存储（content + embedding_content） | 平衡 embedding 效果和 LLM 理解 |
| **字段用途** | content=原文（LLM用）, embedding_content=处理后（embedding用） | 保持向后兼容，不影响现有代码 |
| **表格独立存储** | 大部分场景不需要 | 避免过度设计，简化实现 |
| **上下文扩展** | Postprocessor + 批量查询 | 高效（2 次查询），灵活可控 |
| **向量数据库 ID** | 使用用户指定的 unit_id | 保持系统一致性，支持邻居查询 |

### 6.2 完整数据流

```
文档
  ↓
Splitter（切分 + 表格处理）
  ↓
[TextUnit(content=原文, embedding_content=summary), ...]
  ↓
Embedder（embed embedding_content）
  ↓
VectorIndexer（存储 content + embedding + metadata）
  ↓
VectorStore（unit_id 作为 ID）
  ↓
Retriever（向量检索）
  ↓
Postprocessor（上下文扩展：批量获取 prev/next）
  ↓
[扩展后的 Units（使用 content，包含原始表格）]
  ↓
LLM（获得完整上下文 + 原始表格）
```

### 6.3 核心优势

✅ **简单高效**：不需要复杂的关系管理  
✅ **性能优秀**：最多 2 次查询完成上下文扩展  
✅ **语义完整**：embedding_content 保证 embedding 效果  
✅ **信息保真**：content 保证 LLM 推理准确  
✅ **灵活可扩展**：通过 Postprocessor 控制扩展策略  
✅ **向后兼容**：content 保持原有语义，不影响现有代码

---

## 7. 未来优化方向

### 7.1 可选优化
- 支持可配置的切分级别（只切 H1，或 H1+H2）
- 智能判断表格大小，自动选择是否单独存储
- 支持更复杂的上下文扩展策略（如多跳扩展）

### 7.2 性能优化
- 缓存邻居查询结果
- 并发查询多个向量数据库
- 预加载常用的上下文

---

**文档版本**：v2.0  
**最后更新**：2026-01-09  
**作者**：Qoder & User  
**变更说明**：v2.0 - 将 `original_content` 改为 `embedding_content`，`content` 保持原文语义
