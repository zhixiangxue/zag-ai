# Retriever Fusion 策略详解

本文档详细介绍 `QueryFusionRetriever` 支持的三种融合策略：SIMPLE、RECIPROCAL_RANK 和 RELATIVE_SCORE。

---

## 概述

当使用多个 Retriever 进行检索时，需要将它们的结果融合（merge/fusion）成一个统一的结果列表。不同的融合策略有不同的特点和适用场景。

---

## 1. SIMPLE - 简单融合

### 原理

- 对来自多个 retriever 的结果进行去重
- 对于重复的 unit，保留最高的分数
- 直接按分数排序返回

### 算法流程

```python
for each retriever:
    results = retriever.retrieve(query)
    for each unit in results:
        if unit_id in merged_results:
            # 保留最高分
            merged_results[unit_id].score = max(existing_score, new_score)
        else:
            merged_results[unit_id] = unit

# 按分数降序排序
return sorted(merged_results, key=lambda x: x.score, reverse=True)
```

### 优点

- ✅ **实现简单**：逻辑最直观，易于理解和调试
- ✅ **计算开销小**：只需要简单的去重和排序操作
- ✅ **保留原始分数**：不改变分数的原始语义，便于追溯

### 缺点

- ❌ **分数不可比**：不同 retriever 的分数可能在不同的尺度上
  - 示例：Retriever A 的分数范围是 0.8-0.95，Retriever B 的分数范围是 0.3-0.6
  - 结果：Retriever A 的结果会完全压倒 Retriever B
- ❌ **排名信息丢失**：只看分数，忽略了排名位置的重要性
- ❌ **容易被主导**：分数普遍偏高的 retriever 会压倒其他的

### 适用场景

- 🎯 **相同类型的 retriever**：多个使用相同 embedder 的 VectorRetriever
- 🎯 **快速原型验证**：需要快速组合结果进行测试
- 🎯 **分数已归一化**：确信所有 retriever 的分数在同一尺度上

### 使用示例

```python
from zag.retrievers import QueryFusionRetriever, FusionMode

# 多个相同类型的 retriever
fusion = QueryFusionRetriever(
    retrievers=[retriever1, retriever2],
    mode=FusionMode.SIMPLE,
    top_k=10
)

results = fusion.retrieve("What is machine learning?")
```

---

## 2. RECIPROCAL_RANK - 倒数排名融合 (RRF)

### 原理

使用 RRF (Reciprocal Rank Fusion) 算法，基于排名而非分数进行融合。

**公式**：
```
score(unit) = Σ [1 / (k + rank_i)]
```

其中：
- `k` 是常数（默认 60，来自信息检索领域的经验值）
- `rank_i` 是该 unit 在第 i 个 retriever 中的排名（从 0 开始）
- 对所有 retriever 的倒数排名求和

### 算法示例

假设一个 unit 在三个 retriever 中的排名分别是：第 1、第 3、未出现

```python
# Retriever 1: rank = 0 (第1名)
contribution_1 = 1 / (60 + 0) = 1/60 = 0.0167

# Retriever 2: rank = 2 (第3名，索引从0开始)
contribution_2 = 1 / (60 + 2) = 1/62 = 0.0161

# Retriever 3: 未出现，不贡献分数

# 最终分数
final_score = 0.0167 + 0.0161 = 0.0328
```

### 特性分析

**排名越靠前，贡献越大**：
- 第 1 名：1/60 ≈ 0.0167
- 第 2 名：1/61 ≈ 0.0164
- 第 3 名：1/62 ≈ 0.0161
- 第 10 名：1/69 ≈ 0.0145

**多个 retriever 都返回的 unit 会得到更高分数**（民主投票机制）

### 优点

- ✅ **不依赖原始分数**：只使用排名，完全避免了不同尺度的问题
- ✅ **平衡各个 retriever**：每个 retriever 的贡献相对均衡
- ✅ **对排名靠前的更敏感**：自动给予高排名结果更多权重
- ✅ **理论基础扎实**：在信息检索领域被广泛验证和使用
- ✅ **民主投票机制**：被多个 retriever 认可的结果会获得更高分数

### 缺点

- ❌ **忽略原始分数**：如果原始相似度分数很重要，RRF 会丢失这些信息
  - 例如：第1名分数 0.95 和第1名分数 0.65 在 RRF 中贡献相同
- ❌ **需要完整排序**：必须对每个 retriever 的结果进行排序
- ❌ **参数 k 需要调整**：k=60 是经验值，不同场景可能需要调优

### 适用场景

- 🎯 **不同类型的 retriever**：向量检索 + 关键词检索（最经典场景）
- 🎯 **分数不可比**：多个 retriever 的分数尺度完全不同
- 🎯 **排名比分数更重要**：更关心"哪些结果排在前面"
- 🎯 **混合检索 (Hybrid Search)**：业界标准做法

### 使用示例

```python
from zag.retrievers import VectorRetriever, KeywordRetriever, QueryFusionRetriever, FusionMode

# 向量检索 + 关键词检索
vector_retriever = VectorRetriever(vector_store=chroma_store)
keyword_retriever = KeywordRetriever(keyword_store=meilisearch_store)

# 使用 RRF 融合（经典 hybrid search）
fusion = QueryFusionRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    mode=FusionMode.RECIPROCAL_RANK,
    top_k=10
)

results = fusion.retrieve("semantic search algorithms")
```

---

## 3. RELATIVE_SCORE - 相对分数融合

### 原理

通过 MinMax 归一化将不同 retriever 的分数统一到相同尺度，然后应用权重进行加权融合。

### 算法流程

**步骤 1：MinMax 归一化**
```python
for each retriever:
    scores = [unit.score for unit in results]
    min_score = min(scores)
    max_score = max(scores)
    
    for unit in results:
        normalized_score = (unit.score - min_score) / (max_score - min_score)
```

**步骤 2：应用权重**
```python
weighted_score = normalized_score × retriever_weight
```

**步骤 3：累加分数**
```python
for each unit:
    if unit appears in multiple retrievers:
        final_score = sum(weighted_scores from all retrievers)
```

### 归一化示例

假设 Retriever A 的分数范围是 [0.3, 0.8]，某个 unit 得分 0.65：

```python
normalized = (0.65 - 0.3) / (0.8 - 0.3) = 0.35 / 0.5 = 0.7
```

如果权重是 0.6，则加权分数为：`0.7 × 0.6 = 0.42`

### 优点

- ✅ **保留分数信息**：归一化后仍保留原始分数的相对关系
- ✅ **支持权重调节**：可以根据业务需求调整不同 retriever 的重要性
- ✅ **分数可解释**：最终分数是归一化后的加权和，含义清晰
- ✅ **适合相同类型**：多个同类 retriever 但来源不同时效果好
- ✅ **精细控制**：可以通过权重精确控制每个数据源的影响力

### 缺点

- ❌ **对异常值敏感**：MinMax 归一化容易受极值影响
  - 示例：如果某个 unit 得分特别高（异常值），会压缩其他所有分数
- ❌ **需要合理权重**：权重设置不当会导致某些 retriever 被边缘化
- ❌ **计算开销稍大**：需要对每个 retriever 的结果计算 min/max

### 适用场景

- 🎯 **多个相同类型的 retriever**：多个 VectorRetriever（不同向量库）
- 🎯 **有明确的权重偏好**：某个数据源更可靠，想给更高权重
- 🎯 **分数有实际意义**：相似度分数的绝对值很重要
- 🎯 **联邦检索**：从多个独立的向量库中检索并融合

### 使用示例

```python
from zag.retrievers import VectorRetriever, QueryFusionRetriever, FusionMode

# 两个不同的向量数据库
chroma_retriever = VectorRetriever(vector_store=chroma_store)
pinecone_retriever = VectorRetriever(vector_store=pinecone_store)

# 使用相对分数融合，给 Chroma 更高权重（更可靠）
fusion = QueryFusionRetriever(
    retrievers=[chroma_retriever, pinecone_retriever],
    mode=FusionMode.RELATIVE_SCORE,
    top_k=10,
    retriever_weights=[0.7, 0.3]  # Chroma: 70%, Pinecone: 30%
)

results = fusion.retrieve("vector database comparison")
```

---

## 对比总结

### 特性对比表

| 维度 | SIMPLE | RECIPROCAL_RANK | RELATIVE_SCORE |
|------|--------|-----------------|----------------|
| **计算复杂度** | 低 | 中 | 中高 |
| **是否需要归一化** | 否 | 否（仅用排名） | 是 |
| **是否支持权重** | 否 | 否 | 是 |
| **对分数尺度敏感度** | 高 | 低 | 低（归一化后） |
| **是否保留原始分数语义** | 是 | 否 | 部分保留 |
| **适合异构检索** | ❌ | ✅ | ❌ |
| **适合同构检索** | ✅ | ⚠️ | ✅ |
| **可解释性** | 高 | 中 | 高 |

### 性能对比

| 策略 | 时间复杂度 | 空间复杂度 | 备注 |
|------|-----------|-----------|------|
| SIMPLE | O(n) | O(n) | n 为总结果数 |
| RECIPROCAL_RANK | O(n log n) | O(n) | 需要排序 |
| RELATIVE_SCORE | O(n) | O(n) | 需要两次遍历（min/max + 归一化）|

---

## 选择建议

### 决策树

```
是否是不同类型的 retriever (如向量+关键词)?
├─ 是 → 使用 RECIPROCAL_RANK
└─ 否 → 是否需要精细控制权重?
    ├─ 是 → 使用 RELATIVE_SCORE
    └─ 否 → 分数是否在同一尺度?
        ├─ 是 → 使用 SIMPLE (最快)
        └─ 否 → 使用 RELATIVE_SCORE
```

### 场景推荐

#### 1. 多个向量数据库（相同 embedder）

**推荐：RELATIVE_SCORE**

```python
fusion = QueryFusionRetriever(
    retrievers=[chroma_retriever, pinecone_retriever],
    mode=FusionMode.RELATIVE_SCORE,
    retriever_weights=[0.6, 0.4]  # Chroma 更可靠
)
```

**原因**：
- 同类型 retriever，但分数尺度可能不同
- 可以根据数据源质量设置权重
- 保留分数的相对关系

#### 2. 向量检索 + 关键词检索（混合检索）

**推荐：RECIPROCAL_RANK**

```python
fusion = QueryFusionRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    mode=FusionMode.RECIPROCAL_RANK
)
```

**原因**：
- 不同类型检索，分数完全不可比
- RRF 是业界公认的 hybrid search 标准
- 不需要人工调整权重

#### 3. 快速原型，简单去重

**推荐：SIMPLE**

```python
fusion = QueryFusionRetriever(
    retrievers=[retriever1, retriever2],
    mode=FusionMode.SIMPLE
)
```

**原因**：
- 实现最简单，调试方便
- 性能最好
- 适合快速验证想法

#### 4. 多数据源联邦检索

**推荐：RELATIVE_SCORE**

```python
fusion = QueryFusionRetriever(
    retrievers=[
        internal_kb_retriever,    # 内部知识库
        public_docs_retriever,    # 公开文档
        user_docs_retriever,      # 用户文档
    ],
    mode=FusionMode.RELATIVE_SCORE,
    retriever_weights=[0.5, 0.3, 0.2]  # 按可信度分配权重
)
```

**原因**：
- 需要精细控制不同数据源的影响力
- 各数据源重要性不同
- 分数归一化后可比较

---

## 实现细节

### SIMPLE 实现

```python
def _simple_fusion(self, results: list[list[BaseUnit]]) -> list[BaseUnit]:
    all_units: dict[str, BaseUnit] = {}
    
    for units in results:
        for unit in units:
            unit_id = unit.unit_id
            unit_score = unit.score or 0.0
            
            if unit_id in all_units:
                # 保留最高分
                existing_score = all_units[unit_id].score or 0.0
                if unit_score > existing_score:
                    all_units[unit_id] = unit
            else:
                all_units[unit_id] = unit
    
    return sorted(all_units.values(), key=lambda x: x.score or 0.0, reverse=True)
```

### RECIPROCAL_RANK 实现

```python
def _reciprocal_rank_fusion(self, results: list[list[BaseUnit]]) -> list[BaseUnit]:
    k = 60.0  # RRF 常数
    fused_scores: dict[str, float] = {}
    id_to_unit: dict[str, BaseUnit] = {}
    
    for units in results:
        sorted_units = sorted(units, key=lambda x: x.score or 0.0, reverse=True)
        
        for rank, unit in enumerate(sorted_units):
            unit_id = unit.unit_id
            id_to_unit[unit_id] = unit
            
            if unit_id not in fused_scores:
                fused_scores[unit_id] = 0.0
            
            # RRF 公式
            fused_scores[unit_id] += 1.0 / (k + rank)
    
    # 按融合分数排序
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    result_units = []
    for unit_id, score in sorted_ids:
        unit = id_to_unit[unit_id].model_copy()
        unit.score = score
        result_units.append(unit)
    
    return result_units
```

### RELATIVE_SCORE 实现

```python
def _relative_score_fusion(self, results: list[list[BaseUnit]]) -> list[BaseUnit]:
    all_units: dict[str, BaseUnit] = {}
    
    for i, units in enumerate(results):
        if not units:
            continue
        
        # 提取分数并归一化
        scores = [unit.score or 0.0 for unit in units]
        min_score = min(scores)
        max_score = max(scores)
        
        for unit in units:
            unit_id = unit.unit_id
            original_score = unit.score or 0.0
            
            # MinMax 归一化
            if max_score == min_score:
                normalized_score = 1.0 if max_score > 0 else 0.0
            else:
                normalized_score = (original_score - min_score) / (max_score - min_score)
            
            # 应用权重
            weighted_score = normalized_score * self.retriever_weights[i]
            
            # 累加分数
            if unit_id in all_units:
                all_units[unit_id].score += weighted_score
            else:
                unit_copy = unit.model_copy()
                unit_copy.score = weighted_score
                all_units[unit_id] = unit_copy
    
    return sorted(all_units.values(), key=lambda x: x.score or 0.0, reverse=True)
```

---

## 常见问题

### Q1: 为什么 RRF 的 k 值是 60？

**A**: k=60 是信息检索领域的经验值，来自多年的实践验证。它的作用是：
- 平滑排名差异：避免排名靠前的结果分数过高
- 保持区分度：同时保证排名差异仍能体现在分数上

你可以根据实际情况调整 k 值：
- **k 越小**：排名靠前的结果权重越高（更激进）
- **k 越大**：排名差异影响变小（更保守）

### Q2: RELATIVE_SCORE 中权重如何设置？

**A**: 权重设置建议：
1. **根据数据源质量**：更可靠的数据源给更高权重
2. **根据数据源大小**：数据量大的可以给稍高权重
3. **根据业务重要性**：核心业务数据源给更高权重
4. **A/B 测试调优**：通过实验找到最优权重组合

示例：
```python
# 三个数据源：内部知识库、公开文档、用户上传
weights=[0.5, 0.3, 0.2]  # 内部知识库最可靠，用户上传最不确定
```

### Q3: 可以混合使用不同策略吗？

**A**: 可以！`QueryFusionRetriever` 本身也是一个 `BaseRetriever`，可以嵌套使用：

```python
# 先用 RRF 融合向量和关键词检索
hybrid1 = QueryFusionRetriever(
    retrievers=[vector1, keyword1],
    mode=FusionMode.RECIPROCAL_RANK
)

hybrid2 = QueryFusionRetriever(
    retrievers=[vector2, keyword2],
    mode=FusionMode.RECIPROCAL_RANK
)

# 再用 RELATIVE_SCORE 融合两个混合检索器
final_fusion = QueryFusionRetriever(
    retrievers=[hybrid1, hybrid2],
    mode=FusionMode.RELATIVE_SCORE,
    retriever_weights=[0.6, 0.4]
)
```

### Q4: 如何评估不同策略的效果？

**A**: 评估方法：
1. **离线评估**：
   - 准备测试集（query + 相关文档标注）
   - 计算 MRR (Mean Reciprocal Rank)、NDCG 等指标
   - 对比不同策略的指标

2. **在线 A/B 测试**：
   - 线上分流不同策略
   - 收集用户反馈（点击率、满意度等）
   - 选择表现最好的策略

3. **人工评估**：
   - 抽样检查检索结果
   - 评估相关性和排序质量

---

## 参考资料

### 学术论文

1. **RRF 算法原论文**:
   - Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."

2. **混合检索综述**:
   - Zamani, H., et al. (2022). "Retrieval-Enhanced Machine Learning."

### 相关链接

- [Elasticsearch RRF 文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)
- [LlamaIndex Fusion Retriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion/)
- [Vector Search vs Keyword Search](https://www.pinecone.io/learn/hybrid-search-intro/)

---

## 更新日志

- **2026-01-05**: 初始版本，文档化三种融合策略
