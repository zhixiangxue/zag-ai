# Extractor 设计文档

## 1. 什么是 Extractor？

**Extractor（提取器）是用来给 Unit 自动添加元数据（metadata）的组件**。

### 核心职责

```
输入：Unit（文档分块）
处理：提取额外的元数据
输出：Unit + 增强的 metadata
```

### 设计原则

- ✅ **框架无关**：不耦合具体业务场景
- ✅ **用户可控**：数据结构由开发者定义（Pydantic BaseModel）
- ✅ **复用优先**：利用已有的结构化数据（如 TableUnit）
- ✅ **统一接口**：使用 URI-based LLM 语法（和 embedder 一致）

---

## 2. LLM URI 语法

**统一使用 `provider/model` 格式**（和 embedder 保持一致）：

```python
# OpenAI
llm_uri = "openai/gpt-4o-mini"

# Anthropic
llm_uri = "anthropic/claude-3-5-sonnet"

# Ollama (本地)
llm_uri = "ollama/llama3.2"

# Bailian
llm_uri = "bailian/qwen-plus"
```

**内部适配**：
- zag 内部根据需求选择适配器：
  - **instructor**：用于结构化提取（Pydantic）
  - **chak**：用于普通对话（关键词、摘要等）

**开发者无感知**：
- 只需提供 URI + API key
- zag 自动选择合适的底层实现

---

## 3. 核心 Extractor（MVP）

### 3.1 **TableExtractor**（表格提取器） ⭐ 核心

**作用**：为 TableUnit 生成自然语言描述，提升向量检索效果

**关键设计**：
- ✅ 复用 TableUnit.json_data（已结构化）
- ✅ 不重复解析表格
- ✅ 生成 table_summary 用于向量检索
- ✅ 不重复存储数据（json_data 已在 TableUnit 中）

---

### 3.2 **StructuredExtractor**（结构化提取器） ⭐ 核心

**作用**：让开发者自定义 Pydantic 模型，从 Unit 中提取结构化信息

**关键设计**：
- ✅ 使用 **instructor** 实现（不自己造轮子）
- ✅ 数据结构由开发者定义
- ✅ 使用 URI-based LLM 语法

---

### 3.3 **KeywordExtractor**（关键词提取器） ⭐ 核心

**作用**：提取关键词，用于混合检索

**关键设计**：
- ✅ 使用 **chak** 实现普通对话
- ✅ 提示词可自定义
- ✅ 使用 URI-based LLM 语法

---

## 4. 接口设计

### 4.1 `extractors/base.py`：基类

```python
from abc import abstractmethod
from typing import List, Dict, Sequence

class BaseExtractor:
    """Extractor 基类"""
    
    @abstractmethod
    async def aextract(self, units: Sequence) -> List[Dict]:
        """提取元数据（异步）
        
        Args:
            units: Unit 列表
            
        Returns:
            元数据字典列表
        """
        pass
    
    def extract(self, units: Sequence) -> List[Dict]:
        """提取元数据（同步）"""
        import asyncio
        return asyncio.run(self.aextract(units))
    
    def __call__(self, units: Sequence) -> List:
        """处理 Unit：提取元数据并更新"""
        metadata_list = self.extract(units)
        
        for unit, metadata in zip(units, metadata_list):
            unit.metadata.update(metadata)
        
        return units
```

---

### 4.2 `extractors/table.py`：表格提取器

```python
from extractors.base import BaseExtractor
from typing import List, Dict, Sequence
from zag.schemas import TableUnit
import chak

class TableExtractor(BaseExtractor):
    """表格提取器
    
    为 TableUnit 生成自然语言描述，提升向量检索效果。
    
    关键设计：
    - 复用 TableUnit.json_data（已结构化）
    - 不重复解析表格
    - 只生成 table_summary，不重复存储 json_data
    
    Args:
        llm_uri: LLM URI，格式：provider/model
                 如："openai/gpt-4o-mini", "bailian/qwen-plus"
        api_key: API Key
    
    Example:
        extractor = TableExtractor(
            llm_uri="openai/gpt-4o-mini",
            api_key="sk-xxx"
        )
        units = extractor(units)
    """
    
    def __init__(self, llm_uri: str, api_key: str):
        self.llm_uri = llm_uri
        self.api_key = api_key
        # 创建 chak 会话（用于普通对话）
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit) -> Dict:
        """从单个 Unit 提取表格信息"""
        if not isinstance(unit, TableUnit):
            return {}
        
        # 复用 TableUnit 的结构化数据
        json_data = unit.json_data
        if not json_data:
            return {}
        
        # 生成自然语言描述
        prompt = f"""以下是一个表格的结构化数据：

{json_data}

请用 2-3 句话总结这个表格的内容，突出关键数据和对比关系。
要求：使用完整的句子，便于向量检索。

摘要："""
        
        response = await self._conv.asend(prompt)
        
        # 只返回 summary，不重复存储 json_data
        return {"table_summary": response.content.strip()}
    
    async def aextract(self, units: Sequence) -> List[Dict]:
        """批量提取"""
        import asyncio
        tasks = [self._extract_from_unit(unit) for unit in units]
        return await asyncio.gather(*tasks)
```

---

### 4.3 `extractors/structured.py`：结构化提取器

```python
from extractors.base import BaseExtractor
from typing import List, Dict, Sequence, Type
from pydantic import BaseModel
import instructor

class StructuredExtractor(BaseExtractor):
    """结构化提取器
    
    使用 instructor 从 Unit 中提取结构化信息。
    
    关键设计：
    - 使用 instructor（专业的结构化提取库）
    - 数据结构由开发者定义（Pydantic BaseModel）
    - 自动重试、验证
    
    Args:
        llm_uri: LLM URI，格式：provider/model
                 如："openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"
        api_key: API Key
        schema: Pydantic BaseModel 类（开发者自定义）
        max_retries: 验证失败时的最大重试次数，默认 3
    
    Example:
        # 1. 定义数据结构（开发者自定义）
        class ProductInfo(BaseModel):
            product_name: str
            apr_min: float
            loan_term_years: int
        
        # 2. 创建提取器
        extractor = StructuredExtractor(
            llm_uri="openai/gpt-4o-mini",
            api_key="sk-xxx",
            schema=ProductInfo
        )
        
        # 3. 使用
        units = extractor(units)
        
        # 4. 访问结果（直接是 schema 的字段）
        print(units[0].metadata["product_name"])
        print(units[0].metadata["apr_min"])
        # "30-Year Fixed Rate Mortgage"
        # 6.5
    """
    
    def __init__(
        self,
        llm_uri: str,
        api_key: str,
        schema: Type[BaseModel],
        max_retries: int = 3,
    ):
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.schema = schema
        self.max_retries = max_retries
        
        # 使用 instructor（支持 provider/model URI）
        self._client = instructor.from_provider(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit) -> Dict:
        """从单个 Unit 提取结构化信息"""
        text = unit.text if hasattr(unit, 'text') else str(unit.content)
        
        prompt = f"从以下文本中提取结构化信息：\n\n{text}"
        
        try:
            # 使用 instructor 提取结构化数据
            result = await self._client.chat.completions.create(
                response_model=self.schema,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_retries=self.max_retries,
            )
            
            # 直接展开 schema 字段，不包裹中间层
            return result.model_dump()
        except Exception as e:
            print(f"Warning: Failed to extract structured data: {e}")
            return {}
    
    async def aextract(self, units: Sequence) -> List[Dict]:
        """批量提取"""
        import asyncio
        tasks = [self._extract_from_unit(unit) for unit in units]
        return await asyncio.gather(*tasks)
```

---

### 4.4 `extractors/keyword.py`：关键词提取器

```python
from extractors.base import BaseExtractor
from typing import List, Dict, Sequence
import chak

DEFAULT_TEMPLATE = """从以下文本中提取 {num_keywords} 个最重要的关键词。

文本：
{text}

要求：
1. 关键词应能唯一标识这段文本的核心主题
2. 返回JSON 数组格式，如：["keyword1", "keyword2", "keyword3"]
3. 使用原文语言

关键词数组："""


class KeywordExtractor(BaseExtractor):
    """关键词提取器
    
    使用 LLM 提取关键词，返回列表格式。
    
    Args:
        llm_uri: LLM URI，格式：provider/model
        api_key: API Key
        num_keywords: 提取的关键词数量，默认 5
    
    Example:
        extractor = KeywordExtractor(
            llm_uri="openai/gpt-4o-mini",
            api_key="sk-xxx",
            num_keywords=5
        )
        units = extractor(units)
        
        # 访问结果（列表格式）
        print(units[0].metadata["excerpt_keywords"])
        # ["fixed rate", "30-year", "mortgage", "refinance", "conventional loan"]
    """
    
    def __init__(
        self,
        llm_uri: str,
        api_key: str,
        num_keywords: int = 5,
    ):
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.num_keywords = num_keywords
        
        # 使用 chak（用于普通对话）
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit) -> Dict:
        """从单个 Unit 提取关键词"""
        text = unit.text if hasattr(unit, 'text') else str(unit.content)
        
        prompt = DEFAULT_TEMPLATE.format(
            text=text,
            num_keywords=self.num_keywords
        )
        
        response = await self._conv.asend(prompt)
        
        # 解析 JSON 数组
        import json
        try:
            keywords = json.loads(response.content.strip())
            if not isinstance(keywords, list):
                # 兜底：如果不是列表，尝试按逗号分割
                keywords = [k.strip() for k in response.content.strip().split(',')]
        except json.JSONDecodeError:
            # 兜底：按逗号分割
            keywords = [k.strip() for k in response.content.strip().split(',')]
        
        return {"excerpt_keywords": keywords[:self.num_keywords]}
    
    async def aextract(self, units: Sequence) -> List[Dict]:
        """批量提取"""
        import asyncio
        tasks = [self._extract_from_unit(unit) for unit in units]
        return await asyncio.gather(*tasks)
```

---

## 5. 目录结构

```
zag/
└── extractors/
    ├── __init__.py           # 导出所有 Extractor
    ├── base.py               # BaseExtractor
    ├── table.py              # TableExtractor（使用 chak）
    ├── structured.py         # StructuredExtractor（使用 instructor）
    └── keyword.py            # KeywordExtractor（使用 chak）
```

---

## 6. 使用示例

### 6.1 表格提取

```python
from zag.extractors import TableExtractor

# 创建提取器（URI-based）
extractor = TableExtractor(
    llm_uri="openai/gpt-4o-mini",
    api_key="sk-xxx"
)

# 使用
units = extractor(units)

# 查看结果
for unit in units:
    if isinstance(unit, TableUnit):
        print(f"Summary: {unit.metadata['table_summary']}")
        print(f"Structured Data: {unit.json_data}")  # 结构化数据在 json_data 字段
```

---

### 6.2 结构化提取

```python
from pydantic import BaseModel, Field
from zag.extractors import StructuredExtractor

# 1. 定义数据结构（开发者自定义）
class ProductInfo(BaseModel):
    product_name: str = Field(description="产品名称")
    apr_min: float = Field(description="最低年利率（百分比）")
    apr_max: float = Field(description="最高年利率（百分比）")
    loan_term_years: int = Field(description="贷款期限（年）")

# 2. 创建提取器
extractor = StructuredExtractor(
    llm_uri="openai/gpt-4o-mini",
    api_key="sk-xxx",
    schema=ProductInfo
)

# 3. 使用
units = extractor(units)

# 4. 查看结果（直接展开，无中间层）
print(units[0].metadata["product_name"])
print(units[0].metadata["apr_min"])
# "30-Year Fixed Rate Mortgage"
# 6.5
```

---

### 6.3 关键词提取

```python
from zag.extractors import KeywordExtractor

# 创建提取器
extractor = KeywordExtractor(
    llm_uri="openai/gpt-4o-mini",
    api_key="sk-xxx",
    num_keywords=5
)

# 使用
units = extractor(units)

# 查看结果（列表格式）
print(units[0].metadata["excerpt_keywords"])
# ["fixed rate", "30-year", "mortgage", "refinance", "conventional loan"]
```

---

### 6.4 串联使用

```python
from zag.extractors import TableExtractor, StructuredExtractor, KeywordExtractor

# 统一使用 URI-based 接口
llm_uri = "openai/gpt-4o-mini"
api_key = "sk-xxx"

# 创建提取器
table_extractor = TableExtractor(llm_uri=llm_uri, api_key=api_key)
product_extractor = StructuredExtractor(
    llm_uri=llm_uri,
    api_key=api_key,
    schema=ProductInfo
)
keyword_extractor = KeywordExtractor(llm_uri=llm_uri, api_key=api_key)

# 串联执行
units = table_extractor(units)
units = product_extractor(units)
units = keyword_extractor(units)
```

---

## 7. 完整流程示例

```python
from pydantic import BaseModel, Field
from zag.readers import PDFReader
from zag.splitters import MarkdownSplitter
from zag.extractors import TableExtractor, StructuredExtractor, KeywordExtractor
from zag.embedders import Embedder
from zag.indexers import VectorIndexer
from zag.storages.vector import ChromaStore

# 1. 定义数据结构（开发者自定义）
class LoanProduct(BaseModel):
    product_name: str = Field(description="产品名称")
    apr_min: float = Field(description="最低年利率")
    loan_term_years: int = Field(description="贷款期限")

# 2. 配置
llm_uri = "openai/gpt-4o-mini"
api_key = "sk-xxx"

# 3. 创建提取器
table_extractor = TableExtractor(llm_uri=llm_uri, api_key=api_key)
product_extractor = StructuredExtractor(
    llm_uri=llm_uri,
    api_key=api_key,
    schema=LoanProduct
)
keyword_extractor = KeywordExtractor(llm_uri=llm_uri, api_key=api_key)

# 4. 处理文档
reader = PDFReader(use_docling=True)
units = reader.read("loan_documents/*.pdf")

splitter = MarkdownSplitter(chunk_size=1000)
units = splitter.split(units)

# 5. 提取元数据
units = table_extractor(units)      # 处理表格
units = product_extractor(units)    # 提取产品信息
units = keyword_extractor(units)    # 提取关键词

# 6. 向量化
embedder = Embedder(uri="openai://text-embedding-3-small")
units = embedder.embed(units)

# 7. 索引
store = ChromaStore()
indexer = VectorIndexer(vector_store=store)
indexer.index(units)

# 8. 查询（支持精确过滤）
results = store.query(
    query_text="30-year fixed rate mortgage",
    filters={
        "loan_product.apr_min": {"$lt": 7.0},
    },
    top_k=5
)
```

---

## 8. 内部实现：LLM Adapter（可选）

如果需要更灵活的适配，可以实现一个统一的 LLM Adapter：

```python
# zag/llm/adapter.py
class LLMAdapter:
    """LLM 适配器，统一 URI-based 接口"""
    
    @staticmethod
    def get_instructor_client(llm_uri: str, api_key: str):
        """获取 instructor client（用于结构化提取）"""
        import instructor
        return instructor.from_provider(llm_uri, api_key=api_key)
    
    @staticmethod
    def get_chak_conversation(llm_uri: str, api_key: str):
        """获取 chak Conversation（用于普通对话）"""
        import chak
        return chak.Conversation(llm_uri, api_key=api_key)
```

---

## 9. 依赖管理

在 `pyproject.toml` 中添加依赖：

```toml
[project]
dependencies = [
    # ... 现有依赖
]

[project.optional-dependencies]
extractors = [
    "instructor>=1.0.0",
    "chakpy>=0.2.0",
]

all = [
    "instructor>=1.0.0",
    "chakpy>=0.2.0",
    # ... 其他可选依赖
]
```

**安装**：
```bash
# 基础安装（不包含 extractors）
pip install zag

# 安装 extractors 支持
pip install zag[extractors]

# 或安装全部
pip install zag[all]
```

---

## 10. MVP 清单

- [ ] `extractors/base.py`：BaseExtractor
- [ ] `extractors/table.py`：TableExtractor（使用 chak）
- [ ] `extractors/structured.py`：StructuredExtractor（使用 instructor）
- [ ] `extractors/keyword.py`：KeywordExtractor（使用 chak）
- [ ] 添加依赖：instructor, chakpy
- [ ] 单元测试
- [ ] 使用文档

---

## 11. 总结

### 核心设计

1. ✅ **统一 URI 语法**：`provider/model`（和 embedder 一致）
2. ✅ **框架无关**：数据结构由开发者定义
3. ✅ **专业工具**：
   - instructor：结构化提取
   - chak：普通对话
4. ✅ **接口简洁**：只需提供 URI + API key

### 与之前设计的差异

| 维度 | 之前的设计 | 新设计 |
|------|-----------|--------|
| **LLM 接口** | ❌ 传入 `conv` 对象 | ✅ 使用 URI-based 语法 |
| **结构化提取** | ❌ 自己实现 | ✅ 使用 instructor |
| **普通对话** | ✅ 使用 chak | ✅ 使用 chak |
| **一致性** | ❌ 和 embedder 不统一 | ✅ 和 embedder 统一 |

**简单、统一、专业！**
