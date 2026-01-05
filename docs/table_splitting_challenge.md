# Table Splitting Challenge: 表格处理的多层级设计挑战

## 问题背景

在基于 Markdown 的文档处理中（特别是通过 MinerU/Docling 解析 PDF 后生成的 Markdown），经常会遇到 **HTML 格式的表格**（因为 Markdown 表格不支持 rowspan/colspan）。这带来了一系列设计挑战：

1. **Splitter 能否正确切分包含 HTML 表格的内容？**
2. **如何区分 Markdown 表格和 HTML 表格？**
3. **TableExtractor 如何精准总结复杂 HTML 表格？**
4. **复杂表格会引起哪些连锁反应？**

---

## 核心挑战拆解

### 1️⃣ MarkdownHeaderSplitter 对 HTML 表格的处理能力

**当前实现**: `zag/splitters/markdown/header_based.py`

**能力评估**:
- ✅ 正确处理 Markdown 标准表格（`| col1 | col2 |`）
- ✅ 正确处理 HTML 表格块（作为普通文本内容保留）
- ✅ 避免在代码块内解析标题
- ❌ **不会识别表格的语义边界**，只会按标题切分

**示例场景**:

```markdown
## 产品对比

这是一段介绍文字。

<table>
  <tr><td>Product</td><td>Price</td></tr>
  <tr><td>A</td><td>100</td></tr>
</table>

这是表格后的总结。

## 下一章节
...
```

**当前行为**: 整个 "产品对比" 章节（包括 HTML 表格）会被切分为**一个 TextUnit**，表格的 HTML 代码会保留在 `content` 中。

**问题**: 表格没有被识别为独立的语义单元（TableUnit），影响后续的：
- 向量检索效果（表格内容被稀释在大段文本中）
- 元数据提取（无法针对表格做特殊处理）
- 结构化查询（无法过滤表格类型的 Unit）

---

### 2️⃣ 如何区分 Markdown 表格和 HTML 表格

#### **方案 A: 在 Splitter 中增强表格识别**

创建一个 `TableAwareSplitter`，能够：

```python
class TableAwareSplitter(BaseSplitter):
    """
    Markdown splitter that recognizes both MD and HTML tables
    
    Logic:
    1. Parse markdown by headers (current logic)
    2. Within each section, detect tables:
       - Markdown tables: | col | col |
       - HTML tables: <table>...</table>
    3. Extract tables as separate TableUnits
    4. Keep text before/after tables as TextUnits
    """
    
    def _detect_tables(self, content: str) -> list[dict]:
        """Detect both MD and HTML tables"""
        tables = []
        
        # Detect HTML tables
        html_pattern = r'<table[^>]*>.*?</table>'
        for match in re.finditer(html_pattern, content, re.DOTALL | re.IGNORECASE):
            tables.append({
                'type': 'html',
                'content': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Detect Markdown tables (GFM style)
        md_table_pattern = r'(?:^|\n)(\|.+\|(?:\n\|[-:\s|]+\|)(?:\n\|.+\|)*)'
        for match in re.finditer(md_table_pattern, content, re.MULTILINE):
            tables.append({
                'type': 'markdown',
                'content': match.group(1),
                'start': match.start(),
                'end': match.end()
            })
        
        return sorted(tables, key=lambda x: x['start'])
    
    def _do_split(self, document) -> list[BaseUnit]:
        """
        Split markdown with table awareness
        
        1. Detect all tables in content
        2. Split content into segments:
           - Text segments → TextUnit
           - Table segments → TableUnit
        3. Maintain context_path for each unit
        """
        content = document.content
        tables = self._detect_tables(content)
        
        units = []
        last_pos = 0
        
        for table in tables:
            # Add text before table
            if table['start'] > last_pos:
                text_content = content[last_pos:table['start']].strip()
                if text_content:
                    units.append(TextUnit(
                        content=text_content,
                        # ... metadata
                    ))
            
            # Add table unit
            units.append(TableUnit(
                content=table['content'],
                json_data={'table_type': table['type']},
                # ... metadata
            ))
            
            last_pos = table['end']
        
        # Add remaining text
        if last_pos < len(content):
            remaining = content[last_pos:].strip()
            if remaining:
                units.append(TextUnit(content=remaining))
        
        return units
```

**优点**:
- 能处理纯 Markdown 文档（没有经过 Reader 的场景）
- 统一的 Splitter 接口

**缺点**:
- 正则表达式可能不够鲁棒（嵌套表格、表格属性等）
- 重复了 Reader 的工作（MinerU/Docling 已经识别表格）

---

#### **方案 B: 在 Reader 中直接构建 TableUnit（推荐）**

MinerU 和 Docling 这样的 Reader **已经做了表格识别**，它们的输出中明确标注了表格：

**MinerU 的 content_list 输出**:
```python
{
    "type": "table",
    "html": "<table>...</table>",  # ← 已经识别出来了！
    "latex": "\\begin{tabular}...",
    "page_idx": 2,
    "bbox": [x, y, w, h]
}
```

**Docling 的输出**:
```python
# DoclingDocument 中的 TableItem
table_item = {
    "type": "table",
    "data": {
        "grid": [[cell, cell, ...], ...],
        "num_rows": 5,
        "num_cols": 3
    },
    "prov": [{"bbox": {...}}]
}
```

**最优方案**: 在 Reader 层面（如 `MinerUReader._build_pages_from_content_list`）就构建 `TableUnit`：

```python
def _build_pages_from_content_list(self, content_list: list[dict]) -> list[Page]:
    """Build Page objects with TableUnits"""
    
    page_items = {}
    
    for item in content_list:
        page_num = item.get("page_idx", 0) + 1
        
        if page_num not in page_items:
            page_items[page_num] = {
                "units": []  # 改为 units 列表，统一存储
            }
        
        # Classify item type
        item_type = item.get("type", "text")
        
        if item_type == "text":
            # 创建 TextUnit
            unit = TextUnit(
                unit_id=self.generate_unit_id(),
                content=item.get("text", ""),
                metadata=UnitMetadata(
                    context_path=f"Page{page_num}",
                    custom={
                        "layout_type": item.get("layout_type", "text"),
                        "bbox": item.get("bbox")
                    }
                )
            )
            page_items[page_num]["units"].append(unit)
            
        elif item_type == "table":
            # 创建 TableUnit
            unit = TableUnit(
                unit_id=self.generate_unit_id(),
                content=item.get("html", ""),  # HTML 表格
                json_data={
                    "table_type": "html",
                    "raw_html": item.get("html"),
                    "latex": item.get("latex"),  # MinerU 还提供 LaTeX
                    "bbox": item.get("bbox"),
                    "page_idx": item.get("page_idx")
                },
                metadata=UnitMetadata(
                    context_path=f"Page{page_num}/Table",
                    custom={"bbox": item.get("bbox")}
                )
            )
            page_items[page_num]["units"].append(unit)
            
        elif item_type in ["image", "figure"]:
            # 创建 ImageUnit
            unit = ImageUnit(
                unit_id=self.generate_unit_id(),
                content=b"",  # 需要读取图片二进制
                format="png",
                caption=item.get("caption"),
                metadata=UnitMetadata(
                    context_path=f"Page{page_num}/Image",
                    custom={
                        "path": item.get("img_path"),
                        "bbox": item.get("bbox")
                    }
                )
            )
            page_items[page_num]["units"].append(unit)
    
    # Create Page objects
    pages = []
    for page_num in sorted(page_items.keys()):
        units = page_items[page_num]["units"]
        
        # 构建 Unit 链表关系
        for i in range(len(units)):
            if i > 0:
                units[i].prev_unit_id = units[i - 1].unit_id
            if i < len(units) - 1:
                units[i].next_unit_id = units[i + 1].unit_id
        
        pages.append(Page(
            page_number=page_num,
            content=units,  # 直接存储 Unit 列表
            metadata={
                "unit_count": len(units)
            }
        ))
    
    return pages
```

**优点**:
- 复用 Reader 的高精度解析（MinerU 82-90+ 准确率）
- TableUnit 中包含丰富的结构化信息（bbox, latex, etc.）
- 避免重复解析
- Splitter 不需要做表格识别（只需按需切分 TextUnit）

**缺点**:
- 需要修改 Reader 实现
- Page.content 的数据结构需要调整（从 dict 改为 list[BaseUnit]）

---

### 3️⃣ TableExtractor 如何精准总结复杂 HTML 表格

**当前实现**: `zag/extractors/table.py`

**当前 Prompt** (line 58-65):
```python
prompt = f"""以下是一个表格的结构化数据：

{json_data}  # ← 这里是什么？

请用 2-3 句话总结这个表格的内容，突出关键数据和对比关系。
"""
```

**问题**:
1. `json_data` 的格式是什么？如果是 HTML 字符串，LLM 可能理解不佳
2. 没有表格的上下文信息（标题、前后段落）
3. 对于复杂表格（多层表头、合并单元格），可能丢失结构信息

---

#### **改进方案 1: 增强 JSON 结构**

修改 Reader，让 `TableUnit.json_data` 包含结构化信息：

```python
# 在 MinerUReader 中
{
    "table_type": "html",  # or "markdown"
    "raw_html": "<table>...</table>",
    "parsed_structure": {
        "headers": ["Product", "Q1", "Q2", "Q3"],
        "rows": [
            ["ProductA", "100", "120", "130"],
            ["ProductB", "80", "90", "95"]
        ],
        "merged_cells": [
            {"row": 0, "col": 0, "rowspan": 2, "colspan": 1}
        ]
    },
    "context": {
        "preceding_text": "下表展示了季度销售数据：",
        "following_text": "从表格可以看出，ProductA 增长更快。"
    }
}
```

**增强的 TableExtractor**:

```python
class TableExtractor(BaseExtractor):
    """Enhanced table extractor with HTML parsing"""
    
    def __init__(self, llm_uri: str, api_key: str):
        self.llm_uri = llm_uri
        self.api_key = api_key
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit) -> Dict:
        if not isinstance(unit, TableUnit):
            return {}
        
        json_data = unit.json_data
        if not json_data:
            return {}
        
        # 检测表格类型
        table_type = json_data.get("table_type", "unknown")
        
        if table_type == "html":
            # 解析 HTML 表格
            parsed = self._parse_html_table(json_data["raw_html"])
            
            # 构建结构化 prompt
            prompt = f"""以下是一个 HTML 表格的结构化数据：

表头：{parsed['headers']}
行数：{len(parsed['rows'])}
列数：{len(parsed['headers'])}
数据样例（前3行）：
{self._format_rows(parsed['rows'][:3])}

上下文：
- 前文：{json_data.get('context', {}).get('preceding_text', '无')}
- 后文：{json_data.get('context', {}).get('following_text', '无')}

请用 2-3 句话总结这个表格的内容，突出关键数据和对比关系。
要求：使用完整的句子，便于向量检索。

摘要："""
        else:
            # Markdown 或其他格式
            prompt = f"""以下是一个表格：

{unit.content}

请用 2-3 句话总结这个表格的内容。

摘要："""
        
        response = await self._conv.asend(prompt)
        return {"table_summary": response.content.strip()}
    
    def _parse_html_table(self, html: str) -> dict:
        """Parse HTML table to structured data"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return {"headers": [], "rows": []}
        
        headers = []
        rows = []
        
        # 提取表头
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        else:
            # 如果没有 thead，尝试第一行
            first_row = table.find('tr')
            if first_row:
                headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
        
        # 提取数据行
        tbody = table.find('tbody') or table
        for tr in tbody.find_all('tr')[1 if not thead else 0:]:
            row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if row:
                rows.append(row)
        
        return {
            "headers": headers,
            "rows": rows
        }
    
    def _format_rows(self, rows: list) -> str:
        """Format rows for prompt"""
        return "\n".join([f"  {i+1}. {row}" for i, row in enumerate(rows)])
```

---

#### **改进方案 2: 使用 VLM 理解表格**

对于非常复杂的表格（如财报、科技论文中的对比表），可以：

```python
class VLMTableExtractor(BaseExtractor):
    """Use VLM to understand complex tables"""
    
    def __init__(self, llm_uri: str, api_key: str, use_vlm: bool = False):
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.use_vlm = use_vlm
        
        if use_vlm:
            # 初始化 VLM（如 GPT-4o, Qwen-VL）
            self._vlm = chak.Conversation(
                "bailian/qwen-vl-max",  # 支持视觉的模型
                api_key=api_key
            )
        else:
            self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit: TableUnit) -> Dict:
        if not isinstance(unit, TableUnit):
            return {}
        
        json_data = unit.json_data
        if not json_data or not self.use_vlm:
            # 使用常规方法
            return await self._extract_with_llm(unit)
        
        # Option A: Render HTML to image, use VLM
        if json_data.get('table_type') == 'html':
            # Convert HTML table to image
            table_image = self._render_html_to_image(unit.content)
            
            # Use VLM
            response = await self._vlm.asend(
                "这是一个表格图片，请总结其内容和关键信息（2-3句话）。",
                images=[table_image]
            )
            
            return {"table_summary": response.content.strip()}
        
        # Fallback to text-based
        return await self._extract_with_llm(unit)
    
    def _render_html_to_image(self, html: str) -> bytes:
        """Render HTML table to image using selenium or playwright"""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Wrap table in full HTML
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    table {{ border-collapse: collapse; }}
                    td, th {{ border: 1px solid black; padding: 8px; }}
                </style>
            </head>
            <body>{html}</body>
            </html>
            """
            
            page.set_content(full_html)
            screenshot = page.screenshot()
            browser.close()
            
            return screenshot
```

---

#### **改进方案 3: 分层总结（适合超大表格）**

对于几十行的复杂表格：

```python
class HierarchicalTableExtractor(BaseExtractor):
    """Summarize large tables hierarchically"""
    
    def __init__(self, llm_uri: str, api_key: str, chunk_size: int = 5):
        self.llm_uri = llm_uri
        self.api_key = api_key
        self.chunk_size = chunk_size
        self._conv = chak.Conversation(llm_uri, api_key=api_key)
    
    async def _extract_from_unit(self, unit: TableUnit) -> Dict:
        if not isinstance(unit, TableUnit):
            return {}
        
        json_data = unit.json_data
        if not json_data:
            return {}
        
        # 解析表格
        parsed = self._parse_html_table(json_data.get("raw_html", ""))
        rows = parsed["rows"]
        headers = parsed["headers"]
        
        # 如果表格不大，直接总结
        if len(rows) <= self.chunk_size:
            return await self._summarize_small_table(headers, rows)
        
        # 大表格：分层总结
        # 1. 每 N 行生成一个小摘要
        row_summaries = []
        for i in range(0, len(rows), self.chunk_size):
            chunk = rows[i:i + self.chunk_size]
            summary = await self._summarize_chunk(headers, chunk, i)
            row_summaries.append(summary)
        
        # 2. 汇总所有小摘要
        final_summary = await self._summarize_summaries(headers, row_summaries)
        
        return {"table_summary": final_summary}
    
    async def _summarize_chunk(self, headers: list, rows: list, start_idx: int) -> str:
        """Summarize a chunk of rows"""
        prompt = f"""表头：{headers}

数据行 {start_idx+1} 到 {start_idx+len(rows)}：
{self._format_rows(rows)}

用一句话总结这部分数据的特点。

摘要："""
        
        response = await self._conv.asend(prompt)
        return response.content.strip()
    
    async def _summarize_summaries(self, headers: list, summaries: list) -> str:
        """Summarize all chunk summaries"""
        prompt = f"""这是一个大型表格的分段摘要：

表头：{headers}

分段摘要：
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(summaries)])}

请用 2-3 句话总结整个表格的内容，突出关键数据和趋势。

最终摘要："""
        
        response = await self._conv.asend(prompt)
        return response.content.strip()
```

---

### 4️⃣ 连锁反应：复杂表格如何影响整个 Pipeline

| **阶段** | **潜在问题** | **解决方案** |
|---------|-------------|-------------|
| **Reader** | HTML 解析不准确、表格被识别为文本 | 使用 MinerU/Docling 高精度解析器；在 Reader 层面构建 TableUnit |
| **Splitter** | 表格被切断、与上下文分离、无法识别表格边界 | 使用 TableAwareSplitter；或依赖 Reader 已构建的 TableUnit |
| **Extractor** | LLM 无法理解复杂 HTML、缺少上下文、超长表格 | 使用 BeautifulSoup 解析 HTML；添加上下文信息；使用 VLM 或分层总结 |
| **Embedder** | 表格向量化效果差、语义信息不足 | 依赖高质量的 `table_summary`；考虑单独 embedding 表格和摘要 |
| **Retriever** | 检索不到表格内容、结构化查询失败 | 确保 `table_summary` 质量；支持 metadata 过滤（table_type, page, bbox） |
| **Indexer** | 表格和文本混合索引效果不佳 | 分别索引 TextUnit 和 TableUnit；支持按 unit_type 过滤 |

---

## 推荐的完整设计方案

### **Phase 1: Reader 层面改造（优先级最高）**

**目标**: 让 Reader 直接输出 TableUnit，避免后续重复解析

**涉及文件**:
- `zag/readers/mineru.py`
- `zag/readers/docling.py`
- `zag/schemas/pdf.py` (Page 的数据结构)

**改动点**:

1. **修改 Page.content 的数据结构**:
   ```python
   # 当前: content 是 dict
   Page(
       page_number=1,
       content={
           "texts": [...],
           "tables": [...],
           "images": [...]
       }
   )
   
   # 改为: content 是 list[BaseUnit]
   Page(
       page_number=1,
       content=[
           TextUnit(...),
           TableUnit(...),
           TextUnit(...),
           ImageUnit(...)
       ]
   )
   ```

2. **在 `_build_pages_from_content_list` 中构建 TableUnit**（见上文代码）

3. **TableUnit 包含丰富的 json_data**:
   ```python
   TableUnit(
       content="<table>...</table>",
       json_data={
           "table_type": "html",
           "raw_html": "...",
           "latex": "...",  # MinerU 提供
           "bbox": [x, y, w, h],
           "page_idx": 2
       }
   )
   ```

---

### **Phase 2: TableExtractor 增强（中优先级）**

**目标**: 让 TableExtractor 能够准确总结复杂 HTML 表格

**涉及文件**:
- `zag/extractors/table.py`

**改动点**:

1. 添加 `_parse_html_table` 方法（使用 BeautifulSoup）
2. 改进 Prompt，包含：
   - 结构化的表头和数据
   - 上下文信息（如果有）
   - 表格大小（行数、列数）
3. 可选：支持 VLM 模式
4. 可选：支持分层总结（超大表格）

---

### **Phase 3: TableAwareSplitter（低优先级，可选）**

**目标**: 处理纯 Markdown 文档（没有经过 Reader 的场景）

**使用场景**:
- 用户直接读取 `.md` 文件
- Markdown 中包含 HTML 表格
- 不使用 MinerU/Docling Reader

**涉及文件**:
- `zag/splitters/markdown/table_aware.py` (新建)

---

### **Phase 4: 测试和验证**

**测试用例**:

1. **简单 HTML 表格**（3x3）
   - 验证：能正确识别和总结
   
2. **复杂 HTML 表格**（多层表头、合并单元格）
   - 验证：能保留结构信息
   
3. **超大表格**（50+ 行）
   - 验证：分层总结不会超过 token 限制
   
4. **混合内容**（文本 + 表格 + 图片）
   - 验证：所有 Unit 的链表关系正确
   
5. **Markdown 标准表格 vs HTML 表格**
   - 验证：两种格式都能正确处理

---

## 关键设计决策

### ✅ **推荐做法**

1. **在 Reader 层面识别表格**：复用 MinerU/Docling 的高精度解析
2. **TableUnit 保留多种格式**：
   - `content`: 原始 HTML/Markdown
   - `json_data`: 结构化数据 (headers + rows)
   - `metadata.custom`: bbox, page_idx 等
3. **TableExtractor 先解析后总结**：使用 BeautifulSoup 解析 HTML，再交给 LLM
4. **对于超复杂表格**：考虑使用 VLM 或分层总结

### ❌ **避免的做法**

1. **不要在 Splitter 中重复解析表格**（Reader 已经做了）
2. **不要直接把 HTML 字符串喂给 LLM**（需要先结构化）
3. **不要忽略表格的上下文**（前后段落、标题等）
4. **不要把表格和大段文本混在一个 TextUnit 中**（影响检索）

---

## 实现路线图

| Phase | 任务 | 优先级 | 预计工作量 |
|-------|------|--------|-----------|
| 1 | 修改 Reader 输出 TableUnit | 🔴 高 | 2-3 小时 |
| 2 | 增强 TableExtractor（HTML 解析） | 🟡 中 | 1-2 小时 |
| 3 | 添加 VLM 支持（可选） | 🟢 低 | 2-3 小时 |
| 4 | 实现 TableAwareSplitter（可选） | 🟢 低 | 2-3 小时 |
| 5 | 编写测试用例 | 🟡 中 | 1 小时 |

---

## 相关文件

**当前涉及的文件**:
- `zag/readers/mineru.py` - MinerU Reader
- `zag/readers/docling.py` - Docling Reader
- `zag/splitters/markdown/header_based.py` - Markdown 标题切分器
- `zag/extractors/table.py` - 表格提取器
- `zag/schemas/unit.py` - Unit 定义
- `zag/schemas/pdf.py` - PDF 和 Page 定义

**需要新建的文件**（可选）:
- `zag/splitters/markdown/table_aware.py` - 表格感知切分器
- `zag/extractors/vlm_table.py` - VLM 表格提取器

---

## 总结

这个挑战的核心在于：**不同组件之间的职责划分和数据传递**。

- **Reader**: 负责高精度解析，识别表格并构建 TableUnit
- **Splitter**: 负责按语义切分文本，但不需要重复解析表格
- **Extractor**: 负责理解表格语义，生成高质量摘要

通过在 **Reader 层面构建 TableUnit**，可以最大程度避免重复解析，同时为后续的 Extractor、Embedder、Retriever 提供丰富的结构化信息。
