**核心要点**:

1. ✅ 所有chunk都要预处理,提取信息存在`node.properties`
2. ✅ Persona通过LLM根据chunk的summary自动生成
3. ✅ 单跳组合: chunk + theme + persona + style + length (5个维度)
4. ✅ 多跳组合: chunk对 + overlapped*themes + persona + style + length (5个维度)*
5. ✅ 最后异步并发调用LLM生成问题和答案

这就是**完整的精确答案**!每个步骤我都给出了代码位置和具体数据结构。

```
# 项目结构
zeval/
├── synthetic_data/              # 合成数据生成模块
│   ├── storage/                 # 数据存储层
│   │   └── knowledge_graph.py
│   ├── transforms/              # 数据增强层
│   │   ├── extractors/
│   │   ├── relationship_builders/
│   │   └── engine.py
│   ├── generators/              # 生成器层
│   │   ├── single_hop.py
│   │   └── multi_hop.py
│   └── schemas/                 # 输出格式
│       └── dataset.py
│
├── evaluation/                  # 真实评估模块
│   ├── metrics/
│   └── runners/
│
└── main.py
```





