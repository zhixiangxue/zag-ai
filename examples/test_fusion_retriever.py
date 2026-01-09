#!/usr/bin/env python3
"""
测试 QueryFusionRetriever 示例 - 演示融合检索功能

演示功能：
1. 使用单个 FullTextRetriever 测试基础功能
2. 使用多个 FullTextRetriever 测试融合模式
3. 对比不同融合策略的效果和性能

前置条件：
- Meilisearch 服务需要运行在 http://127.0.0.1:7700
- 运行方式：
  - 下载：https://github.com/meilisearch/meilisearch/releases
  - 启动：./meilisearch
"""

import sys
from pathlib import Path
import time

from zag.indexers import FullTextIndexer
from zag.retrievers import FullTextRetriever, QueryFusionRetriever, FusionMode
from zag.schemas.base import BaseUnit, UnitMetadata


def check_service():
    """检查 Meilisearch 服务是否运行"""
    import meilisearch
    try:
        client = meilisearch.Client("http://127.0.0.1:7700")
        health = client.health()
        return health.get("status") == "available"
    except Exception as e:
        print(f"❌ 无法连接 Meilisearch 服务: {e}")
        print("请确保 Meilisearch 服务正在运行 (http://127.0.0.1:7700)")
        return False


def create_sample_units():
    """创建示例文档单元（房地产主题）"""
    units = [
        BaseUnit(
            unit_id="house_1",
            content="Beautiful 3-bedroom house in downtown San Francisco with modern kitchen and spacious backyard.",
            metadata=UnitMetadata(
                custom={
                    "title": "Modern Downtown House",
                    "city": "San Francisco",
                    "state": "California",
                    "bedrooms": 3,
                    "price": 1200000,
                    "type": "house",
                }
            )
        ),
        BaseUnit(
            unit_id="apt_1",
            content="Luxury apartment with 2 bedrooms in New York City. Great view of Central Park.",
            metadata=UnitMetadata(
                custom={
                    "title": "Central Park Apartment",
                    "city": "New York",
                    "state": "New York",
                    "bedrooms": 2,
                    "price": 850000,
                    "type": "apartment",
                }
            )
        ),
        BaseUnit(
            unit_id="house_2",
            content="Spacious 4-bedroom family home in Austin with large garage and swimming pool.",
            metadata=UnitMetadata(
                custom={
                    "title": "Family Home with Pool",
                    "city": "Austin",
                    "state": "Texas",
                    "bedrooms": 4,
                    "price": 650000,
                    "type": "house",
                }
            )
        ),
        BaseUnit(
            unit_id="condo_1",
            content="Modern condo in Seattle downtown, 1 bedroom with gym and parking included.",
            metadata=UnitMetadata(
                custom={
                    "title": "Downtown Seattle Condo",
                    "city": "Seattle",
                    "state": "Washington",
                    "bedrooms": 1,
                    "price": 450000,
                    "type": "condo",
                }
            )
        ),
        BaseUnit(
            unit_id="house_3",
            content="Cozy 2-bedroom house in Los Angeles suburban area with nice garden.",
            metadata=UnitMetadata(
                custom={
                    "title": "Suburban LA House",
                    "city": "Los Angeles",
                    "state": "California",
                    "bedrooms": 2,
                    "price": 780000,
                    "type": "house",
                }
            )
        ),
    ]
    return units


def print_results(title, units, elapsed_time=None):
    """格式化打印检索结果"""
    print(f"\n{'='*70}")
    print(f"{title}")
    if elapsed_time is not None:
        print(f"耗时: {elapsed_time*1000:.2f}ms")
    print(f"{'='*70}")
    print(f"找到 {len(units)} 条结果:\n")
    
    for i, unit in enumerate(units, 1):
        score_str = f" (分数: {unit.score:.4f})" if hasattr(unit, 'score') and unit.score is not None else ""
        print(f"{i}. {unit.metadata.custom.get('title', 'N/A')}{score_str}")
        print(f"   位置: {unit.metadata.custom.get('city')}, {unit.metadata.custom.get('state')}")
        print(f"   价格: ${unit.metadata.custom.get('price'):,}")
        print()


def main():
    print("=" * 70)
    print("QueryFusionRetriever 测试示例")
    print("=" * 70)
    print()
    
    # 1. 检查服务
    print("1️⃣  检查 Meilisearch 服务...")
    if not check_service():
        sys.exit(1)
    print("   ✓ 服务正常运行")
    print()
    
    # 2. 创建索引器并构建索引
    print("2️⃣  构建测试索引...")
    indexer = FullTextIndexer(
        url="http://127.0.0.1:7700",
        index_name="fusion_test",
        primary_key="unit_id"
    )
    
    indexer.clear()
    indexer.configure_settings(
        searchable_attributes=["content", "title"],
        filterable_attributes=["city", "state", "bedrooms", "price", "type"],
    )
    
    units = create_sample_units()
    indexer.add(units)
    print(f"   ✓ 已添加 {len(units)} 个文档")
    print()
    
    # === 测试1: 单个 Retriever ===
    print("=" * 70)
    print("🔍 测试 1: 单个 FullTextRetriever（基准测试）")
    print("=" * 70)
    print()
    
    retriever = FullTextRetriever(
        url="http://127.0.0.1:7700",
        index_name="fusion_test"
    )
    
    start = time.time()
    results = retriever.retrieve("modern house", top_k=3)
    elapsed = time.time() - start
    print_results("直接检索: 'modern house'", results, elapsed)
    
    # === 测试2: 使用 FusionRetriever 包装单个 Retriever ===
    print("=" * 70)
    print("🔍 测试 2: FusionRetriever 包装单个 Retriever")
    print("=" * 70)
    print()
    
    fusion_single = QueryFusionRetriever(
        retrievers=[retriever],
        mode=FusionMode.SIMPLE,
        top_k=3
    )
    
    start = time.time()
    results = fusion_single.retrieve("modern house", top_k=3)
    elapsed = time.time() - start
    print_results("Fusion检索(单个): 'modern house'", results, elapsed)
    
    # === 测试3: 多个 Retriever + SIMPLE 模式 ===
    print("=" * 70)
    print("🔍 测试 3: 多个 Retriever - SIMPLE 融合模式")
    print("=" * 70)
    print()
    
    # 创建两个相同的 retriever 来模拟多源检索
    retriever1 = FullTextRetriever(
        url="http://127.0.0.1:7700",
        index_name="fusion_test"
    )
    retriever2 = FullTextRetriever(
        url="http://127.0.0.1:7700",
        index_name="fusion_test"
    )
    
    fusion_simple = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        mode=FusionMode.SIMPLE,
        top_k=3
    )
    
    start = time.time()
    results = fusion_simple.retrieve("apartment luxury", top_k=3)
    elapsed = time.time() - start
    print_results("SIMPLE融合: 'apartment luxury'", results, elapsed)
    
    # === 测试4: RRF 模式 ===
    print("=" * 70)
    print("🔍 测试 4: 多个 Retriever - RRF 融合模式")
    print("=" * 70)
    print()
    
    fusion_rrf = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        mode=FusionMode.RECIPROCAL_RANK,
        top_k=3
    )
    
    start = time.time()
    results = fusion_rrf.retrieve("apartment luxury", top_k=3)
    elapsed = time.time() - start
    print_results("RRF融合: 'apartment luxury'", results, elapsed)
    
    # === 测试5: RELATIVE_SCORE 模式 ===
    print("=" * 70)
    print("🔍 测试 5: 多个 Retriever - RELATIVE_SCORE 融合模式")
    print("=" * 70)
    print()
    
    fusion_relative = QueryFusionRetriever(
        retrievers=[retriever1, retriever2],
        mode=FusionMode.RELATIVE_SCORE,
        top_k=3,
        retriever_weights=[0.6, 0.4]  # 第一个权重更高
    )
    
    start = time.time()
    results = fusion_relative.retrieve("apartment luxury", top_k=3)
    elapsed = time.time() - start
    print_results("RELATIVE_SCORE融合: 'apartment luxury'", results, elapsed)
    
    # === 测试6: 带过滤条件的融合检索 ===
    print("=" * 70)
    print("🔍 测试 6: 带过滤条件的融合检索")
    print("=" * 70)
    print()
    
    start = time.time()
    results = fusion_rrf.retrieve(
        "house",
        top_k=3,
        filters={"state": "California"}
    )
    elapsed = time.time() - start
    print_results("RRF融合 + 过滤(California): 'house'", results, elapsed)
    
    # 最终统计
    print("=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print("✅ 所有测试完成!")
    print()
    print("💡 关键发现:")
    print("   - FusionRetriever 支持单个或多个 Retriever")
    print("   - 多个 Retriever 会并发执行（使用 ThreadPoolExecutor）")
    print("   - 三种融合模式:")
    print("     • SIMPLE: 去重并保留最高分")
    print("     • RECIPROCAL_RANK: RRF 算法，适合不同类型检索器")
    print("     • RELATIVE_SCORE: 相对分数融合，适合同类型检索器")


if __name__ == "__main__":
    main()
