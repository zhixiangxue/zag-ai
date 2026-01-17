#!/usr/bin/env python3
"""
测试 FullTextRetriever 示例 - 完整演示索引和检索流程

演示功能：
1. 使用 FullTextIndexer 构建索引
2. 使用 FullTextRetriever 进行各种搜索
   - 简单搜索
   - 过滤搜索
   - 排序搜索
   - 分面搜索
   - 复杂查询
   - 拼写容错

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
from zag.retrievers import FullTextRetriever
from zag.schemas import BaseUnit, UnitMetadata


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
                    "timestamp": 1704067200
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
                    "timestamp": 1704153600
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
                    "timestamp": 1704240000
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
                    "timestamp": 1704326400
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
                    "timestamp": 1704412800
                }
            )
        ),
        BaseUnit(
            unit_id="apt_2",
            content="Brand new 3-bedroom apartment in Boston with modern amenities and rooftop access.",
            metadata=UnitMetadata(
                custom={
                    "title": "New Boston Apartment",
                    "city": "Boston",
                    "state": "Massachusetts",
                    "bedrooms": 3,
                    "price": 720000,
                    "type": "apartment",
                    "timestamp": 1704499200
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
        score_str = f" (相关度: {unit.score:.4f})" if hasattr(unit, 'score') and unit.score is not None else ""
        print(f"{i}. {unit.metadata.custom.get('title', 'N/A')}{score_str}")
        print(f"   位置: {unit.metadata.custom.get('city')}, {unit.metadata.custom.get('state')}")
        print(f"   价格: ${unit.metadata.custom.get('price'):,}")
        print(f"   卧室: {unit.metadata.custom.get('bedrooms')} 间")
        print(f"   类型: {unit.metadata.custom.get('type')}")
        print(f"   简介: {unit.content[:80]}...")
        print()


def main():
    print("=" * 70)
    print("FullTextRetriever 完整测试示例")
    print("=" * 70)
    print()
    
    # 1. 检查服务
    print("1️⃣  检查 Meilisearch 服务...")
    if not check_service():
        sys.exit(1)
    print("   ✓ 服务正常运行")
    print()
    
    # 2. 创建索引器并构建索引
    print("2️⃣  使用 FullTextIndexer 构建索引...")
    indexer = FullTextIndexer(
        url="http://127.0.0.1:7700",
        index_name="real_estate",
        primary_key="unit_id"
    )
    
    # 清空已有数据
    indexer.clear()
    
    # 配置搜索设置
    indexer.configure_settings(
        searchable_attributes=["content", "title"],
        filterable_attributes=["city", "state", "bedrooms", "price", "type"],
        sortable_attributes=["price", "bedrooms", "timestamp"],
        displayed_attributes=["unit_id", "title", "content", "city", "state", "price", "bedrooms", "type"]
    )
    
    # 添加文档
    units = create_sample_units()
    indexer.add(units)
    print(f"   ✓ 已添加 {len(units)} 个房源到索引")
    print(f"   ✓ 当前文档数: {indexer.count()}")
    print()
    
    # 3. 创建检索器
    print("3️⃣  创建 FullTextRetriever...")
    retriever = FullTextRetriever(
        url="http://127.0.0.1:7700",
        index_name="real_estate",
        top_k=10
    )
    print(f"   ✓ {retriever}")
    print()
    
    # === 开始搜索演示 ===
    print("=" * 70)
    print("🔍 开始搜索演示")
    print("=" * 70)
    print()
    
    # 示例1: 简单搜索
    print("📝 示例 1: 简单搜索")
    start = time.time()
    results = retriever.retrieve("modern apartment", top_k=3)
    elapsed = time.time() - start
    print_results("搜索: 'modern apartment'", results, elapsed)
    
    # 示例2: 带过滤条件的搜索
    print("📝 示例 2: 过滤搜索 - 加州的房源")
    start = time.time()
    results = retriever.retrieve(
        "house",
        top_k=5,
        filters={"state": "California"}
    )
    elapsed = time.time() - start
    print_results("搜索: 'house' + 过滤条件: state='California'", results, elapsed)
    
    # 示例3: 价格范围过滤
    print("📝 示例 3: 价格范围 - 60万到80万美元")
    # 注意：Meilisearch 的 filters dict 不支持范围，需要使用更高级的API
    # 这里我们可以直接传递filter字符串
    start = time.time()
    results = retriever.retrieve(
        "",  # 空查询返回所有
        top_k=5,
        filter="price >= 600000 AND price <= 800000"
    )
    elapsed = time.time() - start
    print_results("价格范围: $600,000 - $800,000", results, elapsed)
    
    # 示例4: 排序搜索
    print("📝 示例 4: 排序 - 按价格升序")
    start = time.time()
    results = retriever.retrieve(
        "bedroom",
        top_k=5,
        sort=["price:asc"]
    )
    elapsed = time.time() - start
    print_results("搜索: 'bedroom' + 排序: 价格升序", results, elapsed)
    
    # 示例5: 复杂查询
    print("📝 示例 5: 复杂查询 - 3卧室且价格低于90万")
    start = time.time()
    results = retriever.retrieve(
        "spacious",
        top_k=5,
        filter="bedrooms = 3 AND price < 900000"
    )
    elapsed = time.time() - start
    print_results("搜索: 'spacious' + 3卧室 + 价格<$900,000", results, elapsed)
    
    # 示例6: 拼写容错
    print("📝 示例 6: 拼写容错 - 搜索 'apartmnt' (故意拼错)")
    start = time.time()
    results = retriever.retrieve("apartmnt", top_k=3)
    elapsed = time.time() - start
    print_results("搜索: 'apartmnt' (拼写错误，但仍能找到 apartment)", results, elapsed)
    
    # 示例7: 多条件OR过滤
    print("📝 示例 7: 多城市搜索 - New York 或 Boston")
    start = time.time()
    results = retriever.retrieve(
        "luxury",
        top_k=5,
        filter="city = 'New York' OR city = 'Boston'"
    )
    elapsed = time.time() - start
    print_results("搜索: 'luxury' + 城市: New York OR Boston", results, elapsed)
    
    # 示例8: 返回特定字段
    print("📝 示例 8: 只返回标题和价格字段")
    start = time.time()
    results = retriever.retrieve(
        "house",
        top_k=3,
        attributesToRetrieve=["unit_id", "content", "title", "price", "city"]  # 必须包含 unit_id 和 content
    )
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print("搜索: 'house' (仅返回 title, price, city)")
    print(f"耗时: {elapsed*1000:.2f}ms")
    print(f"{'='*70}\n")
    for i, unit in enumerate(results, 1):
        print(f"{i}. {unit.metadata.custom}")
    print()
    
    # 最终统计
    print("=" * 70)
    print("📊 统计信息")
    print("=" * 70)
    print(f"索引名称: {indexer.index_name}")
    print(f"总文档数: {indexer.count()}")
    print(f"服务地址: {indexer.url}")
    print()
    
    print("=" * 70)
    print("✅ 测试完成!")
    print("=" * 70)
    print()
    print("💡 提示:")
    print("   - Indexer 负责索引管理 (add, update, delete, clear)")
    print("   - Retriever 负责搜索检索 (retrieve with filters, sort, etc.)")
    print("   - 访问 http://127.0.0.1:7700 查看 Meilisearch 仪表板")


if __name__ == "__main__":
    main()
