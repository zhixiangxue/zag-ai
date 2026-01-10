#!/usr/bin/env python3
"""
测试 FullTextIndexer 示例

功能演示：
1. 创建全文索引
2. 配置搜索设置
3. 添加文档
4. 更新文档
5. 删除文档
6. 统计信息

前置条件：
- Meilisearch 服务需要运行在 http://127.0.0.1:7700
- 运行方式：
  - 下载：https://github.com/meilisearch/meilisearch/releases
  - 启动：./meilisearch
"""

import sys
from pathlib import Path

from zag.indexers import FullTextIndexer
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
    """创建示例文档单元"""
    units = [
        BaseUnit(
            unit_id="doc_1",
            content="Python is a high-level programming language known for its simplicity.",
            metadata=UnitMetadata(
                custom={
                    "title": "Introduction to Python",
                    "category": "programming",
                    "difficulty": "beginner",
                    "timestamp": 1640000000
                }
            )
        ),
        BaseUnit(
            unit_id="doc_2",
            content="Machine learning is a subset of artificial intelligence.",
            metadata=UnitMetadata(
                custom={
                    "title": "Machine Learning Basics",
                    "category": "ai",
                    "difficulty": "intermediate",
                    "timestamp": 1640100000
                }
            )
        ),
        BaseUnit(
            unit_id="doc_3",
            content="Docker containers provide isolated environments for applications.",
            metadata=UnitMetadata(
                custom={
                    "title": "Docker Guide",
                    "category": "devops",
                    "difficulty": "intermediate",
                    "timestamp": 1640200000
                }
            )
        ),
        BaseUnit(
            unit_id="doc_4",
            content="RESTful APIs are commonly used for web service communication.",
            metadata=UnitMetadata(
                custom={
                    "title": "REST API Design",
                    "category": "web",
                    "difficulty": "beginner",
                    "timestamp": 1640300000
                }
            )
        ),
    ]
    return units


def main():
    print("=" * 60)
    print("FullTextIndexer 测试示例")
    print("=" * 60)
    print()
    
    # 1. 检查服务
    print("1️⃣  检查 Meilisearch 服务...")
    if not check_service():
        sys.exit(1)
    print("   ✓ 服务正常运行")
    print()
    
    # 2. 创建索引器
    print("2️⃣  创建 FullTextIndexer...")
    indexer = FullTextIndexer(
        url="http://127.0.0.1:7700",
        index_name="test_documents",
        primary_key="unit_id"
    )
    print(f"   ✓ {indexer}")
    print()
    
    # 3. 清空已有数据
    print("3️⃣  清空已有数据...")
    indexer.clear()
    print(f"   ✓ 索引已清空，当前文档数: {indexer.count()}")
    print()
    
    # 4. 配置搜索设置
    print("4️⃣  配置搜索设置...")
    indexer.configure_settings(
        searchable_attributes=["content", "title"],
        filterable_attributes=["category", "difficulty", "timestamp"],
        sortable_attributes=["timestamp"],
        displayed_attributes=["unit_id", "title", "content", "category", "difficulty"]
    )
    print("   ✓ 搜索设置已配置")
    print()
    
    # 5. 添加文档
    print("5️⃣  添加示例文档...")
    units = create_sample_units()
    indexer.add(units)
    print(f"   ✓ 已添加 {len(units)} 个文档")
    print(f"   ✓ 当前文档数: {indexer.count()}")
    print()
    
    # 6. 验证文档存在
    print("6️⃣  验证文档存在...")
    for unit_id in ["doc_1", "doc_2", "doc_999"]:
        exists = indexer.exists(unit_id)
        status = "✓" if exists else "✗"
        print(f"   {status} {unit_id}: {'存在' if exists else '不存在'}")
    print()
    
    # 7. 更新文档
    print("7️⃣  更新文档...")
    updated_unit = BaseUnit(
        unit_id="doc_2",
        content="Machine learning enables computers to learn from data and improve automatically.",
        metadata=UnitMetadata(
            custom={
                "title": "Machine Learning Basics (Updated)",
                "category": "ai",
                "difficulty": "advanced",
                "timestamp": 1640150000
            }
        )
    )
    indexer.update(updated_unit)
    print("   ✓ doc_2 已更新")
    print()
    
    # 8. Upsert 测试（新增 + 更新）
    print("8️⃣  测试 Upsert...")
    upsert_units = [
        BaseUnit(
            unit_id="doc_3",  # 已存在，会更新
            content="Docker and Kubernetes are essential DevOps tools.",
            metadata=UnitMetadata(custom={"title": "Docker & K8s", "category": "devops", "difficulty": "advanced", "timestamp": 1640250000})
        ),
        BaseUnit(
            unit_id="doc_5",  # 新文档，会插入
            content="GraphQL is a query language for APIs.",
            metadata=UnitMetadata(custom={"title": "GraphQL Intro", "category": "web", "difficulty": "intermediate", "timestamp": 1640400000})
        ),
    ]
    indexer.upsert(upsert_units)
    print(f"   ✓ Upsert 完成")
    print(f"   ✓ 当前文档数: {indexer.count()}")
    print()
    
    # 9. 删除单个文档
    print("9️⃣  删除文档...")
    indexer.delete("doc_1")
    print("   ✓ doc_1 已删除")
    print(f"   ✓ 当前文档数: {indexer.count()}")
    print()
    
    # 10. 批量删除
    print("🔟 批量删除文档...")
    indexer.delete(["doc_4", "doc_5"])
    print("   ✓ doc_4, doc_5 已删除")
    print(f"   ✓ 当前文档数: {indexer.count()}")
    print()
    
    # 11. 最终统计
    print("📊 最终统计信息:")
    print(f"   • 索引名称: {indexer.index_name}")
    print(f"   • 文档总数: {indexer.count()}")
    print(f"   • 服务地址: {indexer.url}")
    print()
    
    print("=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)
    print()
    print("💡 提示:")
    print("   - 可以访问 http://127.0.0.1:7700 查看 Meilisearch 仪表板")
    print("   - 索引名称: test_documents")
    print("   - 剩余文档: doc_2, doc_3")


if __name__ == "__main__":
    main()
