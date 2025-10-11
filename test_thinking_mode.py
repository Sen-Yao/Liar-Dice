#!/usr/bin/env python3
"""
测试思考模式开关功能

验证：
1. enable_thinking=False时，LLMAgent和OptimizedLLMAgent不启用思考模式
2. enable_thinking=True时，两者都启用思考模式
3. llm_tournament.py的--enable-thinking参数正确传递
"""

import sys
from agents.llm_agent import LLMAgent
from agents.baseline_agents import OptimizedLLMAgent


def test_llm_agent_thinking_mode():
    """测试LLMAgent的思考模式开关"""
    print("=" * 60)
    print("测试 1: LLMAgent 思考模式开关")
    print("=" * 60)

    # 测试默认值（应该是False）
    agent_default = LLMAgent("test_agent", 2)
    assert agent_default.enable_thinking == False, "默认应该禁用思考模式"
    print("✓ LLMAgent 默认 enable_thinking=False")

    # 测试显式关闭
    agent_off = LLMAgent("test_agent", 2, enable_thinking=False)
    assert agent_off.enable_thinking == False, "显式禁用失败"
    print("✓ LLMAgent enable_thinking=False 工作正常")

    # 测试显式开启
    agent_on = LLMAgent("test_agent", 2, enable_thinking=True)
    assert agent_on.enable_thinking == True, "显式启用失败"
    print("✓ LLMAgent enable_thinking=True 工作正常")

    print()


def test_optimized_llm_agent_thinking_mode():
    """测试OptimizedLLMAgent的思考模式开关"""
    print("=" * 60)
    print("测试 2: OptimizedLLMAgent 思考模式开关")
    print("=" * 60)

    # 测试默认值（应该是False）
    agent_default = OptimizedLLMAgent("test_agent", 2)
    assert agent_default.enable_thinking == False, "默认应该禁用思考模式"
    print("✓ OptimizedLLMAgent 默认 enable_thinking=False")

    # 测试显式关闭
    agent_off = OptimizedLLMAgent("test_agent", 2, enable_thinking=False)
    assert agent_off.enable_thinking == False, "显式禁用失败"
    print("✓ OptimizedLLMAgent enable_thinking=False 工作正常")

    # 测试显式开启
    agent_on = OptimizedLLMAgent("test_agent", 2, enable_thinking=True)
    assert agent_on.enable_thinking == True, "显式启用失败"
    print("✓ OptimizedLLMAgent enable_thinking=True 工作正常")

    print()


def test_tournament_integration():
    """测试llm_tournament.py的集成"""
    print("=" * 60)
    print("测试 3: llm_tournament.py 集成")
    print("=" * 60)

    from llm_tournament import create_agent

    # 测试非LLM代理（不应该受影响）
    random_agent = create_agent("random", "test_agent", 2, enable_thinking=True)
    print("✓ 非LLM代理创建正常（思考模式参数不影响）")

    # 测试LLM代理，默认关闭
    llm_agent_off = create_agent("llm", "test_agent", 2, enable_thinking=False)
    assert llm_agent_off.enable_thinking == False, "LLM代理思考模式应该关闭"
    print("✓ create_agent创建LLM代理，enable_thinking=False 工作正常")

    # 测试LLM代理，显式开启
    llm_agent_on = create_agent("llm", "test_agent", 2, enable_thinking=True)
    assert llm_agent_on.enable_thinking == True, "LLM代理思考模式应该开启"
    print("✓ create_agent创建LLM代理，enable_thinking=True 工作正常")

    print()


def test_stats_tracking():
    """测试思考模式统计功能"""
    print("=" * 60)
    print("测试 4: 思考模式统计追踪")
    print("=" * 60)

    # 创建启用思考模式的代理
    agent = OptimizedLLMAgent("test_agent", 2, enable_thinking=True)

    # 检查统计字段是否存在
    stats = agent.get_stats()
    assert "thinking_mode_calls" in stats, "统计中缺少thinking_mode_calls"
    assert "thinking_content_length" in stats, "统计中缺少thinking_content_length"
    assert "avg_thinking_length" in stats, "统计中缺少avg_thinking_length"

    # 初始值应该为0
    assert stats["thinking_mode_calls"] == 0, "初始thinking_mode_calls应该为0"
    assert stats["thinking_content_length"] == 0, "初始thinking_content_length应该为0"
    assert stats["avg_thinking_length"] == 0.0, "初始avg_thinking_length应该为0.0"

    print("✓ 思考模式统计字段存在且初始化正确")
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试思考模式开关功能")
    print("=" * 60 + "\n")

    try:
        test_llm_agent_thinking_mode()
        test_optimized_llm_agent_thinking_mode()
        test_tournament_integration()
        test_stats_tracking()

        print("=" * 60)
        print("所有测试通过 ✓")
        print("=" * 60)
        print("\n使用方法：")
        print("  python llm_tournament.py --enable-thinking --opponents random --num-games 5")
        print()
        return 0

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
