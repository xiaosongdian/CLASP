#!/usr/bin/env python3
"""
验证三窗口评估是否只在最后一步执行
"""

import json
import sys
from pathlib import Path

def check_three_window_evaluation(jsonl_file):
    """检查三窗口评估是否只在最后一步"""

    with open(jsonl_file, 'r') as f:
        results = [json.loads(line) for line in f]

    print(f"检查文件: {jsonl_file}")
    print(f"总用户数: {len(results)}")
    print()

    # 统计
    users_with_three_window = 0
    total_three_window_evals = 0
    users_with_multiple_evals = 0

    for user in results:
        steps = user['steps']
        three_window_steps = [i for i, s in enumerate(steps) if 'three_window_evaluation' in s]

        if three_window_steps:
            users_with_three_window += 1
            total_three_window_evals += len(three_window_steps)

            if len(three_window_steps) > 1:
                users_with_multiple_evals += 1
                print(f"❌ 用户 {user['user_id']} 有 {len(three_window_steps)} 个三窗口评估:")
                print(f"   步骤索引: {three_window_steps}")
                print(f"   总步骤数: {len(steps)}")
                print()

            # 检查是否在最后一步
            last_step_idx = len(steps) - 1
            if three_window_steps and three_window_steps[-1] != last_step_idx:
                print(f"⚠️  用户 {user['user_id']} 的三窗口评估不在最后一步:")
                print(f"   三窗口评估在步骤: {three_window_steps}")
                print(f"   最后一步索引: {last_step_idx}")
                print()

    print("=" * 60)
    print("统计结果:")
    print(f"  有三窗口评估的用户: {users_with_three_window} / {len(results)}")
    print(f"  总三窗口评估次数: {total_three_window_evals}")
    print(f"  平均每用户: {total_three_window_evals / len(results):.2f}")
    print(f"  有多次评估的用户: {users_with_multiple_evals}")
    print()

    if users_with_multiple_evals == 0:
        print("✅ 所有用户都只在最后一步有三窗口评估")
    else:
        print(f"❌ 有 {users_with_multiple_evals} 个用户有多次三窗口评估")

    return users_with_multiple_evals == 0


def check_static_s0_history(jsonl_file):
    """检查 static_s0 的历史动作是否变化"""

    with open(jsonl_file, 'r') as f:
        results = [json.loads(line) for line in f]

    print("\n" + "=" * 60)
    print("检查 static_s0 的历史动作变化")
    print("=" * 60)

    # 取第一个用户
    user = results[0]
    steps = user['steps']

    # 找到有三窗口评估的步骤
    three_window_step = None
    for step in steps:
        if 'three_window_evaluation' in step:
            three_window_step = step
            break

    if not three_window_step:
        print("❌ 没有找到三窗口评估")
        return False

    three_win = three_window_step['three_window_evaluation']

    print(f"用户 {user['user_id']} 的三窗口评估:")
    print()

    # 检查过去窗口
    if 'past_window' in three_win:
        past = three_win['past_window']
        print(f"过去窗口:")
        print(f"  history: {past['history']}")
        print(f"  target: {past['target']}")
        print(f"  old_profile F: {past['with_old_profile']['F']:.4f}")
        print(f"  new_profile F: {past['with_new_profile']['F']:.4f}")
        print(f"  gain ΔF: {past['gain']['ΔF']:.4f}")
        print()

    # 检查当前窗口
    if 'current_window' in three_win:
        current = three_win['current_window']
        print(f"当前窗口:")
        print(f"  history: {current['history']}")
        print(f"  target: {current['target']}")
        print(f"  old_profile F: {current['with_old_profile']['F']:.4f}")
        print(f"  new_profile F: {current['with_new_profile']['F']:.4f}")
        print(f"  gain ΔF: {current['gain']['ΔF']:.4f}")
        print()

    # 检查未来窗口
    if 'future_window' in three_win:
        future = three_win['future_window']
        print(f"未来窗口:")
        print(f"  history: {future['history']}")
        print(f"  target: {future['target']}")
        print(f"  old_profile F: {future['with_old_profile']['F']:.4f}")
        print(f"  new_profile F: {future['with_new_profile']['F']:.4f}")
        print(f"  gain ΔF: {future['gain']['ΔF']:.4f}")
        print()

    # 检查历史是否变化
    histories = []
    if 'past_window' in three_win:
        histories.append(three_win['past_window']['history'])
    if 'current_window' in three_win:
        histories.append(three_win['current_window']['history'])
    if 'future_window' in three_win:
        histories.append(three_win['future_window']['history'])

    unique_histories = set(histories)
    print(f"历史窗口: {histories}")
    print(f"唯一历史窗口数: {len(unique_histories)}")

    if len(unique_histories) == len(histories):
        print("✅ 历史动作正确变化（每个窗口使用不同的历史）")
        return True
    else:
        print("❌ 历史动作没有变化（多个窗口使用相同的历史）")
        return False


if __name__ == "__main__":
    # 检查 static_s0
    static_file = Path("output/comparison/static_s0/baseline_chain_test.jsonl")
    if static_file.exists():
        ok1 = check_three_window_evaluation(static_file)
        ok2 = check_static_s0_history(static_file)
    else:
        print(f"文件不存在: {static_file}")
        ok1 = ok2 = False

    # 检查 clasp_online
    clasp_file = Path("output/comparison/clasp_online/baseline_chain_test.jsonl")
    if clasp_file.exists():
        print("\n" + "=" * 60)
        ok3 = check_three_window_evaluation(clasp_file)
    else:
        print(f"文件不存在: {clasp_file}")
        ok3 = False

    # 总结
    print("\n" + "=" * 60)
    print("总结:")
    if ok1 and ok2 and ok3:
        print("✅ 所有检查通过")
        sys.exit(0)
    else:
        print("❌ 部分检查失败")
        sys.exit(1)
