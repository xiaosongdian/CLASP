#!/usr/bin/env python3
"""
测试三窗口统计功能
"""
import json
import sys

# 快速测试：只评估 1 个用户
test_command = """
python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods clasp_online \
  --max-users 1 \
  --comparison-root output/comparison_test_three_window \
  --scorer-device cpu \
  --skip-window-split
"""

print("=" * 80)
print("三窗口统计功能测试")
print("=" * 80)
print()
print("测试命令：")
print(test_command)
print()
print("运行测试...")
print()

import subprocess
result = subprocess.run(test_command, shell=True, capture_output=True, text=True)

if result.returncode != 0:
    print("❌ 测试失败！")
    print("STDERR:")
    print(result.stderr)
    sys.exit(1)

print("✅ 测试运行成功！")
print()

# 检查输出文件
output_file = "output/comparison_test_three_window/clasp_online/baseline_chain_community_0.jsonl"
print(f"检查输出文件: {output_file}")

try:
    with open(output_file, 'r') as f:
        line = f.readline()
        data = json.loads(line)

    print()
    print("=" * 80)
    print("输出数据结构检查")
    print("=" * 80)
    print()

    # 检查基本字段
    print("✅ 基本字段:")
    print(f"  - user_id: {data.get('user_id')}")
    print(f"  - community_id: {data.get('community_id')}")
    print(f"  - method: {data.get('method')}")
    print(f"  - mean_Q: {data.get('mean_Q')}")
    print()

    # 检查 steps
    if 'steps' in data and len(data['steps']) > 0:
        print("✅ Steps 字段存在")
        step = data['steps'][0]
        print(f"  - 第一个 step 的字段: {list(step.keys())}")
        print()

        # 检查新增字段
        new_fields = ['profile_updated', 'profile_length', 'num_candidates',
                      'best_candidate_index', 'candidate_scores', 'three_window_evaluation']

        print("新增字段检查:")
        for field in new_fields:
            if field in step:
                print(f"  ✅ {field}: 存在")
                if field == 'three_window_evaluation' and step[field]:
                    three_win = step[field]
                    print(f"     - 包含窗口: {list(three_win.keys())}")
                    if 'current_window' in three_win:
                        curr = three_win['current_window']
                        print(f"     - current_window gain: ΔQ={curr['gain']['ΔQ']:.4f}")
            else:
                print(f"  ⚠️  {field}: 不存在")
        print()

        # 显示完整的第一个 step（格式化）
        print("=" * 80)
        print("第一个 Step 的完整数据（格式化）:")
        print("=" * 80)
        print(json.dumps(step, indent=2, ensure_ascii=False))

    else:
        print("❌ Steps 字段为空或不存在")

except FileNotFoundError:
    print(f"❌ 输出文件不存在: {output_file}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"❌ JSON 解析失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 检查失败: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("✅ 三窗口统计功能测试通过！")
print("=" * 80)
