#!/bin/bash
# 测试并行化评估功能

echo "=========================================="
echo "并行化评估测试"
echo "=========================================="
echo ""

# 测试 1: 串行模式（5个用户）
echo "测试 1: 串行模式（5个用户）"
echo "----------------------------------------"
time python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods static_s0,clasp_online \
  --max-users 5 \
  --comparison-root output/comparison_serial_test \
  --scorer-device cpu \
  --skip-window-split \
  --no-parallel

echo ""
echo "测试 2: 并行模式（5个用户，2个进程）"
echo "----------------------------------------"
time python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods static_s0,clasp_online \
  --max-users 5 \
  --comparison-root output/comparison_parallel_test \
  --scorer-device cpu \
  --skip-window-split \
  --user-processes 2

echo ""
echo "=========================================="
echo "对比结果"
echo "=========================================="
echo ""

echo "串行模式输出:"
ls -lh output/comparison_serial_test/*/baseline_chain_community_0.jsonl

echo ""
echo "并行模式输出:"
ls -lh output/comparison_parallel_test/*/baseline_chain_community_0.jsonl

echo ""
echo "验证数据一致性（行数应该相同）:"
echo "串行 static_s0: $(wc -l < output/comparison_serial_test/static_s0/baseline_chain_community_0.jsonl) 行"
echo "并行 static_s0: $(wc -l < output/comparison_parallel_test/static_s0/baseline_chain_community_0.jsonl) 行"
echo "串行 clasp_online: $(wc -l < output/comparison_serial_test/clasp_online/baseline_chain_community_0.jsonl) 行"
echo "并行 clasp_online: $(wc -l < output/comparison_parallel_test/clasp_online/baseline_chain_community_0.jsonl) 行"
