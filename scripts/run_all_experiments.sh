#!/bin/bash
# =============================================================================
# HIVE Paper Experiments — Run All Three
# =============================================================================
#
# Prerequisites:
#   - .env file with OPENAI_API_KEY in the project root
#   - pip install -r requirements.txt
#
# Estimated cost: ~$1-3 in OpenAI API calls (gpt-4o-mini)
# Estimated time: 15-30 minutes total
#
# Usage:
#   cd honey-prompt-detector
#   bash scripts/run_all_experiments.sh
#
# =============================================================================

set -e

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"
echo ""

# Check .env
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Create one with OPENAI_API_KEY=sk-..."
    exit 1
fi

# Create results dir
mkdir -p results

echo "============================================================"
echo "EXPERIMENT 1/3: Stage 3 End-to-End Agent Evaluation"
echo "  Embeds honey tokens in system prompts, measures detection"
echo "  10 tokens × 40 queries = 400 API calls"
echo "============================================================"
python3 scripts/experiment_stage3_e2e_agent.py --n-tokens 10 --model gpt-4o-mini
echo ""

echo "============================================================"
echo "EXPERIMENT 2/3: Judge-Path Adversarial Evaluation"
echo "  Tests Stage 4 judge against evasion payloads"
echo "  55 total API calls"
echo "============================================================"
python3 scripts/experiment_judge_adversarial.py --model gpt-4o-mini
echo ""

echo "============================================================"
echo "EXPERIMENT 3/3: Multi-Turn Prompt Injection Evaluation"
echo "  Tests detection across multi-turn conversation scenarios"
echo "  ~80 API calls (10 scenarios × ~4 turns × 2 modes)"
echo "============================================================"
python3 scripts/experiment_multiturn.py --model gpt-4o-mini
echo ""

echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  results/experiment_stage3_e2e.json"
echo "  results/experiment_stage3_e2e_summary.txt"
echo "  results/experiment_judge_adversarial.json"
echo "  results/experiment_judge_adversarial_summary.txt"
echo "  results/experiment_multiturn.json"
echo "  results/experiment_multiturn_summary.txt"
echo ""
echo "Next step: share the *_summary.txt files and I'll integrate"
echo "the results into the paper."
