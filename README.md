# memory-rl

**An implementation of the [RL Memorization Study](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/memorization-empirical-study.md) from Thinking Machines' [community projects](https://thinkingmachines.ai/blog/call-for-community-projects/).**

The question: how many bits can RL learn per episode? Their [LoRA post](https://thinkingmachines.ai/blog/lora/) argues that scalar rewards bottleneck learning to ~1 bit/episode—but is that [right](https://x.com/_arohan_/status/1973260831824683173)? [Or not](https://x.com/khoomeik/status/1979236183717875944)?

## The Test

From the project spec: create an environment with a latent random integer S ∈ [0, N) that the policy must memorize. Since S is uniform random, learning it requires log₂(N) bits. The hypothesis: with binary rewards, this takes O(N) episodes regardless of model capacity.

## What We Test

We vary the **reward channel bandwidth**:

| Reward Type | Signal | Capacity |
|-------------|--------|----------|
| **Binary** | 1 if guess = S, else 0 | 1 bit/ep |
| **Binned** | log-distance discretized into B buckets | log₂(B) bits/ep |
| **Continuous** | raw log₁₀(distance + 1) | unbounded |
| **Dense** | binary feedback per bit, k turns | log₂(N) bits/ep |

Binary is the bottleneck case. Dense is the control—it reveals all log₂(N) bits across k turns.

## Measuring Learning

The project spec asks: how do we track learning before full memorization?

We use teacher-forced logprobs to measure how much the model "knows" about S:

```
bits_known = log₂(N) + log P(S) / ln(2)
```

- Uniform over [0, N): P(S) = 1/N → bits_known = 0
- Memorized: P(S) = 1 → bits_known = log₂(N)

This gives a smooth learning curve instead of step-function accuracy.

## Quick Start

```bash
pip install tinker
export TINKER_API_KEY=sk-...

# Binary reward (1 bit/ep) — the bottleneck
python train.py env_type=single_step N=64 reward_type=binary

# Binned reward (3 bits/ep with B=8)
python train.py env_type=single_step N=64 reward_type=binned_log_distance reward_bins=8

# Dense reward (6 bits/ep) — the control
python train.py env_type=multi_step N=64

# Run full sweep (all conditions, 3 seeds each)
python run_sweep.py --sequential
```

## References

- [Thinking Machines - LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [Ord (2025) - The Inefficiency of RL](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning)
- [Li - Information Bandwidth of RL](https://richardli.xyz/post/information-bandwidth-rl/)
