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

Specifically for binary rewards we give 0/1 reward if it guesses the N-bit secret correctly or not. For binned rewards we discretize the log-distance into B buckets e.g. B=8 → {0, 0.14, ..., 1.0}. For continuous rewards we give the raw log₁₀(distance + 1) reward. For dense rewards the environment is a bitstring e.g. "0101010101" and we give a binary reward per bit per turn.

## Measuring Learning

We use teacher-forced evaluation: feed the model the correct answer and measure the probability it assigns. This is computed every 100 steps:

```
bits_known = log₂(N) + log P(S) / ln(2)
```

- `bits_known ≈ 0` → model is guessing uniformly
- `bits_known ≈ 10` → model has memorized the 10-bit secret


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

Note: we include a warmup step where we train the model on the uniform distribution of random numbers for 1000 steps for both single-step and multi-step environments. This ensures the model is in the correct format—we intentionally don't include a formatting reward in the training loop. Here are checkpoints to skip warmup:

```
SINGLE_STEP_CHECKPOINT = "tinker://61ffdf2c-c9ae-52f1-8b0a-d5757c68bee8:train:0/weights/final"
MULTI_STEP_CHECKPOINT = "tinker://1e79325e-97ad-5cfc-aae3-fdc7b5951746:train:0/weights/final"
```


## References

- [Thinking Machines - LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [Ord (2025) - The Inefficiency of RL](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning)
- [Li - Information Bandwidth of RL](https://richardli.xyz/post/information-bandwidth-rl/)
