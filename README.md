# memory-rl

Empirical measurement of RL's information absorption rate.

## Installation

```bash
# Get API access at https://thinkingmachines.ai/
export TINKER_API_KEY=sk-...

# Install dependencies
pip install tinker
pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git

# Clone this repo
git clone https://github.com/sdan/tinker-memory-rl.git
cd tinker-memory-rl
```

## Motivation

> You've done all this work that could be a minute of rollout, and you're sucking the bits of supervision of the final reward signal through a straw and you're broadcasting that across the entire trajectory and using that to upweight or downweight that trajectory. It's just stupid and crazy.
>
> — Andrej Karpathy

Recent theoretical discussions ([Ord](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning), [Li](https://richardli.xyz/post/information-bandwidth-rl/), [Schulman](https://thinkingmachines.ai/blog/lora/)) suggest that standard policy gradient methods with scalar rewards are bottlenecked by an information channel capacity of ~1 bit per episode.

**memory-rl** is a minimal testbed to empirically measure this bottleneck. We train a model to memorize a fixed integer $S \in [0, N-1]$ under different reward regimes and measure the **Information Absorption Rate** (bits per episode).

## The Environment

The task is simple: the environment holds a latent secret integer $S$. The agent must output $S$ to maximize reward.

- **Signal size:** $H(S) = \log_2 N$ bits
- **Goal:** Measure episodes $E$ required to memorize $S$
- **Metric:** Empirical Bit Rate $= \frac{\log_2 N}{E}$

We compare three channel configurations:

| Method | Channel | Theoretical Capacity |
|--------|---------|---------------------|
| **SFT** | Direct supervision | ~$\log_2 N$ bits/example |
| **Scalar RL** | Single reward at episode end | ≤1 bit/episode (binary) |
| **Dense RL** | Per-bit reward across k steps | ~$\log_2 N$ bits/episode |

## Theoretical Expectations

- **Binary rewards:** Empirical bit rate should be ≪1 for large N due to exploration difficulty
- **Dense rewards:** Should approach SFT efficiency, confirming "RL inefficiency" is about reward sparsity, not policy gradients

## Usage

### 1. Generate SFT Data

```bash
python sft_data_gen.py \
    N=64 \
    fixed_secret=42 \
    num_train=1000 \
    output_dir=/tmp/memory_rl/sft_data
```

### 2. Train SFT Baseline

```bash
python sft_train.py \
    N=64 \
    fixed_secret=42 \
    train_data_path=/tmp/memory_rl/sft_data/train.jsonl \
    learning_rate=2e-4 \
    wandb_project=memory-rl
```

### 3. Scalar RL (Single-Step)

```bash
# Binary reward (≤1 bit/episode)
python train.py \
    env_type=single_step \
    N=64 \
    reward_type=binary \
    fixed_secret=42 \
    learning_rate=4e-5 \
    wandb_project=memory-rl

# Log-distance reward (continuous)
python train.py \
    env_type=single_step \
    N=64 \
    reward_type=log_distance \
    fixed_secret=42 \
    learning_rate=4e-5 \
    wandb_project=memory-rl
```

### 4. Dense RL (Multi-Step)

```bash
# Per-bit reward (~log2(N) bits/episode)
python train.py \
    env_type=multi_step \
    N=64 \
    fixed_secret=42 \
    learning_rate=4e-5 \
    wandb_project=memory-rl
```

## Files

| File | Description |
|------|-------------|
| `train.py` | RL training loop (single-step and multi-step) |
| `single_step_env.py` | Scalar reward environment |
| `multi_step_env.py` | Per-bit dense reward environment |
| `task_utils.py` | Shared prompts, parsing, reward functions |
| `sft_train.py` | Supervised fine-tuning baseline |
| `sft_data_gen.py` | Generate SFT training data |

## Reward Types

### Single-Step Environment

- **`binary`**: 1.0 if guess == secret, else 0.0
- **`log_distance`**: `max(0, 1 - log2(|guess - secret| + 1) / log2(N))`

### Multi-Step Environment

- Per-bit reward: 1.0 if bit matches, else 0.0
- Episode length: $k = \lceil \log_2 N \rceil$ steps

## Early Results

### RL with continuous distance reward
![Screenshot 2025-12-01 at 11:42:29 PM](https://github.com/user-attachments/assets/d132cbff-8ab8-4a72-9044-785ff88735a0)

### RL with binary reward
![Screenshot 2025-12-01 at 11:45:11 PM](https://github.com/user-attachments/assets/6bf8ea62-21a8-4c60-afb6-41f23a25c5cd)

### RL with per-bit (dense) reward
![Screenshot 2025-12-01 at 11:45:51 PM](https://github.com/user-attachments/assets/40afd8bf-8f10-488b-bf2d-1b5f8ad494fa)

## References

- [Toby Ord - The Inefficiency of Reinforcement Learning](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning)
- [Richard Li - Information Bandwidth of RL](https://richardli.xyz/post/information-bandwidth-rl/)
- [Thinking Machines - LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
