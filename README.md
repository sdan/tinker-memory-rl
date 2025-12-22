# memory-rl

**How many bits can RL learn per episode?**

A minimal testbed to measure the information-theoretic efficiency of policy gradient methods. The hypothesis: scalar end-of-episode rewards bottleneck learning to ~1 bit/episode, regardless of model capacity.

## The Experiment

A secret integer $S \in [0, N)$ is fixed for the entire run. The model must learn to output $S$.

| Condition | Reward Signal | Theoretical Capacity |
|-----------|---------------|---------------------|
| **SFT** | "The answer is 42" | $\log_2 N$ bits/example |
| **Scalar RL** | 1 if correct, 0 otherwise | ≤1 bit/episode |
| **Binned RL** | log-distance binned into B levels | ≤$\log_2 B$ bits/episode |
| **Dense RL** | 1 per correct bit, k steps | $\log_2 N$ bits/episode |

We measure: **episodes to memorize** → **empirical bits/episode**

If the theory holds:
- Scalar RL should be ~$\log_2 N$ times slower than SFT
- Dense RL should match SFT

## Quick Start

```bash
pip install tinker
pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git
export TINKER_API_KEY=sk-...

# Scalar RL (the bottleneck)
python train.py env_type=single_step N=64 reward_type=binary

# Binned scalar RL (controlled bandwidth)
python train.py env_type=single_step N=64 reward_type=binned_log_distance reward_bins=8

# Dense RL (the control)
python train.py env_type=multi_step N=64

# SFT baseline
python sft_data_gen.py N=64 fixed_secret=42
python sft_train.py N=64 fixed_secret=42
```

Note: RL uses centered returns; you need at least 2 trajectories per update. `train.py` defaults to `group_size=4` (or use `batch_size>=2` / `group_size>=2`).

## Format Warmup (Optional)

The default RL training uses pre-trained warmup checkpoints. To train your own warmup models:

```bash
# Single-step warmup: teaches model to output numbers in correct format
python format_sft_train.py n_values="(64, 256, 1024)" preview_samples=True

# Multi-step warmup: teaches model to output bits in conversational format
python format_sft_train_multi_step.py n_values="(64, 256, 1024)" preview_samples=True
```

**What warmup does:**
- Trains on **all possible outputs** uniformly (not any specific secret)
- Teaches format compliance: "when asked, output a valid number/bit"
- Creates a checkpoint that RL can fine-tune to learn THE secret

**Using your own warmup checkpoint:**
```bash
# Get the sampler path from warmup output, then:
python train.py load_checkpoint_path="tinker://YOUR_CHECKPOINT_PATH" N=64
```

**Pre-trained checkpoints** (used by default):
```
SINGLE_STEP: tinker://61ffdf2c-c9ae-52f1-8b0a-d5757c68bee8:train:0/weights/final
MULTI_STEP:  tinker://1e79325e-97ad-5cfc-aae3-fdc7b5951746:train:0/weights/final
```

## Sweep Experiments

```bash
# Launch all sweeps (A, B, C, D) in parallel via tmux
python run_sweep.py

# Preview without launching
python run_sweep.py --dry-run

# Run specific sweeps
python run_sweep.py --sweep A B      # Only B sweep and N sweep
python run_sweep.py --sweep D        # Only SFT anchor

# With W&B logging
python run_sweep.py --wandb-project memory-rl

# Verbose mode
python run_sweep.py --verbose
```

**Sweeps:**
- **A**: B sweep (scalar bandwidth) @ fixed N=1024
- **B**: N sweep (scalar) @ fixed B=8
- **C**: Dense control (multi-step per-bit) on the same N grid
- **D**: SFT anchor @ N=1024

After launching, attach to the tmux session:

```bash
tmux attach-session -t memory-rl
```

**Control window commands:**
- `0-9`: Jump to window by number
- `↑↓`: Navigate job list
- `k`: Kill selected job
- `K`: Kill entire window group
- `q`: Detach

To kill the entire sweep:
```bash
tmux kill-session -t memory-rl
```

## Stop & Resume

All training scripts resume automatically if `log_path` already exists.

- Stop: `Ctrl+C` (single job) or `tmux kill-session -t memory-rl` (sweep)
- Resume: rerun the same command with the same `log_path`
- Extend a run: rerun with a larger `n_batches` (RL) or `num_epochs` (SFT) using the same `log_path`
- `run_sweep.py` skips runs with a `final` checkpoint; incomplete runs resume.

## Results (WIP)

Preliminary runs on Llama-3.1-8B, N=64 (6 bits):

![Continuous reward](https://github.com/user-attachments/assets/d132cbff-8ab8-4a72-9044-785ff88735a0)
![Binary reward](https://github.com/user-attachments/assets/6bf8ea62-21a8-4c60-afb6-41f23a25c5cd)
![Per-bit reward](https://github.com/user-attachments/assets/40afd8bf-8f10-488b-bf2d-1b5f8ad494fa)

## Why This Matters

> "You've done all this work that could be a minute of rollout, and you're sucking the bits of supervision of the final reward signal through a straw."
> — Karpathy

If RL is fundamentally bottlenecked by reward sparsity (not model capacity or algorithm choice), then:
1. LoRA makes sense for RLHF—you don't need many parameters to store ~1 bit/episode
2. Dense rewards (process supervision, per-step feedback) aren't just "nice to have"—they're the only way to match supervised efficiency
3. The "sample inefficiency of RL" may be an information-theoretic ceiling, not an algorithmic failure

## References

- [Ord (2025) - The Inefficiency of RL](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning)
- [Li - Information Bandwidth of RL](https://richardli.xyz/post/information-bandwidth-rl/)
- [Thinking Machines - LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
