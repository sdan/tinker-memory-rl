# Architecture

Memory RL recipe for studying information transmission through RL training.

## Module Structure

```
tinker-memory-rl/
├── train.py                 # RL training entry point
├── sft_train.py             # SFT baseline entry point
├── sft_data_gen.py          # Generate SFT training data
├── format_sft_train.py      # Format warmup (single-step)
├── format_sft_train_multi_step.py  # Format warmup (multi-step)
├── run_sweep.py             # Launch experiment sweeps via tmux
├── eval.py                  # Information-theoretic evaluation (BitsKnownEvaluator)
└── envs/
    ├── single_step_env.py   # Single-shot integer guessing
    ├── multi_step_env.py    # Bit-by-bit guessing with per-step reward
    └── task_utils.py        # Shared prompts, parsing, reward computation
```

## Data Flow

```
CLI Config (@chz.chz)
    │
    ▼
DatasetBuilder (creates train/test splits)
    │
    ▼
RLDataset.get_batch(index) → list[EnvGroupBuilder]
    │
    ▼
EnvGroupBuilder.make_envs() → list[Env]
    │
    ▼
Env.step(action) → reward, done
    │
    ▼
Training loop (tinker_cookbook.rl.train)
    │
    ▼
Evaluation (BitsKnownEvaluator)
```

## Environment Types

### Single-Step (`env_type="single_step"`)
- Model outputs one integer guess per episode
- Reward computed at end: binary, log_distance, or binned_log_distance
- Use for studying end-of-episode feedback

### Multi-Step (`env_type="multi_step"`)
- Model outputs one bit per turn (k turns for k-bit secret)
- Per-step reward: +1 correct bit, -1 wrong bit
- Use for studying dense per-step feedback

## Reward Types (single-step only)

| Type | Formula | Use Case |
|------|---------|----------|
| `binary` | 1 if exact, 0 otherwise | Sparse signal baseline |
| `log_distance` | `1 - log2(1 + \|guess - secret\|) / log2(N)` | Continuous shaping |
| `binned_log_distance` | Discretized log_distance into B bins | Controlled bandwidth experiments |

## Key Design Decisions

1. **Fixed secret per run**: Same secret for all episodes during training. Prevents memorizing input→output mapping; forces weight-based storage.

2. **Centered returns**: `group_size >= 2` required. Advantages computed relative to group mean (no value network).

3. **Format warmup**: Pre-train on uniform output distribution before RL. Teaches format compliance without teaching THE secret.

4. **Deterministic secrets**: `secret_for(seed, N) = (seed * 9973 + 12345) % N` for reproducibility.

## Extension Points

**Add new reward type:**
1. Add to `RewardType` literal in `single_step_env.py`
2. Implement in `compute_single_step_reward()` in `task_utils.py`
3. Add validation in `validate_reward_config()`

**Add new environment:**
1. Create `envs/new_env.py` with `Env`, `EnvGroupBuilder`, `Dataset`, `DatasetBuilder`
2. Add to `env_type` literal in `train.py`
3. Wire up in `build_config()`

## Checkpoints

Format warmup checkpoints (used by default):
```
SINGLE_STEP: tinker://61ffdf2c-c9ae-52f1-8b0a-d5757c68bee8:train:0/weights/final
MULTI_STEP:  tinker://1e79325e-97ad-5cfc-aae3-fdc7b5951746:train:0/weights/final
```

Override with `load_checkpoint_path="tinker://YOUR_PATH"`.
