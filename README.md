# memory-rl: Empirical Information Absorption Rates

This recipe is a focused memory test: the environment hides a fixed integer and the policy must recover it. By sweeping reward channels (binary, binned scalar, dense per-bit) and signal sizes, we measure how quickly RL absorbs information compared to SFT and to the theoretical “~1 bit/episode” bottleneck discussed in [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/).

This experiment measures the "bit rate" of Reinforcement Learning.

Recent theoretical discussions ([Ord](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning), [Li](https://richardli.xyz/post/information-bandwidth-rl/), [Schulman](https://thinkingmachines.ai/blog/lora/)) suggest that standard policy gradient methods with scalar rewards are bottlenecked by an information channel capacity of $\approx 1$ bit per episode. This hypothesis posits that RL fine-tuning requires fewer trainable parameters (e.g., LoRA) than supervised fine-tuning (SFT) because the training signal itself is sparse.

**memory-rl** is a minimal testbed designed to isolate and quantify this bottleneck. By training a model to memorize a fixed, latent integer $S$ under various reward regimes, we can empirically measure the **Information Absorption Rate** (bits per episode) and compare it to theoretical upper bounds.

At a high level, every run is characterized by three axes:

* **Signal size** (how many bits are “in” the secret).
* **Channel bandwidth** (how many bits of reward signal can pass per episode).
* **Storage capacity** (how many bits can be stored in the trainable parameters).

The code logs a minimal set of metrics for each axis; see [Metrics](#metrics) below.

## The Environment

The task is simple: The environment holds a latent secret integer $S \in [0, N-1]$. The agent must output $S$ to maximize reward.

*   **Signal size:** $H(S) = \log_2 N$ bits.
*   **Goal:** Measure the number of episodes ($E$) required to memorize $S$.
*   **Metric:** $\text{Empirical Bit Rate} = \dfrac{\log_2 N}{E}$.

We compare three distinct channel configurations:

1.  **SFT (Baseline):** The agent is essentially told "The secret is 42".
    *   *Theoretical capacity:* $\approx \log_2 N$ bits/example.
2.  **Scalar RL (Bottleneck):** The agent guesses and receives a single scalar at the end.
    *   *Binary:* Correct/Incorrect. (Capacity $\le 1$ bit/episode).
    *   *Binned distance:* Distance to $S$ quantized into $B$ bins. (Capacity $\le \log_2 B$ bits/episode).
3.  **Dense RL (Control):** The agent outputs bits; reward is given per-bit.
    *   *Theoretical capacity:* Sum of per-step rewards $\approx \log_2 N$ bits/episode.

## Theoretical Expectations

We do not yet have results. Based on the information-theoretic arguments, we set the following hypotheses:

*   **Exploration vs. bandwidth:** For purely binary scalar rewards, we expect the empirical bit rate to be $\ll 1$ for large $N$, heavily penalized by the difficulty of exploration (finding the gradient).
*   **Channel resolution:** For scalar rewards with informative distances (binned), learning speed should scale roughly with channel capacity ($\log_2 B$).
*   **Dense parity:** Dense (per-step) RL—despite being an RL algorithm—should approach the sample efficiency of SFT, confirming that the "inefficiency of RL" is a property of reward sparsity, not the policy gradient method itself.

## How the testbed and credit assignment work

This folder is a minimal "memorization laboratory": every **training split** uses a single **fixed secret** so the policy can concentrate on storing it. Test splits either reuse that secret or draw a seeded list of random secrets so evaluation is repeatable.

- **Single-step scalar channel (one reward per episode):**
  - Environment: `SingleStepEnv` shows the prompt once, the model answers once, and the episode ends.
  - Parsing: we take the **first integer token** from the reply; if parsing fails or the value is out of range, the guess is `None` and the distance is set to the worst case.
  - Reward: `compute_single_step_reward` maps the `(secret, guess)` pair to:
    - `binary`: `1.0` if guess equals secret, else `0.0`.
    - `binned_distance`: unsigned distance normalized to `[0, 1]`, bucketed into `reward_bins` with the bucket midpoint as reward (capacity ≈$\log_2 B$ bits/episode).
    - `binned_distance_signed`: like `binned_distance` but keeps a signed notion of over/under-shoot and maps it to `[-1, 1]`.
  - Credit assignment: `rl_train.py` uses **trajectory-level advantages** here by default (one scalar advantage per trajectory from the total episodic reward), so every token in the completion shares the same scalar credit. This matches the theoretical “≤ 1 bit per episode” channel.

- **Multi-step bitwise channel (per-step rewards):**
  - Environment: `MultiStepEnv` expands the secret into $k=\lceil \log_2 N\rceil$ bits and asks for **one bit per turn** until all bits are produced.
  - Reward per step: each step compares the parsed bit to the true bit at that position and pays `1.0` if correct, `0.0` otherwise. We accumulate these over the $k$ steps.
  - Metrics: we log `correct_bit` at each step, `bit_index`, `reward_signal`, and `bitstring_correct` on the final step (1 only if *all* bits were correct).
  - Credit assignment: `rl_train.py` turns on **stepwise advantages** for this env by default. We compute advantages from the per-transition rewards, optionally normalize them, and attach them to the tokens from each step separately. This gives up to `k` bits of reward signal per episode.

- **Sequence length and tokens:**
  - RL rollouts and SFT evals cap generation with `max_tokens` / `eval_max_tokens`. For scalar RL, only the first integer is used for reward; extra tokens are ignored by the reward function but still consume bandwidth and appear in token-count metrics.
  - For both envs, the RL core (`assemble_training_data`) spreads the chosen advantage scalar over the tokens of the sampled answer at that step, so every output token “shares” that reward for gradient purposes.

- **Supervised baseline as the same testbed:**
  - `sft_train.py` trains on JSONL conversations produced by `sft_data_gen.py` using standard NLL with `TrainOnWhat.LAST_ASSISTANT_MESSAGE`.
  - For evaluation, it reuses the RL envs via `RLTestSetEvaluator` with the same `SingleStepDatasetBuilder` / `MultiStepDatasetBuilder` and `max_tokens`, so SFT and RL are compared on exactly the same episodes and metrics (including `bits_learned`).

- **Channel bookkeeping and “bits learned”:**
  - `rl_train.py` logs run-level metadata (`cfg/log2_N`, `cfg/num_bits`, `theory/max_bits_per_episode`, `algo/type`) describing the reward channel capacity for that config.
  - `RLTestSetEvaluator` computes `env/...` metrics from trajectories and then derives `bits_learned` from `env/all/correct` (single-step) or `env/all/correct_bit` (multi-step) plus `N`. This gives a direct, comparable “downloaded bits” curve for both RL and SFT.

## Usage

We primarily use `uv` for environment management.

### 0. Full memory testbed in one go
Runs binary RL, binned RL, per-bit RL, and the SFT baseline across multiple `N` values. Outputs live under `/tmp/tinker-examples/memory_rl_sweep/...`.

```bash
uv run python -m tinker_cookbook.recipes.memory_rl.run \
    N_values='(4,16,64,256)' \
    rl_batches=200 rl_batch_size=128 rl_learning_rate=4e-5 \
    sft_train_examples=1000 sft_test_examples=200 sft_epochs=2 \
    wandb_project=memorize_rl \
    parallel_mode=sequential
```

### 1. Supervised Baseline (SFT)
Establishes the "speed of light" for memorization on this architecture.

```bash
# Train SFT on a secret of size N=64 (6 bits)
uv run python -m tinker_cookbook.recipes.memory_rl.sft_train \
    N=64 \
    learning_rate=1e-4 \
    batch_size=32 \
    n_steps=500 \
    wandb_project=memory-rl-sft
```

### 2. Scalar RL (Single-Step)
Tests the "1 bit per episode" hypothesis. Sweep `reward_bins` to test channel capacity scaling.

```bash
# Hard mode: Binary reward (Success/Fail)
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=single_step \
    N=64 \
    reward_type=binary \
    learning_rate=4e-5 \
    lora_rank=8 \
    wandb_project=memory-rl-scalar

# Informative mode: Distance quantized into 8 bins (3 bits theoretical max)
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=single_step \
    N=64 \
    reward_type=binned_distance \
    reward_bins=8 \
    learning_rate=4e-5 \
    wandb_project=memory-rl-scalar
```

### 3. Dense RL (Multi-Step)
Tests if per-step rewards restore high-bandwidth learning. The model outputs the secret as a bitstring.

```bash
# Dense rewards: +1 per correct bit
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=multi_step \
    N=64 \
    learning_rate=4e-5 \
    wandb_project=memory-rl-dense
```

### 4. Debugging / Preview
To visualize the prompt structure and model outputs without training:

```bash
uv run python -m tinker_cookbook.recipes.memory_rl.preview_env \
    env_type=single_step \
    N=64 \
    num_preview_examples=5 \
    preview_temperature=1.0

### 5. Quick “how to run” commands per method (chz style)
These mirror the defaults in this folder. Adjust `wandb_project`, `checkpoint_path`, or `fixed_secret` as needed.

**RL — single-step, binary reward (sparse):**
```bash
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=single_step \
    N=64 \
    n_batches=1000 \
    batch_size=128 \
    learning_rate=4e-5 \
    reward_type=binary \
    fixed_secret=32 \
    wandb_project=memorize_rl
```

**RL — single-step, binned distance (denser scalar channel):**
```bash
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=single_step \
    N=64 \
    n_batches=1000 \
    batch_size=128 \
    learning_rate=4e-5 \
    reward_type=binned_distance \
    reward_bins=8 \
    fixed_secret=32 \
    wandb_project=memorize_rl
```

**RL — multi-step, per-bit reward (dense, k steps):**
```bash
uv run python -m tinker_cookbook.recipes.memory_rl.rl_train \
    env_type=multi_step \
    N=64 \
    n_batches=1000 \
    batch_size=128 \
    learning_rate=4e-5 \
    fixed_secret=32 \
    wandb_project=memorize_rl
```

**SFT baseline (generate data then train, matching the single-step prompt):**
```bash
# 1) Generate supervised data for a fixed secret (train/test JSONL)
uv run python -m tinker_cookbook.recipes.memory_rl.sft_data_gen \
    N=64 \
    fixed_secret=32 \
    num_train=1000 \
    num_test=200 \
    output_dir=/tmp/tinker-examples/memory_rl/sft_data

# 2) Train SFT on that data and evaluate on the RL env
uv run python -m tinker_cookbook.recipes.memory_rl.sft_train \
    N=64 \
    train_data_path=/tmp/tinker-examples/memory_rl/sft_data/train.jsonl \
    test_data_path=/tmp/tinker-examples/memory_rl/sft_data/test.jsonl \
    num_epochs=2 \
    batch_size=128 \
    eval_env_type=single_step \
    reward_type=binary \
    wandb_project=memorize_rl
```
Note: the SFT CLI accepts `reward_type=binary` (mirrors the RL binary env) or `reward_type=normalized_distance` for a slightly denser eval signal.

## Metrics

The experiment is instrumented to log a small set of metrics aligned with the three axes above.

### Run-level (logged once per run)

These are recorded when `rl_train.py` starts:

- `cfg/env_type` – `"single_step"` or `"multi_step"`.
- `algo/type` – `"rl_end_binary"`, `"rl_end_binned"`, or `"rl_per_step"`.
- `cfg/reward_type`, `cfg/reward_bins` – reward regime and bin count (for scalar RL).
- `cfg/log2_N` – $\log_2 N$, the entropy of the secret.
- `cfg/num_bits` – $\lceil \log_2 N \rceil$, number of bits in the dense/bitwise setting.
- `signal/total_bits` – total bits of signal (single-secret case: equal to `cfg/log2_N`).
- `theory/max_bits_per_episode` – channel capacity upper bound for this regime:
  - SFT: $\approx \log_2 N$
  - single-step binary RL: $1.0$
  - single-step binned RL: $\log_2(\text{reward_bins})$
  - multi-step RL: `cfg/num_bits`
- `model/trainable_params` – number of trainable parameters (from the training client, if available).
- `capacity/bit_capacity` – estimated storage capacity $\approx 2 \times \text{trainable\_params}$.
- `summary/load_ratio` – `signal/total_bits / capacity/bit_capacity`.

These are enough to place each run in the Signal / Bandwidth / Capacity space.

### Train-time RL metrics (per iteration)

Logged on both train and eval steps (eval prefixed with `test/`):

**Reward channel (bandwidth proxy):**

- `env/all/reward/total` – mean total episode reward.
- `env/all/reward/entropy_bits` – empirical entropy $H(R_\text{episode})$ of total rewards (bits/episode).

**Task correctness:**

- Single-step env:
  - `env/all/correct` – fraction of episodes where the integer guess equals the secret.
  - `env/all/distance` – mean absolute distance between guess and secret.
- Multi-step env:
  - `env/all/correct_bit` – per-bit accuracy.
  - `env/all/bitstring_correct` – fraction of episodes where the entire bitstring matches.

These metrics are aggregated across tags (e.g., `env/single_step_.../correct`) as well as `env/all/...`.

**Policy behavior:**

- `env/all/policy/entropy_per_token_bits` – average policy entropy per generated token (bits).
  - High: exploratory; low: collapsed / deterministic.

Additional existing RL metrics (KL, advantage stats, tokens per turn, timing, etc.) are still logged by the shared RL core but are not specific to this experiment.

### Eval-time “bits learned” (download bar)

The `RLTestSetEvaluator` used by both RL and SFT env evaluations logs:

- `bits_learned` – an approximate mutual information between the secret and the model’s predictions:
  - For single-step (scalar) envs: derived from `env/all/correct` and the class count $N$.
  - For multi-step (bitwise) envs: derived from per-bit accuracy (`env/all/correct_bit`) and scaled by $\log_2 N$ bits.

On eval steps this appears as `test/bits_learned` (or `test/env/bits_learned` depending on configuration). It gives a continuous “download bar” in bits, whereas `env/all/bitstring_correct` and `env/all/correct` are step functions.

### SFT runs

The SFT script (`sft_train.py`) logs the usual supervised metrics:

- `train/loss`, `eval/loss` – NLL on SFT data.

If env evaluation is enabled (the default in this recipe), it also runs the same `RLTestSetEvaluator` as RL, so you get:

- `test/env/all/correct` / `test/env/all/bitstring_correct`
- `test/env/all/reward/total`, `test/env/all/reward/entropy_bits`
- `test/env/bits_learned`

This makes SFT directly comparable to RL on the same metric axes: signal size, reward channel, bits learned, and episodes to memorize.

## Sweep runner (sequential or tmux)

Use `run.py` for small sweeps that stay consistent with the chz CLIs in this folder. It can launch everything sequentially (default) or fan out with tmux sessions (one tmux session per N/seed).

Sequential sweep (writes to `/tmp/tinker-examples/memory_rl_sweep/run_<timestamp>`):

```bash
uv run python -m tinker_cookbook.recipes.memory_rl.run \
    N_values='(4,16,64,256)' \
    rl_batches=200 \
    rl_batch_size=128 \
    rl_learning_rate=4e-5 \
    sft_train_examples=1000 \
    sft_test_examples=200 \
    sft_epochs=2 \
    checkpoint_path=tinker://61ffdf2c-c9ae-52f1-8b0a-d5757c68bee8/weights/final \
    wandb_project=memorize_rl \
    parallel_mode=sequential
```

Tmux mode (methods launch sequentially, but N×seed variants run in parallel tmux sessions):

```bash
uv run python -m tinker_cookbook.recipes.memory_rl.run \
    parallel_mode=tmux_method \
    session_prefix=memrl \
    base_dir=/tmp/tinker-examples/memory_rl_sweep
```

Defaults:

- Methods: `rl_binary`, `rl_binned` (single-step), `rl_per_bit` (multi-step), and `sft`.
- Seeds: RL methods=3, SFT=2.
- Learning rate / batch knobs match the examples above.

Each run logs stdout/stderr to its experiment directory and writes an `experiment_results.json` summary in the sweep folder. In tmux mode, use `tmux ls` to check sessions and `tmux attach -t <name>` to peek at a run.
