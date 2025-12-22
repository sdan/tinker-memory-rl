import asyncio
import logging
import math
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

# Use local envs/ module instead of tinker_cookbook.recipes.memory_rl
from envs import (
    MultiStepDatasetBuilder,
    SingleStepDatasetBuilder,
    num_bits_for_space,
    validate_reward_config,
)
from bits_evaluator import BitsKnownEvaluator

logger = logging.getLogger(__name__)


SINGLE_STEP_CHECKPOINT = "tinker://61ffdf2c-c9ae-52f1-8b0a-d5757c68bee8:train:0/weights/final"
MULTI_STEP_CHECKPOINT = "tinker://1e79325e-97ad-5cfc-aae3-fdc7b5951746:train:0/weights/final"


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: str | None = None
    load_checkpoint_path: str | None = SINGLE_STEP_CHECKPOINT

    N: int = 16
    reward_type: str = "binary"
    fixed_secret: int | None = None
    use_standard_prefix: bool = False
    env_type: Literal["single_step", "multi_step"] = "single_step"

    batch_size: int = 1
    group_size: int = 1
    n_batches: int = 1000
    lora_rank: int = 1
    learning_rate: float = 4e-5
    max_tokens: int = 8
    eval_every: int = 10
    eval_steps: tuple[int, ...] | None = chz.field(
        default=None,
        munger=lambda _, v: (
            tuple(int(x) for x in v.split(",") if x)
            if isinstance(v, str)
            else (tuple(v) if v is not None else None)
        ),
    )
    save_every: int = 20
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    dataset_seed: int = 0
    eval_bits: bool = True

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    use_stepwise_advantages: bool | None = None
    normalize_advantages: bool = True
    advantage_norm_eps: float = 1e-8


def build_config(cli: Config) -> train.Config:
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)

    if cli.env_type == "single_step":
        validate_reward_config(cli.reward_type)

    prefix = "standard" if cli.use_standard_prefix else None

    if cli.env_type == "single_step":
        builder = SingleStepDatasetBuilder(
            batch_size=cli.batch_size,
            group_size=cli.group_size,
            renderer_name=renderer_name,
            model_name_for_tokenizer=cli.model_name,
            N=cli.N,
            reward_type=cli.reward_type,
            n_batches=cli.n_batches,
            fixed_secret=cli.fixed_secret,
            convo_prefix=prefix,
            seed=cli.dataset_seed,
        )
    elif cli.env_type == "multi_step":
        builder = MultiStepDatasetBuilder(
            batch_size=cli.batch_size,
            group_size=cli.group_size,
            renderer_name=renderer_name,
            model_name_for_tokenizer=cli.model_name,
            N=cli.N,
            n_batches=cli.n_batches,
            fixed_secret=cli.fixed_secret,
            convo_prefix=prefix,
            seed=cli.dataset_seed,
        )
    else:
        raise ValueError(f"Unknown env_type {cli.env_type}")

    date_and_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_run_name = (
        f"{cli.env_type}_{cli.reward_type}_N{cli.N}_bs{cli.batch_size}_gs{cli.group_size}_lr{cli.learning_rate}_{date_and_time}"
    )
    log_path = cli.log_path or f"/tmp/tinker-examples/memory_rl/{default_run_name}"
    wandb_name = cli.wandb_name or default_run_name

    entropy_bits = math.log2(cli.N) if cli.N > 0 else 0.0
    num_bits = num_bits_for_space(cli.N)
    if cli.env_type == "single_step":
        if cli.reward_type == "binary":
            algo_type = "rl_end_binary"
            max_bits = 1.0
        elif cli.reward_type == "log_distance":
            algo_type = "rl_end_continuous"
            max_bits = None
        else:
            algo_type = "rl_end_unknown"
            max_bits = None
    else:
        algo_type = "rl_per_step"
        max_bits = float(num_bits)

    initial_metrics = {
        "cfg/env_type": cli.env_type,
        "algo/type": algo_type,
        "cfg/reward_type": cli.reward_type,
        "cfg/log2_N": entropy_bits,
        "cfg/num_bits": num_bits,
        "signal/total_bits": entropy_bits,
        "theory/max_bits_per_episode": max_bits,
    }
    initial_metrics = {k: v for k, v in initial_metrics.items() if v is not None}

    use_stepwise_advantages = (
        cli.use_stepwise_advantages
        if cli.use_stepwise_advantages is not None
        else (cli.env_type == "multi_step")
    )

    evaluator_builders = []
    if cli.eval_bits:
        evaluator_builders.append(
            lambda builder=builder, env_type=cli.env_type: BitsKnownEvaluator(
                dataset_builder=builder,
                env_type=env_type,
                metric_prefix="test/bits",
            )
        )

    return train.Config(
        model_name=cli.model_name,
        load_checkpoint_path=cli.load_checkpoint_path,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli.learning_rate,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        eval_every=cli.eval_every,
        eval_steps=cli.eval_steps,
        save_every=cli.save_every,
        loss_fn=cli.loss_fn,
        wandb_project=cli.wandb_project,
        wandb_name=wandb_name,
        initial_metrics=initial_metrics,
        evaluator_builders=evaluator_builders,
        use_stepwise_advantages=use_stepwise_advantages,
        normalize_advantages=cli.normalize_advantages,
        advantage_norm_eps=cli.advantage_norm_eps,
    )


async def main():
    cli = chz.entrypoint(Config)
    config = build_config(cli)

    entropy_bits = math.log2(cli.N) if cli.N > 0 else 0
    num_bits = num_bits_for_space(cli.N)
    if cli.env_type == "single_step":
        channel_desc = f"{cli.reward_type} (1 bit max)" if cli.reward_type == "binary" else "continuous"
    else:
        channel_desc = f"{num_bits} bits (1 per step)"

    logger.info(f"Environment: {cli.env_type}, N={cli.N} ({entropy_bits:.2f} bits)")
    logger.info(f"Channel: {channel_desc}")
    logger.info(f"Secret: {cli.fixed_secret if cli.fixed_secret is not None else 'Random'}")
    logger.info(f"Batches: {cli.n_batches} × {cli.batch_size} × {cli.group_size}")
    logger.info(f"Log path: {config.log_path}")

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    await train.main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
