import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Literal

import chz
import numpy as np
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer

from envs import (
    build_multi_step_system_prompt,
    build_multi_step_user_prompt,
    build_single_step_user_prompt,
    format_secret_bits,
    num_bits_for_space,
    validate_secret,
)

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: str | None = None

    model_path: str | None = None
    base_model: str | None = None

    env_type: Literal["single_step", "multi_step"] = "single_step"
    N: int = 16
    fixed_secret: int | None = None
    use_standard_prefix: bool = False

    eval_episodes: int = 100
    seed: int = 0
    output_path: str | None = None


def _get_renderer(model_name: str, renderer_name: str | None) -> renderers.Renderer:
    tokenizer = get_tokenizer(model_name)
    resolved_name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    return renderers.get_renderer(resolved_name, tokenizer=tokenizer)


def _build_single_step_messages(N: int, secret: int, use_standard_prefix: bool) -> list[renderers.Message]:
    messages: list[renderers.Message] = []
    if use_standard_prefix:
        messages.extend(
            [
                {"role": "user", "content": build_single_step_user_prompt(N)},
                {"role": "assistant", "content": "0"},
            ]
        )
    messages.append({"role": "user", "content": build_single_step_user_prompt(N)})
    messages.append({"role": "assistant", "content": str(secret)})
    return messages


def _build_multi_step_messages(
    N: int, secret: int, use_standard_prefix: bool
) -> tuple[list[renderers.Message], int]:
    num_bits = num_bits_for_space(N)
    secret_bits = format_secret_bits(secret, N)

    messages: list[renderers.Message] = [
        {"role": "system", "content": build_multi_step_system_prompt(N)},
    ]
    if use_standard_prefix:
        messages.extend(
            [
                {"role": "user", "content": build_multi_step_user_prompt(0, num_bits)},
                {"role": "assistant", "content": "0"},
            ]
        )
    for idx, bit in enumerate(secret_bits):
        messages.append({"role": "user", "content": build_multi_step_user_prompt(idx, num_bits)})
        messages.append({"role": "assistant", "content": bit})

    return messages, num_bits


def _sum_weighted_logprobs(
    logprobs: list[float | None], weights: list[float]
) -> tuple[float, int]:
    total = 0.0
    count = 0
    for logprob, weight in zip(logprobs, weights, strict=True):
        if weight <= 0:
            continue
        if logprob is None:
            continue
        total += logprob * weight
        count += 1
    return total, count


async def _score_secret(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    env_type: str,
    N: int,
    secret: int,
    use_standard_prefix: bool,
) -> dict[str, float | int]:
    if env_type == "single_step":
        messages = _build_single_step_messages(N, secret, use_standard_prefix)
        train_on_what = TrainOnWhat.LAST_ASSISTANT_MESSAGE
        signal_bits = math.log2(N) if N > 1 else 1.0
    else:
        messages, _num_bits = _build_multi_step_messages(N, secret, use_standard_prefix)
        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES
        signal_bits = math.log2(N) if N > 1 else 1.0

    # build_supervised_example gives a tokenized transcript plus a 0/1 mask for target tokens.
    tokens, weights = renderer.build_supervised_example(messages, train_on_what=train_on_what)
    token_list = tokens.tolist()
    weight_list = weights.tolist()

    prompt = tinker.ModelInput.from_ints(token_list)
    logprobs = await sampling_client.compute_logprobs_async(prompt)

    target_logprob, target_token_count = _sum_weighted_logprobs(logprobs, weight_list)

    # bits_known = log2(N) + log2 P(target | prompt); this measures information about the secret.
    bits_known_raw = signal_bits + (target_logprob / math.log(2))
    bits_known_clamped = max(0.0, min(signal_bits, bits_known_raw))

    return {
        "secret": int(secret),
        "target_logprob": target_logprob,
        "target_token_count": int(target_token_count),
        "bits_known_raw": bits_known_raw,
        "bits_known_clamped": bits_known_clamped,
    }


async def main() -> None:
    cfg = chz.entrypoint(Config)

    if (cfg.model_path is None) == (cfg.base_model is None):
        raise ValueError("Provide exactly one of model_path or base_model.")

    if cfg.fixed_secret is not None:
        validate_secret(cfg.fixed_secret, cfg.N)

    renderer = _get_renderer(cfg.model_name, cfg.renderer_name)

    service_client = tinker.ServiceClient()
    if cfg.model_path is not None:
        sampling_client = service_client.create_sampling_client(model_path=cfg.model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=cfg.base_model)

    rng = np.random.RandomState(cfg.seed)
    if cfg.fixed_secret is not None:
        secrets = [cfg.fixed_secret] * cfg.eval_episodes
    else:
        secrets = rng.randint(0, cfg.N, size=cfg.eval_episodes).tolist()

    records: list[dict[str, float | int]] = []
    for secret in secrets:
        record = await _score_secret(
            sampling_client=sampling_client,
            renderer=renderer,
            env_type=cfg.env_type,
            N=cfg.N,
            secret=int(secret),
            use_standard_prefix=cfg.use_standard_prefix,
        )
        records.append(record)

    if cfg.output_path:
        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    mean_bits = float(np.mean([r["bits_known_raw"] for r in records]))
    mean_bits_clamped = float(np.mean([r["bits_known_clamped"] for r in records]))
    mean_logprob = float(np.mean([r["target_logprob"] for r in records]))
    mean_token_count = float(np.mean([r["target_token_count"] for r in records]))

    logger.info("Eval summary")
    logger.info(f"env_type={cfg.env_type} N={cfg.N} episodes={cfg.eval_episodes}")
    logger.info(f"mean_target_logprob={mean_logprob:.4f}")
    logger.info(f"mean_target_token_count={mean_token_count:.2f}")
    logger.info(f"mean_bits_known_raw={mean_bits:.4f}")
    logger.info(f"mean_bits_known_clamped={mean_bits_clamped:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
