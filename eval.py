from typing import Literal

import numpy as np
import tinker
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.renderers import TrainOnWhat

from envs import (
    build_multi_step_system_prompt,
    build_multi_step_user_prompt,
    build_single_step_user_prompt,
    compute_bits_known,
    format_secret_bits,
    num_bits_for_space,
    sum_weighted_logprobs,
)


class BitsKnownEvaluator(SamplingClientEvaluator):
    """Evaluates bits of information the model has learned about the secret."""

    def __init__(
        self,
        dataset_builder,
        env_type: Literal["single_step", "multi_step"],
        metric_prefix: str = "bits",
    ):
        self._dataset_builder = dataset_builder
        self._env_type = env_type
        self._metric_prefix = metric_prefix.rstrip("/")
        self._dataset = None

    async def _ensure_dataset(self):
        if self._dataset is None:
            _, test_dataset = await self._dataset_builder()
            if test_dataset is None:
                raise ValueError("BitsKnownEvaluator requires a test dataset.")
            self._dataset = test_dataset
        return self._dataset

    def _get_eval_secrets(self, dataset) -> list[int]:
        if hasattr(dataset, "test_secrets"):
            return [int(x) for x in dataset.test_secrets]
        if getattr(dataset, "fixed_secret", None) is not None:
            total = len(dataset) * dataset.batch_size
            return [int(dataset.fixed_secret)] * total
        raise ValueError("Test dataset does not expose secrets for evaluation.")

    def _build_messages(self, N: int, secret: int, convo_prefix: list[dict] | None) -> list[dict]:
        prefix = list(convo_prefix or [])
        if self._env_type == "single_step":
            return prefix + [
                {"role": "user", "content": build_single_step_user_prompt(N)},
                {"role": "assistant", "content": str(secret)},
            ]

        num_bits = num_bits_for_space(N)
        secret_bits = format_secret_bits(secret, N)
        messages = [{"role": "system", "content": build_multi_step_system_prompt(N)}] + prefix
        for idx, bit in enumerate(secret_bits):
            messages.append({"role": "user", "content": build_multi_step_user_prompt(idx, num_bits)})
            messages.append({"role": "assistant", "content": bit})
        return messages

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        dataset = await self._ensure_dataset()
        renderer = dataset.renderer
        N = dataset.N
        train_on = (
            TrainOnWhat.LAST_ASSISTANT_MESSAGE
            if self._env_type == "single_step"
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        secrets = self._get_eval_secrets(dataset)

        bits_raw = []
        bits_clamped = []
        target_logprobs = []

        for secret in secrets:
            messages = self._build_messages(N, secret, dataset.convo_prefix)
            tokens, weights = renderer.build_supervised_example(messages, train_on_what=train_on)
            token_list = tokens.tolist()
            weight_list = weights.tolist()

            logprobs = await sampling_client.compute_logprobs_async(
                tinker.ModelInput.from_ints(token_list)
            )
            target_logprob, _ = sum_weighted_logprobs(logprobs, weight_list)
            bits_known_raw, bits_known_clamped = compute_bits_known(target_logprob, N)

            bits_raw.append(bits_known_raw)
            bits_clamped.append(bits_known_clamped)
            target_logprobs.append(target_logprob)

        bits_raw_arr = np.array(bits_raw, dtype=float)
        bits_clamped_arr = np.array(bits_clamped, dtype=float)
        target_logprob_arr = np.array(target_logprobs, dtype=float)

        prefix = self._metric_prefix
        return {
            f"{prefix}/known_raw_mean": float(bits_raw_arr.mean()),
            f"{prefix}/known_raw_std": float(bits_raw_arr.std()),
            f"{prefix}/known_clamped_mean": float(bits_clamped_arr.mean()),
            f"{prefix}/known_clamped_std": float(bits_clamped_arr.std()),
            f"{prefix}/target_logprob_mean": float(target_logprob_arr.mean()),
            f"{prefix}/target_logprob_std": float(target_logprob_arr.std()),
        }
