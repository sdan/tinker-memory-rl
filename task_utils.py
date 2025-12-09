from __future__ import annotations

import math
import re
from typing import Literal


RewardType = Literal[
    "binary",
    "log_distance",
]


def validate_secret(secret: int, N: int) -> None:
    if N <= 0:
        raise ValueError("N must be positive")
    if not (0 <= secret < N):
        raise ValueError(f"Secret {secret} must be in [0, {N - 1}] when N={N}")


def validate_reward_config(reward_type: RewardType) -> None:
    if reward_type not in {"binary", "log_distance"}:
        raise ValueError(
            f"Unsupported reward_type '{reward_type}'. "
            f"Use 'binary' or 'log_distance'."
        )


def build_single_step_user_prompt(N: int) -> str:
    return (
        "You are playing a memorization game. "
        f"There is a secret integer between 0 and {N - 1}. "
        "Output ONLY the integer, on the first line, with NO extra text.\n"
        "What is the secret number?"
    )


def build_multi_step_system_prompt(N: int) -> str:
    num_bits = num_bits_for_space(N)
    return (
        f"You are playing a bit-by-bit memorization game. "
        f"The secret number is between 0 and {N - 1} and uses {num_bits} bits. "
        f"Output exactly one bit (0 or 1) per response with no punctuation."
    )


def build_multi_step_user_prompt(position: int, num_bits: int) -> str:
    remaining = num_bits - position - 1
    return f"Provide bit #{position + 1}. Remaining bits after this: {remaining}."


def parse_single_step_guess(sample_str: str, N: int) -> int | None:
    stripped = sample_str.strip()
    if not stripped:
        return None

    # Sometimes the tokenizer splits larger numbers into multiple tokens
    digits = re.sub(r'[^\d]', '', stripped.split('\n')[0])
    if not digits:
        return None

    try:
        guess = int(digits)
    except ValueError:
        return None

    return guess if 0 <= guess < N else None


def parse_bit_guess(sample_str: str) -> str | None:
    stripped = sample_str.strip()
    if not stripped:
        return None
    ch = stripped[0]
    if ch in ("0", "1"):
        return ch
    return None


def compute_single_step_reward(
    secret: int,
    guess: int | None,
    reward_type: RewardType,
    N: int,
) -> tuple[float, bool, float]:
    max_bits = math.log2(N) if N > 1 else 1.0

    if guess is None:
        distance = float(N - 1)
        correct = False
    else:
        distance = float(abs(guess - secret))
        correct = guess == secret

    if reward_type == "binary":
        base_reward = float(correct)
    elif reward_type == "log_distance":
        base_reward = max(0.0, 1.0 - (math.log2(distance + 1) / max_bits))

    return base_reward, correct, distance


def num_bits_for_space(N: int) -> int:
    if N <= 1:
        return 1
    return max(1, math.ceil(math.log2(N)))


def format_secret_bits(secret: int, N: int) -> str:
    width = num_bits_for_space(N)
    return format(secret, f"0{width}b")
