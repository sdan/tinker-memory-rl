import json
import logging
from pathlib import Path

import chz

from tinker_cookbook.recipes.memory_rl.task_utils import build_single_step_user_prompt

logger = logging.getLogger(__name__)


def generate_sft_data(N: int, secret: int, num_examples: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_single_step_user_prompt(N)

    with output_path.open("w") as f:
        for _ in range(num_examples):
            example = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": str(secret)},
                ],
            }
            f.write(json.dumps(example) + "\n")


@chz.chz
class Config:
    N: int = 16
    fixed_secret: int = 0
    seed: int = 0
    num_train: int = 1000
    num_test: int = 200
    output_dir: str = "/tmp/tinker-examples/memory_rl/sft_data"


def main() -> None:
    cfg = chz.entrypoint(Config)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    generate_sft_data(cfg.N, cfg.fixed_secret, cfg.num_train, train_path)
    generate_sft_data(cfg.N, cfg.fixed_secret, cfg.num_test, test_path)

    logger.info(f"Secret: {cfg.fixed_secret}")
    logger.info(f"Wrote {cfg.num_train} train examples to {train_path}")
    logger.info(f"Wrote {cfg.num_test} test examples to {test_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
