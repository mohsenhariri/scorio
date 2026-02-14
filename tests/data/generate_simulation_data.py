from pathlib import Path

import numpy as np


def _build_simulation_dataset(seed: int, trials: int) -> dict[str, np.ndarray]:
    tasks = ["aime24", "aime25", "hmmt_feb_2025", "brumo_2025"]
    rng = np.random.default_rng(seed)

    data: dict[str, np.ndarray] = {}
    for task in tasks:
        data[task] = rng.integers(0, 2, size=(20, 30, trials), dtype=np.int8)
    data["combined"] = np.concatenate([data[task] for task in tasks], axis=1)
    return data


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    top_p = _build_simulation_dataset(seed=20260214, trials=80)
    greedy = _build_simulation_dataset(seed=20260215, trials=1)

    np.savez_compressed(out_dir / "R_top_p.npz", **top_p)
    np.savez_compressed(out_dir / "R_greedy.npz", **greedy)


if __name__ == "__main__":
    main()
