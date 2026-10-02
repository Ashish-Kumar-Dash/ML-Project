"""Generate synthetic CMU-MOSI formatted pickle files for testing and fast iteration.

The pickle layout exactly matches the canonical MulT / MPLMM MOSI pickle structure:
{
    'train': {'text': (N_train, L, d_l), 'audio': (N_train, A, d_a),
              'vision': (N_train, V, d_v), 'labels': (N_train, 1, 1)},
    'valid': {...},
    'test':  {...}
}

Configurable sequence lengths (L, A, V) allow rapid creation of mock rungs
for testing length-invariance without needing multi-gigabyte raw video extraction.

Usage:
    python tools/make_fake_mosi.py --output data/fake_mosi_l50.pkl
    python tools/make_fake_mosi.py --output data/fake_mosi_l100.pkl --len_a 100 --len_v 100
"""

import argparse
from pathlib import Path
import pickle
import numpy as np


def generate_fake_mosi_dict(
    len_l: int = 50,
    len_a: int = 50,
    len_v: int = 50,
    dim_l: int = 300,
    dim_a: int = 5,
    dim_v: int = 20,
    n_train: int = 32,
    n_valid: int = 16,
    n_test: int = 16,
    seed: int = 42,
) -> dict:
    """Generate in-memory synthetic MOSI dataset dictionary.

    Args:
        len_l: Sequence length for text/language (L)
        len_a: Sequence length for audio (A)
        len_v: Sequence length for vision (V)
        dim_l: Feature dimension for text (default 300)
        dim_a: Feature dimension for audio (default 5)
        dim_v: Feature dimension for vision (default 20)
        n_train: Number of samples in train split
        n_valid: Number of samples in valid split
        n_test: Number of samples in test split
        seed: Random seed for reproducibility

    Returns:
        Dictionary structured as {split: {vision, text, audio, labels}}
    """
    rng = np.random.RandomState(seed)

    splits = {
        "train": n_train,
        "valid": n_valid,
        "test": n_test,
    }

    dataset = {}
    for split_name, n_samples in splits.items():
        dataset[split_name] = {
            "text": rng.randn(n_samples, len_l, dim_l).astype(np.float32),
            "audio": rng.randn(n_samples, len_a, dim_a).astype(np.float32),
            "vision": rng.randn(n_samples, len_v, dim_v).astype(np.float32),
            # Labels in range [-3.0, 3.0] with shape (N, 1, 1)
            "labels": rng.uniform(-3.0, 3.0, size=(n_samples, 1, 1)).astype(np.float32),
        }

    return dataset


def save_fake_mosi_pickle(
    output_path: str,
    len_l: int = 50,
    len_a: int = 50,
    len_v: int = 50,
    dim_l: int = 300,
    dim_a: int = 5,
    dim_v: int = 20,
    n_train: int = 32,
    n_valid: int = 16,
    n_test: int = 16,
    seed: int = 42,
) -> Path:
    """Generate and serialize synthetic MOSI dataset to disk."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    data = generate_fake_mosi_dict(
        len_l=len_l,
        len_a=len_a,
        len_v=len_v,
        dim_l=dim_l,
        dim_a=dim_a,
        dim_v=dim_v,
        n_train=n_train,
        n_valid=n_valid,
        n_test=n_test,
        seed=seed,
    )

    with open(out, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out


def main():
    parser = argparse.ArgumentParser(description="Create synthetic MOSI pickle file.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/fake_mosi.pkl",
        help="Target output pickle path (default: data/fake_mosi.pkl)",
    )
    parser.add_argument("--len_l", type=int, default=50, help="Text sequence length L")
    parser.add_argument("--len_a", type=int, default=50, help="Audio sequence length A")
    parser.add_argument("--len_v", type=int, default=50, help="Vision sequence length V")
    parser.add_argument("--dim_l", type=int, default=300, help="Text feature dimension")
    parser.add_argument("--dim_a", type=int, default=5, help="Audio feature dimension")
    parser.add_argument("--dim_v", type=int, default=20, help="Vision feature dimension")
    parser.add_argument("--n_train", type=int, default=32, help="Train samples count")
    parser.add_argument("--n_valid", type=int, default=16, help="Valid samples count")
    parser.add_argument("--n_test", type=int, default=16, help="Test samples count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    saved_path = save_fake_mosi_pickle(
        output_path=args.output,
        len_l=args.len_l,
        len_a=args.len_a,
        len_v=args.len_v,
        dim_l=args.dim_l,
        dim_a=args.dim_a,
        dim_v=args.dim_v,
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_test=args.n_test,
        seed=args.seed,
    )

    print(f"Fake MOSI pickle successfully written to: {saved_path}")
    print(f"Lengths: L={args.len_l}, A={args.len_a}, V={args.len_v}")
    print(f"Dims:    d_l={args.dim_l}, d_a={args.dim_a}, d_v={args.dim_v}")
    print(f"Splits:  train={args.n_train}, valid={args.n_valid}, test={args.n_test}")


if __name__ == "__main__":
    main()
