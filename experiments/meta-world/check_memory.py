import argparse
from pathlib import Path

import torch


def safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_model_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path = path / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find model.pt: {path}")
    return path


def tensor_shape(value):
    return tuple(value.shape) if hasattr(value, "shape") else None


def main():
    parser = argparse.ArgumentParser(
        description="Inspect whether a saved RARL model.pt contains memory."
    )
    parser.add_argument("path", type=Path, help="Path to a saved model directory or model.pt")
    parser.add_argument("--map-location", default="cpu")
    args = parser.parse_args()

    model_path = resolve_model_path(args.path)
    model = safe_torch_load(model_path, map_location=args.map_location)
    memory = getattr(model, "memory", None)

    print(f"model_path: {model_path}")
    print(f"model_type: {type(model).__name__}")
    print(f"has_memory: {memory is not None}")

    if memory is None:
        return

    print(f"memory_type: {type(memory).__name__}")
    print(f"memory_capacity: {getattr(memory, 'capacity', None)}")
    print(f"memory_size: {len(memory) if hasattr(memory, '__len__') else getattr(memory, 'size', None)}")
    print(f"memory_ptr: {getattr(memory, 'ptr', None)}")
    print(f"states_shape: {tensor_shape(getattr(memory, 'states', None))}")
    print(f"actions_shape: {tensor_shape(getattr(memory, 'actions', None))}")
    print(f"has_rewards: {hasattr(memory, 'rewards')}")

    if hasattr(memory, "rewards"):
        print(f"rewards_shape: {tensor_shape(memory.rewards)}")
        print(f"num_replacements: {getattr(memory, 'num_replacements', None)}")
        print(f"num_skips: {getattr(memory, 'num_skips', None)}")


if __name__ == "__main__":
    main()
