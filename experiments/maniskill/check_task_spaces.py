import argparse
from collections.abc import Mapping

import gymnasium as gym
import numpy as np


DEFAULT_TASKS = [
    "PushCube-v1",
    "PickCube-v1",
    "PegInsertionSide-v1",
    "PushT-v1",
    "UnitreeG1PlaceAppleInBowl-v1",
    "AnymalC-Reach-v1",
    "OpenCabinetDrawer-v1",
]


def flatten_obs_keys(obs, prefix=""):
    rows = []
    if isinstance(obs, Mapping):
        for key, value in obs.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_obs_keys(value, name))
        return rows

    arr = np.asarray(obs)
    rows.append((prefix, tuple(arr.shape), int(arr.size), str(arr.dtype)))
    return rows


def space_summary(space):
    shape = getattr(space, "shape", None)
    dtype = getattr(space, "dtype", None)
    if shape is not None:
        return f"{space.__class__.__name__}(shape={shape}, dtype={dtype})"
    return repr(space)


def make_env(env_id, obs_mode, control_mode):
    kwargs = {"obs_mode": obs_mode}
    if control_mode is not None:
        kwargs["control_mode"] = control_mode
    return gym.make(env_id, **kwargs)


def inspect_env(env_id, obs_mode, control_mode):
    try:
        env = make_env(env_id, obs_mode=obs_mode, control_mode=control_mode)
    except Exception as exc:
        print(f"\n[{env_id}] create failed: {type(exc).__name__}: {exc}")
        return

    try:
        obs, _ = env.reset(seed=0)
        print(f"\n[{env_id}]")
        print(f"  obs_mode: {obs_mode}")
        print(f"  control_mode: {getattr(env.unwrapped, 'control_mode', control_mode)}")
        print(f"  observation_space: {space_summary(env.observation_space)}")
        print(f"  action_space:      {space_summary(env.action_space)}")

        action_shape = getattr(env.action_space, "shape", None)
        action_dim = int(np.prod(action_shape)) if action_shape is not None else None
        print(f"  action_dim:        {action_dim}")

        rows = flatten_obs_keys(obs)
        total = sum(size for _, _, size, _ in rows)
        print(f"  observed_dim:      {total}")
        for key, shape, size, dtype in rows:
            label = key if key else "<flat>"
            print(f"    {label:40s} shape={shape!s:18s} size={size:<5d} dtype={dtype}")
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument(
        "--control-mode",
        default=None,
        help="Optional ManiSkill control_mode. Omit to use each task's default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import mani_skill.envs  # noqa: F401 - register ManiSkill environments

    args = parse_args()
    for task in args.tasks:
        inspect_env(task, obs_mode=args.obs_mode, control_mode=args.control_mode)
