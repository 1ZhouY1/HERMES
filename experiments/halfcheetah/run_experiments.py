import subprocess
import argparse
import sys
import os
from pathlib import Path

# 动态导入 tasks.py 中的任务列表和场景字典
try:
    from tasks import tasks as TASKS, SCENARIOS
except ImportError:
    print("Error: Can not import tasks.py. Please make sure it exists and is in the same directory as run_experiments.py.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="HalfCheetah Continual RL Launcher")

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["simple", "sdnet", "finetune", "componet", "prognet", "packnet"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--scenario", 
        type=str, 
        choices=list(SCENARIOS.keys()),
        help="Required for continual learning algorithms (finetune, prognet, etc.)"
    )
    # 仅针对 simple 的可选参数
    parser.add_argument(
        "--task", 
        type=str, 
        default=None,
        help="Optional: Specify a single task to run (ONLY works with algorithm 'simple')"
    )
    parser.add_argument("--save-dir", type=str, default="agents")
    parser.add_argument("--exp-name", type=str, default="SAC")
    parser.add_argument(
        "--simple-exp-names",
        nargs="*",
        default=["SAC", "run_sac", "SAC_hybridRAGRL", "sac_20k"],
        help="Experiment names searched for the first simple previous unit.",
    )
    parser.add_argument("--no-run", default=False, action="store_true")

    return parser.parse_args()


def unique_task_sequence(task_names):
    """Remove the repeated second/fourth passes from a scenario."""
    seen = set()
    sequence = []
    for task_name in task_names:
        if task_name in seen:
            continue
        seen.add(task_name)
        sequence.append(task_name)
    return sequence


def model_dir_name(task_id, model_name, exp_name, seed):
    return f"task_{task_id}__{model_name}__{exp_name}__{seed}"


def checkpoint_exists(model_dir, model_name):
    required_files = {
        "simple": ("model.pt",),
        "finetune": ("model.pt",),
        "componet": ("net_mean.pt",),
        "packnet": ("packnet.pt",),
        "prognet": ("prognet.pt",),
    }
    return any((model_dir / filename).exists() for filename in required_files.get(model_name, ("model.pt",)))


def resolve_previous_model_dir(args, task_id, model_name):
    """Find a previous checkpoint, including legacy simple experiment names."""
    save_dir = Path(args.save_dir).expanduser()
    if model_name in {"simple", "sdnet"}:
        root = save_dir
        exp_names = args.simple_exp_names
    else:
        root = save_dir / args.scenario
        exp_names = [args.exp_name]

    candidates = [
        root / model_dir_name(task_id, model_name, exp_name, args.seed)
        for exp_name in exp_names
    ]
    for candidate in candidates:
        if checkpoint_exists(candidate, model_name):
            return candidate.resolve()

    tried = "\n".join(f"    {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"No previous {model_name} checkpoint found for task {task_id}. Tried:\n{tried}"
    )


if __name__ == "__main__":
    args = parse_args()

    # ── 1. 逻辑分流：获取需要运行的任务序列 ──
    if args.algorithm in ["simple", "sdnet"]:
        is_continual = False
        # 如果是 simple 并且指定了 task，就只跑这个 task
        if args.task:
            if args.task not in TASKS:
                print(f"*** Error: 任务 '{args.task}' 不存在！请检查拼写。")
                sys.exit(1)
            tasks_to_run = [args.task]
        else:
            # 否则照常跑完所有的 simple 任务
            tasks_to_run = TASKS
    else:
        is_continual = True
        # 持续学习模式：强制要求 scenario，并忽略 task 参数
        if not args.scenario:
            print(f"*** Error: Algorithm '{args.algorithm}' requires a --scenario argument.")
            sys.exit(1)
        tasks_to_run = unique_task_sequence(SCENARIOS[args.scenario])
        
        if args.task:
            print(f"*** Warning: --task 参数仅对 simple/sdnet 生效。在 {args.algorithm} 模式下将被忽略，继续执行完整场景序列。")

    # ── 2. 动态计算起始点与权重继承逻辑 ──
    start_idx = 0
    # Finetune 和 CompoNet 不训练 Task 0，直接从 Task 1 开始并继承 Simple 的 Task 0 权重
    if is_continual and args.algorithm not in ["packnet", "prognet"]:
        start_idx = 1

    # 【精简修改】：如果是 simple 则不加任何子目录；否则加上场景名
    def get_run_name(algo, task_id):
        if algo in ["simple", "sdnet"]:
            return model_dir_name(task_id, algo, args.exp_name, args.seed)
        else:
            return f"{args.scenario}/{model_dir_name(task_id, algo, args.exp_name, args.seed)}"

    # ── 3. 执行任务序列 ──
    for i in range(start_idx, len(tasks_to_run)):
        task_name = tasks_to_run[i]
        task_id = TASKS.index(task_name)
        
        # 【精简修改】：只有在跑持续学习时才传 scenario 参数
        scenario_arg = f"--scenario={args.scenario}" if is_continual else ""
        
        # 基础参数映射到 SAC.py
        params = f"--model-type={args.algorithm} --task-id={task_id} --seed={args.seed} --save-dir={args.save_dir} --exp-name={args.exp_name} {scenario_arg}".strip()

        # 处理持续学习的权重继承 (--prev-units)
        if is_continual and i > 0:
            if args.algorithm in ["finetune", "packnet"]:
                # 单一前置模型继承
                prev_task_name = tasks_to_run[i - 1]
                prev_task_id = TASKS.index(prev_task_name)
                # 判断前一个模型是用 simple 跑的还是当前算法跑的
                prev_algo = args.algorithm if i > 1 or args.algorithm in ['packnet', 'prognet'] else "simple"
                prev_dir = resolve_previous_model_dir(args, prev_task_id, prev_algo)
                params += f" --prev-units {prev_dir}"
                
            elif args.algorithm in ["componet", "prognet"]:
                # 多前置模型继承
                params += " --prev-units"
                for j in range(i):
                    past_task_name = tasks_to_run[j]
                    past_task_id = TASKS.index(past_task_name)
                    past_algo = args.algorithm if j > 0 or args.algorithm in ['packnet', 'prognet'] else "simple"
                    prev_dir = resolve_previous_model_dir(args, past_task_id, past_algo)
                    params += f" {prev_dir}"

        # 启动实验进程
        cmd = f"python run_sac.py {params}"
        if is_continual and args.algorithm not in ["packnet", "prognet"]:
            # The first scenario task is represented by the existing simple
            # model, so only the remaining tasks train CompoNet/Finetune.
            progress = (
                f"Scenario task {i + 1}/{len(tasks_to_run)} | "
                f"{args.algorithm.upper()} task {i}/{len(tasks_to_run) - 1}"
            )
        else:
            progress = f"Task {i + 1}/{len(tasks_to_run)}"
        print(f"\n[{args.algorithm.upper()} | {progress} | {task_name}]")
        print(cmd)

        if not args.no_run:
            res = subprocess.run(cmd.split(" "))
            if res.returncode != 0:
                print(f"*** Process returned code {res.returncode}. Stopping on error.")
                sys.exit(1)
