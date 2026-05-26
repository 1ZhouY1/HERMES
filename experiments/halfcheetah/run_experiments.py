import subprocess
import argparse
import sys
import os

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
    parser.add_argument("--no-run", default=False, action="store_true")

    return parser.parse_args()


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
        tasks_to_run = SCENARIOS[args.scenario]
        
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
            return f"task_{task_id}__{algo}__SAC__{args.seed}"
        else:
            return f"{args.scenario}/task_{task_id}__{algo}__SAC__{args.seed}"

    # ── 3. 执行任务序列 ──
    for i in range(start_idx, len(tasks_to_run)):
        task_name = tasks_to_run[i]
        task_id = TASKS.index(task_name)
        
        # 【精简修改】：只有在跑持续学习时才传 scenario 参数
        scenario_arg = f"--scenario={args.scenario}" if is_continual else ""
        
        # 基础参数映射到 SAC.py
        params = f"--model-type={args.algorithm} --task-id={task_id} --seed={args.seed} --save-dir={args.save_dir} --exp-name=SAC {scenario_arg}".strip()

        # 处理持续学习的权重继承 (--prev-units)
        if is_continual and i > 0:
            if args.algorithm in ["finetune", "packnet"]:
                # 单一前置模型继承
                prev_task_name = tasks_to_run[i - 1]
                prev_task_id = TASKS.index(prev_task_name)
                # 判断前一个模型是用 simple 跑的还是当前算法跑的
                prev_algo = args.algorithm if i > 1 or args.algorithm in ['packnet', 'prognet'] else "simple"
                params += f" --prev-units {args.save_dir}/{get_run_name(prev_algo, prev_task_id)}"
                
            elif args.algorithm in ["componet", "prognet"]:
                # 多前置模型继承
                params += " --prev-units"
                for j in range(i):
                    past_task_name = tasks_to_run[j]
                    past_task_id = TASKS.index(past_task_name)
                    past_algo = args.algorithm if j > 0 or args.algorithm in ['packnet', 'prognet'] else "simple"
                    params += f" {args.save_dir}/{get_run_name(past_algo, past_task_id)}"

        # 启动实验进程
        cmd = f"python run_sac.py {params}"
        print(f"\n[{args.algorithm.upper()} | Task {i}/{len(tasks_to_run)-1} | {task_name}]")
        print(cmd)

        if not args.no_run:
            res = subprocess.run(cmd.split(" "))
            if res.returncode != 0:
                print(f"*** Process returned code {res.returncode}. Stopping on error.")
                sys.exit(1)