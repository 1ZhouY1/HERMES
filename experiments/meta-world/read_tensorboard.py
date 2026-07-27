import argparse
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="直接读取 TensorBoard tfevents 并计算 success"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="TensorBoard 日志目录",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="success 对应的 scalar tag，例如 charts/success",
    )

    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="最小时间步，默认 0",
    )

    parser.add_argument(
        "--max_step",
        type=int,
        default=1000000,
        help="最大时间步，例如 1000000",
    )

    parser.add_argument(
        "--list_tags",
        action="store_true",
        help="列出日志中的所有 scalar tags",
    )

    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="保留重复 Step。默认同一 Step 只保留最后写入的一条记录",
    )

    return parser.parse_args()


def load_events(log_dir):
    log_path = Path(log_dir)

    if not log_path.exists():
        raise FileNotFoundError(
            f"日志路径不存在：{log_path.resolve()}"
        )

    accumulator = EventAccumulator(
        str(log_path),
        size_guidance={
            # 读取全部 scalar，不进行 TensorBoard reservoir sampling
            "scalars": 0,
        },
    )

    accumulator.Reload()

    return accumulator


def list_scalar_tags(accumulator):
    scalar_tags = accumulator.Tags().get("scalars", [])

    if not scalar_tags:
        print("没有找到任何 scalar tag。")
        return

    print("可用的 scalar tags：")

    for index, tag in enumerate(scalar_tags, start=1):
        number_of_records = len(
            accumulator.Scalars(tag)
        )

        print(
            f"{index:>3}. {tag} "
            f"(记录数：{number_of_records})"
        )


def calculate_success(
    accumulator,
    tag,
    min_step,
    max_step,
    keep_duplicates,
):
    scalar_tags = accumulator.Tags().get("scalars", [])

    if tag not in scalar_tags:
        print(f"找不到 scalar tag：{tag}\n")
        list_scalar_tags(accumulator)
        raise ValueError("请指定有效的 --tag")

    events = accumulator.Scalars(tag)

    df = pd.DataFrame(
        {
            "Wall time": [
                event.wall_time
                for event in events
            ],
            "Step": [
                event.step
                for event in events
            ],
            "Value": [
                event.value
                for event in events
            ],
        }
    )

    # 限制时间步范围
    df = df[df["Step"] >= min_step].copy()

    if max_step is not None:
        df = df[df["Step"] <= max_step].copy()

    if df.empty:
        print(
            f"在 Step [{min_step}, {max_step}] 范围内"
            "没有找到数据。"
        )
        return

    # 检查重复 Step
    duplicate_rows = df.duplicated(
        subset="Step",
        keep=False,
    )

    duplicate_row_count = int(duplicate_rows.sum())
    duplicate_step_count = int(
        df.loc[duplicate_rows, "Step"].nunique()
    )

    if not keep_duplicates:
        # 同一个 Step 保留 Wall time 最新的一条记录
        df = (
            df.sort_values(["Step", "Wall time"])
            .drop_duplicates(
                subset="Step",
                keep="last",
            )
            .sort_values("Step")
            .reset_index(drop=True)
        )
    else:
        df = df.sort_values(
            ["Step", "Wall time"]
        ).reset_index(drop=True)

    success_sum = df["Value"].sum()
    success_mean = df["Value"].mean()

    print("=" * 60)
    print("TensorBoard success 统计结果")
    print("=" * 60)
    print(f"日志目录：{Path(args.log_dir).resolve()}")
    print(f"Scalar tag：{tag}")
    print(f"时间步范围：{min_step} ～ {max_step}")
    print(f"实际最小 Step：{df['Step'].min()}")
    print(f"实际最大 Step：{df['Step'].max()}")
    print(f"记录数量：{len(df)}")
    print(f"重复 Step 数量：{duplicate_step_count}")
    print(f"重复 Step 相关行数：{duplicate_row_count}")
    print("-" * 60)
    print(f"success 总和：{success_sum}")
    print(f"success 平均值：{success_mean}")
    print("=" * 60)


def main():
    global args
    args = parse_args()

    accumulator = load_events(args.log_dir)

    if args.list_tags:
        list_scalar_tags(accumulator)
        return

    if args.tag is None:
        print("没有指定 --tag。\n")
        list_scalar_tags(accumulator)
        return

    calculate_success(
        accumulator=accumulator,
        tag=args.tag,
        min_step=args.min_step,
        max_step=args.max_step,
        keep_duplicates=args.keep_duplicates,
    )


if __name__ == "__main__":
    main()