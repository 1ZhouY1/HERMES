#!/bin/bash

# 定义 tmux 会话的名称
SESSION_NAME="sac_experiments"

# 1. 创建一个后台运行的 tmux 会话 (-d)，将会话命名为 sac_experiments，并将第一个窗口命名为 task_0
tmux new-session -d -s $SESSION_NAME -n "task_0"

# 2. 向第 0 个窗口发送 task-id 0 的命令，并加上 C-m (相当于按下回车键执行)
tmux send-keys -t $SESSION_NAME:0 "conda activate meta-world && python run.py --algorithm sdnet --start_mode 0 --seed 42" C-m

# 3. 循环创建 task-id 1 到 9 的任务
for i in {1..9}
do
    # 在当前会话中创建一个新窗口，命名为 task_$i
    tmux new-window -t $SESSION_NAME -n "task_$i"
    
    # 向刚刚创建的窗口发送对应的 python 命令并执行
    tmux send-keys -t $SESSION_NAME:$i "conda activate meta-world && python run.py --algorithm sdnet --start_mode $i --seed 42" C-m
done

echo "所有 10 个任务已在 tmux 会话 '$SESSION_NAME' 中启动！"

# 4. 自动接入该 tmux 会话
tmux attach-session -t $SESSION_NAME
