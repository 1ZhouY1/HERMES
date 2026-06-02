import random

# 生成一个 0.0 到 1.0 之间的随机浮点数
random_float = random.random()
print(f"随机浮点数: {random_float}")

# 生成一个指定范围内的随机整数（例如 1 到 100，包含 1 和 100）
random_int = random.randint(1, 1024)
print(f"随机整数 (1-100): {random_int}")

