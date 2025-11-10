import time
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms

def test_dataloader_speed():
    # ==== 数据预处理 ====
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])

    # ==== 加载数据集 ====
    dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)

    # ==== 检测CPU核心数 ====
    cpu_cores = os.cpu_count()
    print(f"检测到 CPU 核心数：{cpu_cores}")

    # ==== 测试不同num_workers ====
    worker_list = [0, 1, 2, 4, 8, 12, 16]
    worker_list = [n for n in worker_list if n <= cpu_cores]

    batch_size = 64
    time_list = []

    print("\n开始测试不同 num_workers 的加载速度...\n")

    for num in worker_list:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num)
        start = time.time()
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= 100:  # 仅取前100个batch测试速度
                break
        elapsed = time.time() - start
        time_list.append(elapsed)
        print(f"num_workers={num:<2} | 加载 100 个 batch 用时: {elapsed:.2f} 秒")

    print("\n✅ 测试完成，请观察哪个 num_workers 的时间最短。")

    # ==== 自动分析结果 ====
    best_time = min(time_list)
    best_worker = worker_list[np.argmin(time_list)]
    base_time = time_list[0]
    speedup = (base_time - best_time) / base_time * 100

    print("\n========= 自动分析结果 =========")
    print(f"最佳 num_workers: {best_worker}")
    print(f"最短时间: {best_time:.2f} 秒")
    print(f"单线程基准时间: {base_time:.2f} 秒")
    print(f"相对提升: {speedup:.1f}%")

    # ==== 推荐决策 ====
    if speedup > 10 and best_worker != 0:
        print(f"✅ 建议设置 num_workers = {best_worker} （性能提升显著）")
    else:
        print("⚙️  建议设置 num_workers = 0 （单线程更快且更稳定）")

if __name__ == '__main__':
    test_dataloader_speed()
