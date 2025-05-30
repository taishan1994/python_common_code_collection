import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import random


# 1. 初始化分布式环境
def setup_distributed(backend=''):
    """Initialize the distributed environment."""
    if dist.is_available() and dist.is_initialized():
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # GPU to use

    # NCCL is preferred for GPU communication
    # Use 'gloo' for CPU-only distributed training/inference
    if torch.cuda.is_available():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}.")
    else:
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        print(f"Rank {rank}/{world_size} initialized on CPU (gloo backend).")


def cleanup_distributed():
    """Cleanup the distributed environment."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        print(f"Rank {dist.get_rank() if dist.is_initialized() else 'N/A'} cleaned up.")


# 2. 定义简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters needed for x^2

    def forward(self, x):
        return x * x


# 3. 创建自定义数据集
class RandomDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = [torch.tensor([float(i)]) for i in range(num_samples)]
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Generated {num_samples} random samples.")
            # print("Sample data (first 5):", [d.item() for d in self.data[:5]])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


def run_inference(rank, world_size, total_samples=20, batch_size=4):
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Rank {rank} using device: {device}")

    # 创建模型并移动到设备
    model = SimpleModel().to(device)
    # 注意：对于纯推理，如果模型相同且不需要同步梯度，DDP不是必须的。
    # 如果你想用DDP的模式（比如模型状态可能在rank间不同步需要广播），可以取消注释下面这行
    # if torch.cuda.is_available() and world_size > 1:
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # else:
    #     model = nn.parallel.DistributedDataParallel(model) # For CPU DDP with Gloo

    # 创建数据集和分布式采样器
    dataset = RandomDataset(num_samples=total_samples)
    print(dataset.data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model.eval()
    local_predictions = []
    local_inputs = []  # To verify later

    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data.to(device)
            outputs = model(inputs)
            local_predictions.append(outputs)
            local_inputs.append(inputs)  # Store original inputs for verification

        local_predictions = torch.cat(local_predictions).cpu()
        total_predictions = [torch.empty_like(local_predictions) for _ in range(world_size)]
        dist.all_gather(total_predictions, local_predictions)

        print(local_predictions)

        local_inputs = torch.cat(local_inputs).cpu()
        total_inputs = [torch.empty_like(local_inputs) for _ in range(world_size)]
        dist.all_gather(total_inputs, local_inputs)

        if rank == 0:
            total_predictions = torch.stack(total_predictions, dim=0)
            feat_dim = total_predictions.shape[-1]
            total_predictions = total_predictions.permute(1, 0, 2).reshape(-1, feat_dim)
            total_predictions = total_predictions[: len(dataset)].numpy()
            print(total_predictions)

            total_inputs = torch.stack(total_inputs, dim=0)
            feat_dim = total_inputs.shape[-1]
            total_inputs = total_inputs.permute(1, 0, 2).reshape(-1, feat_dim)
            total_inputs = total_inputs[: len(dataset)].numpy()
            print(total_inputs)

            """
            total_predictions = total_predictions.permute(1, 0, 2).reshape(-1, feat_dim):
                permute(1, 0, 2): 将维度从 (world_size, num_samples_per_rank, feat_dim) 交换为 (num_samples_per_rank, world_size, feat_dim)。
                reshape(-1, feat_dim): 将其展平。这一系列操作的目的是将来自不同 rank 的 embedding 交错合并，从而恢复原始数据集的顺序。
                例如，如果 rank 0 有 [e0_0, e0_1, e0_2]，rank 1 有 [e1_0, e1_1, e1_2]，
                stack 后是 [[e0_0, e0_1, e0_2], [e1_0, e1_1, e1_2]]。
                permute 后是 [[e0_0, e1_0], [e0_1, e1_1], [e0_2, e1_2]] (概念上，实际是3D张量)。
                reshape 后是 [e0_0, e1_0, e0_1, e1_1, e0_2, e1_2, ...]。
                这正是 DistributedSampler (当 shuffle=False) 分配数据的方式：rank 0 得到样本 0, world_size, 2*world_size, ...; rank 1 得到样本 1, world_size+1, ...
            截断到原始数据集长度:
                total_predictions = total_predictions[: len(dataset)].numpy():
                这是一个很好的处理方式，特别是如果 DistributedSampler 为了使每个 rank 处理相同数量的样本而引入了填充样本（例如，sampler 内部可能复制了一些样本以使总数能被 world_size 整除）。通过 [: len(origin_dataset)]，你确保只保存原始数据集对应的有效 embedding。
            """



if __name__ == "__main__":
    # torchrun 会自动设置 RANK, WORLD_SIZE, LOCAL_RANK 等环境变量
    setup_distributed()

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # 参数
    TOTAL_SAMPLES = 23  # 可以改成不能被 world_size * batch_size 整除的数来测试填充
    BATCH_SIZE = 5

    run_inference(rank, world_size, total_samples=TOTAL_SAMPLES, batch_size=BATCH_SIZE)

    cleanup_distributed()
