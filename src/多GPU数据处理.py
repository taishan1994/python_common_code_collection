"""

CUDA_VISIBLE_DEVICES=3,4 torchrun --nnodes 1 --nproc_per_node 2 --master_port 30111 \
    test_torchrun.py
"""

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        try:
            return self.data[index]
        except Exception as e:
            print_with_rank(f"ERROR:{e}\n<{index}> 读取失败, 跳过...")
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.data)


def init_dist():
    try:
        dist.init_process_group()
    except:
        ...


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def print_rank_0(message, rank=None):
    if rank is None:
        rank = get_rank()

    if rank == 0:
        print(message, flush=True)


def print_with_rank(*args, rank=None, **kwargs):
    if rank is None:
        rank = get_rank()

    print(f"rank={rank}: ", *args, **kwargs)


if __name__ == "__main__":
    init_dist()
    rank = get_rank()
    world_size = get_world_size()
    print_rank_0(f"\nworld size: {world_size}", rank)

    device = f"cuda:{rank}"

    data = list(range(1000))
    origin_dataset = MyDataset(data)
    sampler = DistributedSampler(origin_dataset, num_replicas=world_size, rank=rank)
    collate_fn = None

    origin_dataset_loader = DataLoader(
        origin_dataset,
        batch_size=10,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )


    for idx, batch in enumerate(tqdm(origin_dataset_loader, desc=f"{rank}")):
        print(batch)
