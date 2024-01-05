from __future__ import annotations

from itertools import cycle
from pathlib import Path
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple

import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Triplet(NamedTuple):
    illustration: torch.Tensor
    lineart: torch.Tensor
    palette: torch.Tensor


@torch.jit.script
def post_lineart(x: torch.Tensor) -> torch.Tensor: return (x * 255).clip(0, 255).permute(1, 2, 0)
@torch.jit.script
def post_illustration(x: torch.Tensor) -> torch.Tensor: return (127.5 + 127.5 * x).clip(0, 255).permute(1, 2, 0)
@torch.jit.script
def pre_lineart(x: torch.Tensor) -> torch.Tensor: return 1.0 - x / 255
@torch.jit.script
def pre_illustration(x: torch.Tensor) -> torch.Tensor: return 2.0 * x / 255 - 1.0


class PaintsTorchDataset(Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.paths = list(path.glob("*.npy.gz"))
        
        def load(path: Path) -> np.ndarray:
            with gzip.open(path, "rb") as f:
                return np.load(f)
        self.buffers = list(map(load, tqdm(self.paths, desc="Loading")))

    def __len__(self) -> int: return len(self.buffers)
    def __getitem__(self, idx: int) -> Triplet:
        buffer = self.buffers[idx]
        H, W, C = buffer.shape
        y = np.random.randint(0, H - 512 + 1)
        x = np.random.randint(0, W - 512 + 1)
        tensor = torch.from_numpy(buffer).permute(2, 0, 1)
        tensor = tensor[:, y:y + 512, x:x + 512].float()
        return Triplet(
            pre_illustration(tensor[:3]),
            pre_lineart(tensor[3 + np.random.randint(0, 3), None]),
            pre_illustration(tensor[6:]),
        )
    
class ConvBlock(nn.Sequential):
    def __init__(self, ic: int, io: int, stride: int = 1, bn: bool = True, act: bool = True) -> None:
        layers = [nn.Conv2d(ic, io, 3, padding=1, stride=stride, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(io))
        if act: layers.append(nn.SiLU())
        super().__init__(*layers)


class UNetLevel(nn.Module):
    def __init__(self, ic: int, io: int, bottleneck: nn.Module, bn: bool = True, act: bool = True) -> None:
        super().__init__()
        self.down = ConvBlock(ic, io, 2)
        self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), ConvBlock(io, ic))
        self.bottleneck = bottleneck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.up(self.bottleneck(x) + x)

class UNet(nn.Sequential):
    def __init__(self) -> None:
        input = ConvBlock(1, 16)
        output = nn.Sequential(nn.Conv2d(16, 3, kernel_size=3, padding=1), nn.Tanh())
        super().__init__(input, UNetLevel(16, 32, UNetLevel(32, 64, UNetLevel(64, 128, UNetLevel(128, 256, UNetLevel(256, 512, lambda x: x))))), output)

    def fit(self, loader: DataLoader, optimizer: Optimizer, scheduler: LRScheduler, steps: int, accumulate: int, device: torch.device, cbk: callable) -> list[float]:
        history = []
        batches = cycle(iter(loader))
        pbar = tqdm(range(steps))
        for step in pbar:
            loss = 0.0
            for _ in range(accumulate):
                illustration, lineart, palette = map(lambda t: t.to(device), next(batches))
                loss += F.l1_loss(self(lineart), illustration, reduction="none").sum([1, 2, 3]).mean([0])
            loss = loss / accumulate
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{loss.item():.2e}", lr=f"{scheduler.get_last_lr()[-1]:.2e}")
            history.append(loss.item())
            if step % 100 == 0: cbk()
        return history


if __name__ == "__main__":
    from PIL import Image
    from tqdm import tqdm
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    import matplotlib.pyplot as plt


    device = torch.device("cpu")
    steps = 500
    batch_size = 4
    accumulate = 128 // batch_size
    lr = 1e-3

    train_set = PaintsTorchDataset(Path("prepared/train"))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = UNet().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1)

    test_lineart = train_set[0].lineart[None].to(device)
    @torch.inference_mode()
    def cbk() -> None:
        Image.fromarray(post_lineart(test_lineart[0])[..., 0].cpu().numpy().astype(np.uint8)).save("lineart.png")
        Image.fromarray(post_illustration(model(test_lineart)[0]).cpu().numpy().astype(np.uint8)).save("generation.png")
    
    history = model.fit(train_loader, optimizer, scheduler, steps, accumulate, device, cbk)
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.title("Training Loss over Time")
    plt.xlabel("step")
    plt.ylabel("mse")
    plt.savefig("paintstorch_ae.png")
    
    device = torch.device('cpu')
    model = model.to(device)
    torch.save(model.state_dict(), 'paintstorch_ae.pt')