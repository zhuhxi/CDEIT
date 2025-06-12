import os
import argparse
import importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EITdataset
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import random

def init_seed(seed=2019, reproducibility=True) -> None:
    r"""初始化随机种子，使得numpy、torch、cuda和cudnn中的随机函数结果可复现
    
    Args:
        seed (int): 随机种子
        reproducibility (bool): 是否要求复现性
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def main(args):
    # 设置固定种子
    init_seed(seed=2019, reproducibility=True)

    # 动态导入模型类
    model_module = importlib.import_module(f"models.{args.model_name}")
    ModelClass = getattr(model_module, args.model_class)

    # 初始化设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass().to(device)

    # 加载测试数据
    test_dataset = EITdataset(args.test_path, modelname='DEIT')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 加载最好的模型
    ckpt_dir = os.path.join("checkpoints", args.ckpt_name)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pt")))
    model.eval()

    # 计算指标
    test_loss = 0.0
    RMSE = []
    preds = []
    gts = []

    with torch.no_grad():
        loop = tqdm(test_loader, desc="[Test]", ncols=100)

        for ys, _, xs in loop:
            ys = ys.to(device)
            xs = xs.to(device)

            pred = model(ys)
            loss = nn.MSELoss()(pred, xs)

            test_loss += loss.item()
            RMSE.append((xs - pred).square().mean().sqrt().item())

            preds.append(pred.cpu())
            gts.append(xs.cpu())

            loop.set_postfix(loss=loss.item())

    avg_test_loss = test_loss / len(test_loader)
    avg_rmse = np.mean(RMSE)

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)

    # 计算PSNR（可选）
    max_val = torch.max(gts)
    psnr = 10 * torch.log10(max_val**2 / ((gts - preds).square().mean([1, 2, 3]) + 1e-12))

    # 输出结果
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print(f"Average PSNR: {psnr.mean():.6f}")

    # 保存结果
    sio.savemat(os.path.join(ckpt_dir, "test_results.mat"), {'preds': preds.numpy(), 'gts': gts.numpy()})
    np.savez(os.path.join(ckpt_dir, "rmse_history.npz"), rmse=RMSE)

    print(f"📁 测试结果已保存到 {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model on EIT data.")
    parser.add_argument("--model-name", type=str, default="eitnet", help="Module name in models/ [eitnet sadb_net]")
    parser.add_argument("--model-class", type=str, default="EITNet", help="Class name of model [EITNet SADB_Net]")
    parser.add_argument("--ckpt-name", type=str, default="Unet", help="Checkpoint subdirectory name [EITNet SADB_Net]")
    parser.add_argument("--test-path", type=str, default="/home/zhx/word/work/CDEIT/data/test/", help="Test data path")

    args = parser.parse_args()
    main(args)
