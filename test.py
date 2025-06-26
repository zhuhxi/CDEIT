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


# 手动计算 SSIM
def calculate_ssim(img1, img2, C1=1e-4, C2=9e-4):
    """
    计算结构相似性指数（SSIM）
    :param img1: 第一张图片（Numpy 数组）
    :param img2: 第二张图片（Numpy 数组）
    :param C1: 常数 C1，默认 1e-4
    :param C2: 常数 C2，默认 9e-4
    :return: SSIM 值
    """
    # 计算均值
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # 计算方差
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    
    # 计算协方差
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    
    # 计算结构相似性指数
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    
    ssim = numerator / denominator
    return ssim


# 手动计算皮尔逊相关系数 (CC)
def calculate_cc(img1, img2):
    """
    计算皮尔逊相关系数（CC）
    :param img1: 第一张图片（Numpy 数组）
    :param img2: 第二张图片（Numpy 数组）
    :return: CC 值
    """
    # 扁平化两个图像
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    # 计算均值
    mean1 = np.mean(img1_flat)
    mean2 = np.mean(img2_flat)
    
    # 计算相关系数
    numerator = np.sum((img1_flat - mean1) * (img2_flat - mean2))
    denominator = np.sqrt(np.sum((img1_flat - mean1) ** 2) * np.sum((img2_flat - mean2) ** 2))
    
    cc = numerator / denominator
    return cc


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
    ssim_values = []
    cc_values = []
    preds = []
    gts = []

    with torch.no_grad():
        loop = tqdm(test_loader, desc="[Test]", ncols=100)

        for ys, y_st, xs in loop:
            ys = ys.to(device)
            xs = xs.to(device)

            if "CDEIT" in args.model_class:
                pred, _ = model(xs, ys)
            else:
                pred = model(ys)
            loss = nn.MSELoss()(pred, xs)

            test_loss += loss.item()
            RMSE.append((xs - pred).square().mean().sqrt().item())

            preds.append(pred.cpu())
            gts.append(xs.cpu())

            # 计算 SSIM 和 CC
            # 需要转换为 numpy 数组，且应在去除 batch 维度之后进行计算
            pred_np = pred.cpu().numpy()
            xs_np = xs.cpu().numpy()

            # 计算每个样本的 SSIM 和 CC
            for i in range(pred_np.shape[0]):
                ssim_val = calculate_ssim(xs_np[i], pred_np[i])
                ssim_values.append(ssim_val)

                cc_val = calculate_cc(xs_np[i], pred_np[i])
                cc_values.append(cc_val)

            loop.set_postfix(loss=loss.item())

    avg_test_loss = test_loss / len(test_loader)
    avg_rmse = np.mean(RMSE)
    avg_ssim = np.mean(ssim_values)
    avg_cc = np.mean(cc_values)

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)

    # 计算PSNR（可选）
    max_val = torch.max(gts)
    psnr = 10 * torch.log10(max_val**2 / ((gts - preds).square().mean([1, 2, 3]) + 1e-12))

    # 输出结果
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print(f"Average PSNR: {psnr.mean():.6f}")
    print(f"Average SSIM: {avg_ssim:.6f}")
    print(f"Average CC: {avg_cc:.6f}")

    # 保存结果
    sio.savemat(os.path.join(ckpt_dir, "test_results.mat"), {'preds': preds.numpy(), 'gts': gts.numpy()})
    np.savez(os.path.join(ckpt_dir, "rmse_history.npz"), rmse=RMSE)

    print(f"📁 测试结果已保存到 {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model on EIT data.")
    parser.add_argument("--model-name", type=str, default="eitnet", help="Module name in models/ [eitnet sadb_net cnneim ecnet]")
    parser.add_argument("--model-class", type=str, default="EITNet", help="Class name of model [EITNet SADB_Net CNN_EIM EcNet]")
    parser.add_argument("--ckpt-name", type=str, default="EITNet", help="Checkpoint subdirectory name [EITNet SADB_Net CNN_EIM EcNet]")
    parser.add_argument("--test-path", type=str, default="/home/zhx/word/work/CDEIT/data/test/", help="Test data path")

    args = parser.parse_args()
    main(args)
