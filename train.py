import os
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 禁用显示，使用Agg后端生成图像
import numpy as np
from torch.utils.data import DataLoader
from dataset import EITdataset
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

    # 初始化模型
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ModelClass().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_valid_loss = float("inf")

    # 加载数据
    train_dataset = EITdataset(args.train_path, modelname='DEIT')
    valid_dataset = EITdataset(args.valid_path, modelname='DEIT')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    # 保存目录
    ckpt_dir = os.path.join("checkpoints", args.ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_losses = []
    valid_losses = []

    if args.model_class == 'CDEIT':
        from diffusion import create_diffusion
        diffusion = create_diffusion(timestep_respacing="") 

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{args.epochs}", ncols=100)

        for ys, y_st, xs in loop:
            xs = xs.to(device)
            ys = ys.to(device)

            if args.model_class == "CDEIT":
                y_st = y_st.to(device)
                t = torch.randint(0, diffusion.num_timesteps, (xs.shape[0],), device=device)
                model_kwargs = dict(y=ys, y_st=y_st)
                loss_dict = diffusion.training_losses(model, xs, t, model_kwargs)
                loss = loss_dict["loss"]  # *args.global_batch_size
            else:
                pred = model(ys)
                loss = criterion(pred, xs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for ys, y_st, xs in valid_loader:
                xs = xs.to(device)
                ys = ys.to(device)

                if args.model_class == "CDEIT":
                    y_st = y_st.to(device)
                    t = torch.randint(0, diffusion.num_timesteps, (xs.shape[0],), device=device)
                    model_kwargs = dict(y=ys, y_st=y_st)
                    loss_dict = diffusion.training_losses(model, xs, t, model_kwargs)
                    loss = loss_dict["loss"]  # *args.global_batch_size
                else:
                    pred = model(ys)
                    loss = criterion(pred, xs)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))
            print(f"✅ Best model saved at epoch {epoch+1}")

    # 保存损失记录
    np.savez(os.path.join(ckpt_dir, "loss_history.npz"),
             train_losses=train_losses,
             valid_losses=valid_losses)

    # 绘制曲线
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "loss_curve.png"))
    plt.close()

    print(f"📁 所有文件已保存到 {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on EIT data.")
    parser.add_argument("--model-name", type=str, default="eitnet", help="Module name in models/ [eitnet sadb_net cnneim ecnet cdeit]")
    parser.add_argument("--model-class", type=str, default="EITNet", help="Class name of model [EITNet SADB_Net CNN_EIM EcNet CDEIT]")
    parser.add_argument("--ckpt-name", type=str, default="EITNet", help="Checkpoint subdirectory name [EITNet SADB_Net CNN_EIM EcNet CDEIT]")
    parser.add_argument("--train-path", type=str, default="/home/zhx/word/work/CDEIT/data/train/", help="Training data path")
    parser.add_argument("--valid-path", type=str, default="/home/zhx/word/work/CDEIT/data/valid/", help="Validation data path")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")

    args = parser.parse_args()
    main(args)
