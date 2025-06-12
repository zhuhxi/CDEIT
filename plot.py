import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def save_comparison_images(preds, gts, ckpt_dir, num_samples=5):
    # 创建目录，如果目录不存在
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # 创建图像
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 15))

    for i in range(num_samples):
        ax1, ax2 = axes[i]

        # 选择第 i 个样本
        pred_img = preds[i]
        gt_img = gts[i]

        # 画出预测图像
        ax1.imshow(pred_img.transpose(1, 2, 0))  # 假设数据格式是 CxHxW，转为 HxWxC
        ax1.set_title(f"Predicted {i+1}")
        ax1.axis('off')

        # 画出真实图像
        ax2.imshow(gt_img.transpose(1, 2, 0))  # 假设数据格式是 CxHxW，转为 HxWxC
        ax2.set_title(f"Ground Truth {i+1}")
        ax2.axis('off')

    plt.tight_layout()

    # 保存图像到指定的目录
    image_path = os.path.join(ckpt_dir, "comparison_images.png")
    plt.savefig(image_path)
    plt.close()  # 关闭图像，以避免内存溢出

    print(f"📁 图像已保存到 {image_path}")

def main():
    # 使用argparse配置命令行参数
    parser = argparse.ArgumentParser(description="Comparison image generation for model predictions")
    parser.add_argument('--ckpt_dir', type=str, required=True, help="路径到 checkpoints 目录")
    parser.add_argument('--num_samples', type=int, default=5, help="要生成的样本数量")

    args = parser.parse_args()

    # 加载 test_results.mat 文件
    test_results_path = os.path.join(args.ckpt_dir, 'test_results.mat')
    
    if not os.path.exists(test_results_path):
        print(f"❌ 错误: {test_results_path} 文件不存在!")
        return

    test_results = sio.loadmat(test_results_path)

    preds = test_results['preds']
    gts = test_results['gts']

    # 调用函数保存图像
    save_comparison_images(preds, gts, args.ckpt_dir, num_samples=args.num_samples)

if __name__ == "__main__":
    main()
