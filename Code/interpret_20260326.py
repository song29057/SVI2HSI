import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from PIL import Image
import pandas as pd


# ==========================================
# 1. 模型结构定义 (MobileHSINet)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileResidualBlock(nn.Module):
    def __init__(self, channels):
        super(MobileResidualBlock, self).__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gelu = nn.GELU()
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        identity = x
        out = self.dw_conv(x)
        out = self.gelu(out)
        out = self.pw_conv(out)
        out = self.ca(out)
        return identity + out


class SpectralFeatureExtractor(nn.Module):
    def __init__(self, in_ch=549, compressed_ch=64, feat_ch=64, frames=183):
        super(SpectralFeatureExtractor, self).__init__()
        self.frames = frames
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5, padding=2, groups=3, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.compress_conv = nn.Conv2d(in_ch, compressed_ch, kernel_size=1, bias=True)
        self.spatial_conv = nn.Conv2d(compressed_ch, feat_ch, kernel_size=3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = y.view(b, 3, self.frames)
        y = self.eca_conv(y)
        y = y.view(b, c, 1, 1)
        weight = self.sigmoid(y)
        x = x * weight
        x = self.compress_conv(x)
        x = self.gelu(x)
        x = self.spatial_conv(x)
        return self.gelu(x)


class MobileHSINet(nn.Module):
    def __init__(self, in_ch=549, compressed_ch=64, feat_ch=64, num_blocks=4, out_ch=284):
        super(MobileHSINet, self).__init__()
        self.stem = SpectralFeatureExtractor(in_ch=in_ch, compressed_ch=compressed_ch, feat_ch=feat_ch)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(MobileResidualBlock(feat_ch))
        self.body = nn.Sequential(*blocks)
        self.final_conv = nn.Sequential(
            nn.Conv2d(feat_ch, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, _, H, W = x.size()
        target_size = (int(H * 1.25), int(W * 1.25))
        x_up = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        pad_size = 8
        x_padded = F.pad(x_up, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
        feat = self.stem(x_padded)
        out = self.body(feat) + feat
        hsi_out_padded = self.final_conv(out)
        hsi_out = hsi_out_padded[:, :, pad_size:-pad_size, pad_size:-pad_size]
        return hsi_out


# ==========================================
# 2. 数据加载器
# ==========================================
class SVI2HSIDatasetAnalysis(Dataset):
    def __init__(self, svi_dir, hsi_dir, sample_indices):
        self.svi_dir = svi_dir
        self.hsi_dir = hsi_dir
        self.indices = sample_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        svi_path = os.path.join(self.svi_dir, f"{file_idx}.npy")

        try:
            svi_data = np.load(svi_path).astype(np.float32)
            svi_data = np.transpose(svi_data, (2, 0, 1))
        except Exception as e:
            raise RuntimeError(f"读取文件失败: {file_idx}.npy. 详细错误: {e}")

        return torch.from_numpy(np.ascontiguousarray(svi_data)), file_idx


# ==========================================
# 3. 核心改进：并行计算四种梯度矩阵
# ==========================================
def compute_global_correlation_matrix(model, test_dataset, results_dir="./", device='cuda'):
    num_samples = len(test_dataset)
    print(f"\n[Analysis] 正在计算测试集全局平均梯度归因矩阵 (共包含 {num_samples} 个样本)...")
    model.eval()

    # 初始化四个全局矩阵
    global_corr_abs = torch.zeros((549, 284), device=device)
    global_corr_raw = torch.zeros((549, 284), device=device)
    global_corr_pos = torch.zeros((549, 284), device=device)
    global_corr_neg = torch.zeros((549, 284), device=device)

    # 遍历测试集中所有的样本
    for sample_index in range(num_samples):
        svi_tensor, sample_id = test_dataset[sample_index]
        svi = svi_tensor.unsqueeze(0).to(device)
        svi = torch.clamp(svi / 255.0, 0.0, 1.0)
        svi.requires_grad = True

        pred = model(svi)

        # 存储当前单个样本的四种梯度矩阵
        sample_corr_abs = torch.zeros((549, 284), device=device)
        sample_corr_raw = torch.zeros((549, 284), device=device)
        sample_corr_pos = torch.zeros((549, 284), device=device)
        sample_corr_neg = torch.zeros((549, 284), device=device)

        for band_idx in range(284):
            model.zero_grad()
            if svi.grad is not None:
                svi.grad.zero_()

            target_band_mean = pred[0, band_idx, :, :].mean()
            target_band_mean.backward(retain_graph=True)

            grad_data = svi.grad.data

            # 1. 绝对值 (整体敏感度/重要性)
            sample_corr_abs[:, band_idx] = grad_data.abs().mean(dim=(0, 2, 3))

            # 2. 原始平均值 (包含正负抵消)
            sample_corr_raw[:, band_idx] = grad_data.mean(dim=(0, 2, 3))

            # 3. 纯正值 (仅保留促进作用)
            sample_corr_pos[:, band_idx] = torch.clamp(grad_data, min=0.0).mean(dim=(0, 2, 3))

            # 4. 纯负值 (仅保留抑制作用)
            sample_corr_neg[:, band_idx] = torch.clamp(grad_data, max=0.0).mean(dim=(0, 2, 3))

        # 累加到全局矩阵
        global_corr_abs += sample_corr_abs
        global_corr_raw += sample_corr_raw
        global_corr_pos += sample_corr_pos
        global_corr_neg += sample_corr_neg

        print(f" -> 样本 ID {sample_id} ({sample_index + 1}/{num_samples}) 计算完成。")

        del pred
        torch.cuda.empty_cache()

    # ==================================
    # 计算均值并保存四个矩阵
    # ==================================
    os.makedirs(results_dir, exist_ok=True)

    # 均值化并转为 numpy
    matrices = {
        "abs": (global_corr_abs / num_samples).cpu().numpy(),
        "raw": (global_corr_raw / num_samples).cpu().numpy(),
        "pos": (global_corr_pos / num_samples).cpu().numpy(),
        "neg": (global_corr_neg / num_samples).cpu().numpy()
    }

    row_labels = [f"R_{i}" for i in range(1, 184)] + \
                 [f"G_{i}" for i in range(1, 184)] + \
                 [f"B_{i}" for i in range(1, 184)]
    col_labels = [f"{wl:.1f}nm" for wl in np.linspace(400, 700, 284)]

    # 保存 CSV 和 NPY
    for name, matrix_np in matrices.items():
        np.save(os.path.join(results_dir, f"corr_matrix_{name}.npy"), matrix_np)
        df = pd.DataFrame(matrix_np, index=row_labels, columns=col_labels)
        df.to_csv(os.path.join(results_dir, f"corr_matrix_{name}.csv"))
        print(f"[Save] {name.upper()} 矩阵数据已保存.")

    # ==================================
    # 循环绘制四个热力图
    # ==================================
    # 定义 Abs 矩阵专用的渐变色
    colors = ["#FFFFFF", "#FFFF00", "#FF0000"]
    custom_cmap_abs = LinearSegmentedColormap.from_list("white_yellow_red", colors)

    # 顶部颜色条图像读取 (只读一次)
    color_bar_path = r"E:\songweiran\Imaging\colourbar.jpg"
    img_concat = None
    try:
        img_cb = Image.open(color_bar_path).convert('RGB')
        img_cb_np = np.array(img_cb)
        if img_cb_np.shape[0] > img_cb_np.shape[1]:
            img_cb_np = np.transpose(img_cb_np, (1, 0, 2))
        img_concat = np.concatenate([img_cb_np, img_cb_np, img_cb_np], axis=1)
    except Exception as e:
        print(f"\n[Warning] 无法加载颜色条图像: {e}")

    # 绘图配置列表
    plot_configs = [
        {"name": "abs", "title": "Absolute Correlation (Importance)", "cmap": custom_cmap_abs, "symmetric": False},
        {"name": "raw", "title": "Raw Average Correlation (With Cancellation)", "cmap": "RdBu_r", "symmetric": True},
        {"name": "pos", "title": "Positive Correlation (Promotion)", "cmap": "RdBu_r", "symmetric": True},
        {"name": "neg", "title": "Negative Correlation (Inhibition)", "cmap": "RdBu_r", "symmetric": True}
    ]

    for config in plot_configs:
        name = config["name"]
        matrix_to_plot = matrices[name].T

        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(2, 2, height_ratios=[0.5, 10], width_ratios=[20, 0.4], hspace=0.01, wspace=0.03)

        ax_cb = fig.add_subplot(gs[0, 0])
        ax_hm = fig.add_subplot(gs[1, 0])
        cax = fig.add_subplot(gs[1, 1])

        # 1. 顶部颜色条
        if img_concat is not None:
            ax_cb.imshow(img_concat, aspect='auto', extent=[1, 549, 0, 1])
        ax_cb.axis('off')
        ax_cb.spines['bottom'].set_visible(False)
        ax_cb.set_xlim(1, 549)
        ax_cb.set_title(config["title"], fontsize=14, pad=10)

        # 2. 核心热力图数值范围与色带设定
        if config["symmetric"]:
            # 对于 Raw, Pos, Neg，强制设定对称边界 [-Max, Max]，确保 0 对应纯白色
            max_val = np.percentile(np.abs(matrix_to_plot), 99)
            # 防止矩阵全为0的情况
            max_val = max_val if max_val > 0 else 1e-5
            vmin = -max_val
            vmax = max_val
        else:
            # 对于 Abs，只有正数，直接映射 0 到 99% 分位数
            vmin = 0
            vmax = np.percentile(matrix_to_plot, 99)

        im = ax_hm.imshow(matrix_to_plot, aspect='auto', cmap=config["cmap"], vmin=vmin, vmax=vmax,
                          extent=[1, 549, 400, 700], origin='lower')

        ax_hm.set_xlabel("Video Input Channels (Index 1 to 549)", fontsize=12, labelpad=8)
        ax_hm.set_ylabel("Reconstructed HSI Wavelength (nm)", fontsize=12, labelpad=8)

        ax_hm.axvline(x=183, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
        ax_hm.axvline(x=366, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
        ax_hm.set_xlim(1, 549)

        # 3. 右侧 Colorbar
        cbar = fig.colorbar(im, cax=cax)

        plt.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.1)

        # 保存图片
        save_fig_path = os.path.join(results_dir, f"corr_matrix_{name}_heatmap.png")
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 关闭画布释放内存
        print(f"[Save] 热力图已保存: {save_fig_path}")

    print("\n[Finish] 🏆 所有的相关性矩阵数据与热力图已全部生成并保存完毕！")


# ==========================================
# 4. 主函数运行入口
# ==========================================
if __name__ == '__main__':
    # ---------------- 本地路径配置 ----------------
    svi_dir = r"E:\songweiran\Imaging\Data20260326\Video\npy"
    hsi_dir = r"E:\songweiran\Imaging\Data20260326\HSI\npy"
    results_dir = r"E:\songweiran\Imaging\Data20260326\Results"

    model_weights_path = os.path.join(results_dir, "best_model_features2HSI.pth")
    # ----------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    model = MobileHSINet(in_ch=549, compressed_ch=64, feat_ch=64, num_blocks=4, out_ch=284).to(device)
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("✅ 成功加载预训练模型权重。")
    else:
        print("⚠️ 未找到预训练权重，当前使用随机初始化的权重进行测试。")

    # 选取所有测试集的 index
    test_indices = list(range(86, 101))
    test_dataset = SVI2HSIDatasetAnalysis(svi_dir, hsi_dir, test_indices)

    # 传入整个测试集进行全局计算
    compute_global_correlation_matrix(
        model=model,
        test_dataset=test_dataset,
        results_dir=results_dir,
        device=device
    )