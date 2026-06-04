import os
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio  # 仅保留用于保存预测结果的 .mat 文件
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric


# ==========================================
# 0. 全局设置与随机种子初始化 (保证实验可复现)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    print(f"[Initialization] 随机种子已统一设置为: {seed}")


# ==========================================
# 1. 数据集定义 (修改：一次性加载到内存 + 移除CPU归一化)
# ==========================================
class SVI2HSIDataset(Dataset):
    def __init__(self, svi_dir, hsi_dir, sample_indices, is_train=False):
        self.svi_dir = svi_dir
        self.hsi_dir = hsi_dir
        self.indices = sample_indices
        self.is_train = is_train

        # --- 新增：内存缓存字典 ---
        self.data_cache = {}
        print(f"[Dataset] 正在预加载 {len(sample_indices)} 个样本到内存, 请稍候...")
        for idx in sample_indices:
            svi_path = os.path.join(self.svi_dir, f"{idx}.npy")
            hsi_path = os.path.join(self.hsi_dir, f"{idx}.npy")

            try:
                # 预先读取
                svi_data = np.load(svi_path).astype(np.float32)  # 原始尺寸 [144, 144, 549]
                hsi_data = np.load(hsi_path).astype(np.float32)  # 原始尺寸 [180, 180, 284]

                # 提取基线白光数据：取第一帧的 RGB 通道 (索引 0, 183, 366)
                frame_indices = [0, 183, 366]
                svi_data = svi_data[:, :, frame_indices]

                # 调整数据维度顺序：[H, W, C] -> [C, H, W]
                svi_data = np.transpose(svi_data, (2, 0, 1))
                hsi_data = np.transpose(hsi_data, (2, 0, 1))

                # 存入缓存
                self.data_cache[idx] = (svi_data, hsi_data)
            except Exception as e:
                raise RuntimeError(f"读取文件失败: {idx}.npy. 详细错误: {e}")
        print("[Dataset] 预加载完成！")

    def __len__(self):
        # 训练集: 每个文件切分为 16 块 (4固定 + 12随机)
        if self.is_train:
            return len(self.indices) * 16
        # 测试/验证集: 采用整图输入，数量与索引一致
        return len(self.indices)

    def __getitem__(self, idx):
        # 建立全局 idx 到文件索引及切块位置的映射
        if self.is_train:
            file_idx = self.indices[idx // 16]
            crop_idx = idx % 16  # 0~3 为四个角，4~15 为随机
        else:
            file_idx = self.indices[idx]

        # --- 修改：直接从内存中获取数据，0 次磁盘 I/O ---
        svi_data, hsi_data = self.data_cache[file_idx]

        # --- 训练阶段：执行固定与随机裁剪 ---
        if self.is_train:
            # SVI 尺寸 144，目标 72。 HSI 尺寸 180，目标 90。比例 1.25
            svi_size = 144
            svi_crop = 72
            hsi_crop = 90

            if crop_idx == 0:
                # 左上
                svi_h_start, svi_w_start = 0, 0
            elif crop_idx == 1:
                # 右上
                svi_h_start, svi_w_start = 0, svi_size - svi_crop
            elif crop_idx == 2:
                # 左下
                svi_h_start, svi_w_start = svi_size - svi_crop, 0
            elif crop_idx == 3:
                # 右下
                svi_h_start, svi_w_start = svi_size - svi_crop, svi_size - svi_crop
            else:
                # 随机位置 (使用 randrange 并设置步长为 4)
                svi_h_start = random.randrange(0, (svi_size - svi_crop) + 1, 4)
                svi_w_start = random.randrange(0, (svi_size - svi_crop) + 1, 4)

            hsi_h_start = int(svi_h_start * 1.25)
            hsi_w_start = int(svi_w_start * 1.25)

            svi_data = svi_data[:, svi_h_start: svi_h_start + svi_crop, svi_w_start: svi_w_start + svi_crop]
            hsi_data = hsi_data[:, hsi_h_start: hsi_h_start + hsi_crop, hsi_w_start: hsi_w_start + hsi_crop]

        # --- 修改：移除了此处的除法和 clip 归一化操作，交由 GPU 处理 ---
        return torch.from_numpy(np.ascontiguousarray(svi_data)), torch.from_numpy(np.ascontiguousarray(hsi_data))


# ==========================================
# 2. 轻量化网络模型构建：MobileHSINet
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
        # 修正：去除了 BN 层，因此必须将 bias 改为 True，否则网络失去偏移补偿能力
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gelu = nn.GELU()
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        identity = x
        out = self.dw_conv(x)
        # 移除 BN
        out = self.gelu(out)
        out = self.pw_conv(out)
        # 移除 BN
        out = self.ca(out)
        return identity + out


class MobileHSINet(nn.Module):
    def __init__(self, in_ch=3, feat_ch=64, num_blocks=4, out_ch=284):
        super(MobileHSINet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1),
            nn.GELU()
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(MobileResidualBlock(feat_ch))
        self.body = nn.Sequential(*blocks)

        self.final_conv = nn.Sequential(
            nn.Conv2d(feat_ch, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 将上采样移至前部
        _, _, H, W = x.size()
        target_size = (int(H * 1.25), int(W * 1.25))
        x_up = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        pad_size = 8
        x_padded = F.pad(x_up, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

        feat = self.stem(x_padded)
        out = self.body(feat) + feat
        hsi_out_padded = self.final_conv(out)

        # 去除 padding
        hsi_out = hsi_out_padded[:, :, pad_size:-pad_size, pad_size:-pad_size]

        return hsi_out


# ==========================================
# 3. 评估指标计算模块
# ==========================================
def calculate_metrics(single_pred, single_gt):
    """
    修正：接收单张图像的 Tensor [C, H, W]，移除外层 batch 维度的限制，适配 batch_size != 1
    """
    pred_np = single_pred.cpu().numpy().transpose(1, 2, 0)
    gt_np = single_gt.cpu().numpy().transpose(1, 2, 0)

    psnr = psnr_metric(gt_np, pred_np, data_range=1.0)
    ssim = ssim_metric(gt_np, pred_np, data_range=1.0, channel_axis=-1)

    dot_product = np.sum(pred_np * gt_np, axis=2)
    norm_pred = np.linalg.norm(pred_np, axis=2)
    norm_gt = np.linalg.norm(gt_np, axis=2)
    denom = norm_pred * norm_gt
    denom[denom == 0] = 1e-8

    cos_theta = dot_product / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sam_map = np.arccos(cos_theta)
    sam = np.mean(sam_map) * (180 / np.pi)

    return psnr, ssim, sam


# ==========================================
# 4. 训练、验证与日志记录主进程
# ==========================================
def train_and_evaluate():
    # 数据集和输出路径配置
    svi_dir = r"E:\songweiran\Imaging\Data20260324\Video\npy"
    hsi_dir = r"E:\songweiran\Imaging\Data20260324\HSI\npy"
    save_pred_dir = r"E:\songweiran\Imaging\Data20260324\Video\predicted_RGB"
    results_dir = r"E:\songweiran\Imaging\Data20260324\Results"

    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 结果保存路径设定
    model_save_path = os.path.join(results_dir, "best_model_RGB2HSI.pth")
    csv_save_path = os.path.join(results_dir, "loss_log_RGB2HSI.csv")

    # 数据集划分
    train_indices = list(range(1, 61))
    val_indices = list(range(61, 71))
    test_indices = list(range(71, 81))

    # 实例化 Dataset
    train_dataset = SVI2HSIDataset(svi_dir, hsi_dir, train_indices, is_train=True)
    val_dataset = SVI2HSIDataset(svi_dir, hsi_dir, val_indices, is_train=False)
    test_dataset = SVI2HSIDataset(svi_dir, hsi_dir, test_indices, is_train=False)

    # 验证集和测试集的 batch_size 可以自由修改，不再强制为 1
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 硬件设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-" * 50)
    print(f"[System Info] 硬件平台: {device}")
    if torch.cuda.is_available():
        print(f"[System Info] GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"[Data Info] 训练集样本数 (增强后): {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")
    print(f"[Model Info] 最佳模型将被保存至: {model_save_path}")
    print("-" * 50)

    # 模型与优化器初始化
    model = MobileHSINet(in_ch=3, feat_ch=64, num_blocks=4, out_ch=284).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 学习率调度策略
    epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_mae = float('inf')

    # 记录训练过程用于生成图表和CSV
    history_train_loss = []
    history_val_loss = []

    print("[Train] 开始模型训练进程...")
    for epoch in range(epochs):
        model.train()
        train_mae = 0.0
        for svi, hsi in train_loader:
            # 加入 non_blocking=True 加速流转
            svi, hsi = svi.to(device, non_blocking=True), hsi.to(device, non_blocking=True)

            # --- 修改：在 GPU 端进行归一化和裁剪，降低 CPU 负载 ---
            svi = torch.clamp(svi / 255.0, 0.0, 1.0)
            hsi = torch.clamp(hsi / 2854.0, 0.0, 1.0)

            optimizer.zero_grad()
            outputs = model(svi)
            loss = criterion(outputs, hsi)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            # 使用乘以 batch_size 来累加
            train_mae += loss.item() * svi.size(0)

        train_mae /= len(train_dataset)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 验证环节
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for svi, hsi in val_loader:
                svi, hsi = svi.to(device, non_blocking=True), hsi.to(device, non_blocking=True)

                # --- 修改：在 GPU 端进行归一化和裁剪 ---
                svi = torch.clamp(svi / 255.0, 0.0, 1.0)
                hsi = torch.clamp(hsi / 2854.0, 0.0, 1.0)

                outputs = model(svi)
                loss = criterion(outputs, hsi)
                val_mae += loss.item() * svi.size(0)

        val_mae /= len(val_dataset)

        # 记录 loss
        history_train_loss.append(train_mae)
        history_val_loss.append(val_mae)

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] | LR: {current_lr:.6e} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

        # 保存最优模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> [Checkpoint] 验证集 MAE 达到新低 ({best_val_mae:.4f})，模型权重已更新。")

    # ==========================================
    # 4.1 将 Loss 保存为 CSV 文件并可视化展示
    # ==========================================
    print(f"\n[Log] 正在将训练日志保存至: {csv_save_path}")
    with open(csv_save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_MAE", "Val_MAE"])
        for i in range(epochs):
            writer.writerow([i + 1, history_train_loss[i], history_val_loss[i]])

    # 绘制 Epoch vs Loss 图 (仅显示，不保存)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), history_train_loss, label='Train MAE (RGB2HSI)', color='blue')
    plt.plot(range(1, epochs + 1), history_val_loss, label='Validation MAE (RGB2HSI)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ==========================================
    # 5. 测试集验证、性能评估与图像重建可视化
    # ==========================================
    print(f"\n[Eval] 训练结束。开始加载最佳模型权重，并在测试集 (ID: 71-80) 上进行评估...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_mae, test_psnr, test_ssim, test_sam = 0.0, 0.0, 0.0, 0.0

    # 引入全局样本索引，用来跟踪每个图像的具体 ID (解决 batch_size != 1 时对应测试集 ID 错乱问题)
    sample_idx = 0

    with torch.no_grad():
        for svi, hsi in test_loader:
            svi, hsi = svi.to(device, non_blocking=True), hsi.to(device, non_blocking=True)

            # --- 修改：在 GPU 端进行归一化和裁剪 ---
            svi = torch.clamp(svi / 255.0, 0.0, 1.0)
            hsi = torch.clamp(hsi / 2854.0, 0.0, 1.0)

            pred = model(svi)
            # MAE loss is batch-averaged by default, multiply by batch size to accumulate
            mae_batch = criterion(pred, hsi).item()
            test_mae += mae_batch * svi.size(0)

            # 遍历 Batch 中的每一个样本，单独计算指标和保存
            for b in range(svi.size(0)):
                single_pred = pred[b]
                single_gt = hsi[b]

                psnr, ssim, sam = calculate_metrics(single_pred, single_gt)
                test_psnr += psnr
                test_ssim += ssim
                test_sam += sam

                current_test_id = test_indices[sample_idx]
                print(f" 测试样本 ID {current_test_id:02d} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | SAM: {sam:.2f}°")

                # 保存预测结果 (.mat 格式)
                pred_save_raw = single_pred.cpu().numpy().transpose(1, 2, 0) * 2854.0
                mat_filename = os.path.join(save_pred_dir, f"{current_test_id}.mat")
                sio.savemat(mat_filename, {'reconstructed_hsi': pred_save_raw})

                # 对测试集第一个样本执行可视化
                if sample_idx == 0:
                    visualize_results(single_pred, single_gt, sample_id=current_test_id, results_dir=results_dir)

                # 更新全局指针
                sample_idx += 1

    num_test = len(test_dataset)
    print("\n" + "=" * 50)
    print(f"[Results] 基于 RGB 白光的 HSI 重建测试集平均性能指标：")
    print(f" -> 平均 MAE:  {test_mae / num_test:.4f}")

    # 修正：除以实际样本总数，而不是 len(test_loader)
    print(f" -> 平均 PSNR: {test_psnr / num_test:.2f} dB")
    print(f" -> 平均 SSIM: {test_ssim / num_test:.4f}")
    print(f" -> 平均 SAM:  {test_sam / num_test:.2f} °")
    print(f"[Output] 所有测试集预测结果均已成功保存至目录: {save_pred_dir}")
    print("=" * 50)


def visualize_results(single_pred, single_gt, sample_id, results_dir):
    """可视化重建结果与真值的光谱和图像对比"""
    # 修正：处理无 batch 维度的输入
    pred_np = single_pred.cpu().numpy()
    hsi_np = single_gt.cpu().numpy()

    # 选取对应波长的通道进行展示
    target_band = 142
    band_pred = pred_np[target_band, :, :]
    band_gt = hsi_np[target_band, :, :]

    # 提取中心 5x5 ROI 用于绘制光谱曲线
    roi_start, roi_end = 88, 93
    curve_pred = np.mean(pred_np[:, roi_start:roi_end, roi_start:roi_end], axis=(1, 2))
    curve_gt = np.mean(hsi_np[:, roi_start:roi_end, roi_start:roi_end], axis=(1, 2))

    wavelengths = np.linspace(400, 700, 284)

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Reconstruction Performance - Sample ID: {sample_id} (RGB2HSI Baseline)", fontsize=16)

    plt.subplot(1, 3, 1)
    plt.title("Ground Truth (~550nm)")
    plt.imshow(band_gt, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted (~550nm)")
    plt.imshow(band_pred, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Mean Spectral Curve (5x5 ROI at Center)")
    plt.plot(wavelengths, curve_gt, label='Ground Truth', color='blue')
    plt.plot(wavelengths, curve_pred, label='Predicted (RGB2HSI)', color='red', linestyle='--')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance (Normalized)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # 保存可视化图片，命名带有 RGB2HSI 标识
    save_fig_path = os.path.join(results_dir, f"RGB2HSI_Result_Sample_{sample_id}.png")
    plt.savefig(save_fig_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    set_seed(42)
    train_and_evaluate()