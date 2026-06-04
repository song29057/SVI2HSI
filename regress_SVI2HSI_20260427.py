import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import csv


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
    print(f"[Initialization] 随机种子已统一设置为: {seed}")


# ==========================================
# 1. 数据集定义与 5折交叉验证样本级划分
# ==========================================
class DistillationDataset(Dataset):
    def __init__(self, svi_data, hsi_data, y_data):
        self.svi = torch.from_numpy(svi_data).float()
        self.hsi = torch.from_numpy(hsi_data).float()
        self.y = torch.from_numpy(y_data).float()

    def __len__(self):
        return len(self.svi)

    def __getitem__(self, idx):
        return self.svi[idx], self.hsi[idx], self.y[idx]


def load_data_and_generate_folds(data_dir):
    svi_path = os.path.join(data_dir, "X_SVI.csv")
    hsi_path = os.path.join(data_dir, "X_HSI.csv")
    y_path = os.path.join(data_dir, "Y.csv")

    print("[Data] 正在读取 SVI, HSI 与 Y 标签数据集...")
    X_SVI_data = pd.read_csv(svi_path, header=None).values / 255.0
    X_HSI_data = pd.read_csv(hsi_path, header=None).values / 3804.0
    Y_df = pd.read_csv(y_path, header=None)

    Y_data = Y_df.values.flatten()
    if len(Y_data) != 1550:
        raise ValueError(f"Y.csv 的行数异常: {len(Y_data)}，期望为 1550。")

    all_samples = list(range(1, 63))
    forced_train_samples = [1, 2, 61, 62]  # 极端值强制加入训练集

    remaining_samples = [s for s in all_samples if s not in forced_train_samples]
    random.shuffle(remaining_samples)

    # 按照 1234512345 循环顺序分配到 5 个折中
    folds = {1: [], 2: [], 3: [], 4: [], 5: []}
    for i, sample in enumerate(remaining_samples):
        fold_idx = (i % 5) + 1
        folds[fold_idx].append(sample)

    for k in folds:
        folds[k].sort()

    return X_SVI_data, X_HSI_data, Y_data, forced_train_samples, folds


def get_row_indices(sample_list):
    indices = []
    for s in sample_list:
        start_idx = (s - 1) * 25
        end_idx = s * 25
        indices.extend(list(range(start_idx, end_idx)))
    return indices


# ==========================================
# 2. 网络构建 (微型残差架构 Micro-ResMLP)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim=32):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)


class Base_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(Base_Net, self).__init__()

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(dim=hidden_dim),
            ResidualBlock(dim=hidden_dim)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        feat = self.res_blocks(x)  # <-- 拦截特征
        out = self.head(feat)
        return feat, out.squeeze(-1)


class HSI_Teacher_Net(Base_Net):
    def __init__(self): super().__init__(input_dim=616)


class SVI_Student_Net(Base_Net):
    def __init__(self): super().__init__(input_dim=552)


# ==========================================
# 3. 关系一致性蒸馏 (RKD) 损失定义
# ==========================================
class RKDLoss(nn.Module):
    def forward(self, feat_S, feat_T):
        dist_S = torch.cdist(feat_S, feat_S, p=2)
        dist_T = torch.cdist(feat_T, feat_T, p=2)

        mean_S = dist_S.mean()
        dist_S = dist_S / (mean_S + 1e-8)

        mean_T = dist_T.mean()
        dist_T = dist_T / (mean_T + 1e-8)

        loss = F.l1_loss(dist_S, dist_T)
        return loss


def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: return 0.0
    return 1 - (ss_res / (ss_tot + 1e-8))


# ==========================================
# 4. 训练、评估与日志记录主进程 (5折交叉验证)
# ==========================================
def train_and_evaluate():
    data_dir = r"E:\songweiran\Imaging\Data20260427"
    results_dir = r"E:\songweiran\Imaging\Data20260427\Results"
    os.makedirs(results_dir, exist_ok=True)

    csv_log_path = os.path.join(results_dir, "loss_log_SVI_RKD_pure_CV.csv")
    pred_save_path = os.path.join(results_dir, "predictions_SVI_RKD_pure_CV.csv")

    X_SVI, X_HSI, Y_data, forced_train, folds = load_data_and_generate_folds(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 100
    avg_history_train_loss = np.zeros(epochs)
    avg_history_val_loss = np.zeros(epochs)

    all_oof_preds = []
    all_oof_trues = []
    oof_details = []

    print("\n" + "=" * 60)
    print(" [Data Split Info] 5折交叉验证 样品分配情况清单:")
    print("-" * 60)
    print(f" 强制训练集 | 共 {len(forced_train):2d} 样品: {forced_train}")
    for f in range(1, 6):
        print(f" 折 {f} 测试集 | 共 {len(folds[f]):2d} 样品: {folds[f]}")
    print("=" * 60)

    for fold in range(1, 6):
        print(f"\n[{'=' * 15} 开始 Fold {fold}/5 RKD 纯净蒸馏训练 {'=' * 15}]")

        # 保存和读取对应的 fold 权重
        model_save_path = os.path.join(results_dir, f"best_model_SVI_RKD_pure_fold{fold}.pth")
        teacher_weight_path = os.path.join(results_dir, f"best_model_HSI_fold{fold}.pth")  # <--- 核心对齐：读取对应折的教师

        # 划分当前折数据
        test_samples = folds[fold]
        train_samples = forced_train + [s for f_idx, samps in folds.items() if f_idx != fold for s in samps]

        train_idx = get_row_indices(train_samples)
        test_idx = get_row_indices(test_samples)

        SVI_tr, HSI_tr, Y_tr = X_SVI[train_idx], X_HSI[train_idx], Y_data[train_idx]
        SVI_ts, HSI_ts, Y_ts = X_SVI[test_idx], X_HSI[test_idx], Y_data[test_idx]

        train_loader = DataLoader(DistillationDataset(SVI_tr, HSI_tr, Y_tr), batch_size=64, shuffle=True)
        val_loader = DataLoader(DistillationDataset(SVI_ts, HSI_ts, Y_ts), batch_size=64, shuffle=False)

        # 1. 加载对应的 HSI 教师网络
        teacher = HSI_Teacher_Net().to(device)
        if os.path.exists(teacher_weight_path):
            teacher.load_state_dict(torch.load(teacher_weight_path, map_location=device))
            print(f" [Success] 对应折教师网络已加载: {teacher_weight_path}")
        else:
            raise FileNotFoundError(f"找不到 Fold {fold} 的教师权重，请确保先运行了 HSI 交叉验证基线！")
        teacher.eval()
        for param in teacher.parameters(): param.requires_grad = False

        # 2. 初始化 SVI 学生网络
        student = SVI_Student_Net().to(device)

        criterion_task = nn.L1Loss()
        criterion_rkd = RKDLoss()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        w_task = 1.0
        w_rkd = 1.5
        best_val_mae = float('inf')

        for epoch in range(epochs):
            student.train()
            train_mae_total = 0.0

            for b_svi, b_hsi, b_y in train_loader:
                b_svi, b_hsi, b_y = b_svi.to(device), b_hsi.to(device), b_y.to(device)
                optimizer.zero_grad()

                with torch.no_grad():
                    feat_T, _ = teacher(b_hsi)

                feat_S, pred_S = student(b_svi)

                loss_task = criterion_task(pred_S, b_y)
                loss_rkd = criterion_rkd(feat_S, feat_T)

                total_loss = (w_task * loss_task) + (w_rkd * loss_rkd)
                total_loss.backward()
                optimizer.step()

                train_mae_total += loss_task.item() * b_svi.size(0)

            train_mae = train_mae_total / len(SVI_tr)
            scheduler.step()

            student.eval()
            val_mae_total = 0.0
            with torch.no_grad():
                for b_svi, _, b_y in val_loader:
                    b_svi, b_y = b_svi.to(device), b_y.to(device)
                    _, p_s = student(b_svi)
                    val_mae_total += F.l1_loss(p_s, b_y).item() * b_svi.size(0)

            val_mae = val_mae_total / len(SVI_ts)

            avg_history_train_loss[epoch] += train_mae / 5.0
            avg_history_val_loss[epoch] += val_mae / 5.0

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(
                    f"  Fold {fold} - Epoch [{epoch + 1:03d}/{epochs}] | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(student.state_dict(), model_save_path)

        print(f"  --> [Fold {fold} Checkpoint] 最佳验证 MAE 达到 ({best_val_mae:.4f})，已保存至 {model_save_path}")

        # --- 收集当前折的最佳预测结果 ---
        student.load_state_dict(torch.load(model_save_path))
        student.eval()
        with torch.no_grad():
            for b_svi, _, b_y in val_loader:
                b_svi = b_svi.to(device)
                _, p = student(b_svi)
                preds = p.cpu().numpy()
                all_oof_preds.extend(preds.tolist())
                all_oof_trues.extend(b_y.numpy().tolist())
                for pred_val, true_val in zip(preds, b_y.numpy()):
                    oof_details.append([fold, true_val, pred_val])

    # ==========================================
    # 4. 保存 5 折平均 Loss 和 预测结果
    # ==========================================
    with open(csv_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Avg_Train_MAE", "Avg_Val_MAE"])
        for i in range(epochs):
            writer.writerow([i + 1, avg_history_train_loss[i], avg_history_val_loss[i]])

    pred_df = pd.DataFrame(oof_details, columns=["Fold_ID", "True_Label", "Predicted_Label_SVI_RKD_pure"])
    pred_df.to_csv(pred_save_path, index=False)
    print(f"\n[Output] 5折交叉验证 预测结果已成功保存至: {pred_save_path}")

    # ==========================================
    # 5. 5折 OOF (Out-of-Fold) 综合指标计算
    # ==========================================
    oof_preds_arr = np.array(all_oof_preds)
    oof_trues_arr = np.array(all_oof_trues)

    cv_rmse_final = np.sqrt(np.mean((oof_preds_arr - oof_trues_arr) ** 2))
    cv_mae_final = np.mean(np.abs(oof_preds_arr - oof_trues_arr))
    cv_r2_final = calculate_r2(oof_trues_arr, oof_preds_arr)

    print("\n" + "=" * 60)
    print(" [Results Summary] 5折交叉验证 SVI_RKD_ResMLP (纯净蒸馏版) 评估结果:")
    print("-" * 60)
    print(f" | Data Set | Samples | Spectra |   RMSE   |   MAE    |    R²    |")
    print("-" * 60)
    print(f" | 全局 CV  |    58   |   1450  |  {cv_rmse_final:.4f}  |  {cv_mae_final:.4f}  |  {cv_r2_final:.4f}  |")
    print("=" * 60)

    # ==========================================
    # 6. 图表绘制 (屏幕展示, 不保存文件) - 带 Error Bar
    # ==========================================
    plt.figure(figsize=(15, 6))

    # --- 子图 1: 平均 Loss 曲线 ---
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), avg_history_train_loss, label='5-Fold Avg Train MAE', color='blue')
    plt.plot(range(1, epochs + 1), avg_history_val_loss, label='5-Fold Avg Validation MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (L1 Loss)')
    plt.title('Average Training and Validation Loss Curve (SVI RKD pure CV)')
    plt.legend()
    plt.grid(True)

    # --- 子图 2: 预测值 vs 真值 散点图 (按样品聚合并画 Error Bar) ---
    plt.subplot(1, 2, 2)

    # 交叉验证数据处理
    oof_preds_np = oof_preds_arr.reshape(-1, 25)
    oof_trues_np = oof_trues_arr.reshape(-1, 25)[:, 0]
    oof_preds_mean = oof_preds_np.mean(axis=1)
    oof_preds_std = oof_preds_np.std(axis=1)

    # 绘制带有标准差的交叉验证散点图
    plt.errorbar(oof_trues_np, oof_preds_mean, yerr=oof_preds_std, fmt='s',
                 alpha=0.8, label='Cross-Validation OOF (Mean ± SD)', color='blue', capsize=4)

    # 添加 y = x 理想基准线
    min_val = min(oof_trues_np.min(), oof_preds_mean.min())
    max_val = max(oof_trues_np.max(), oof_preds_mean.max())
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
             'r--', linewidth=2, label='Ideal (y=x)')

    plt.xlabel('True Values (Per Sample)')
    plt.ylabel('Predicted Values (Mean of 25 Spectra)')
    plt.title('Predicted vs. True Values with Variance (SVI RKD pure CV)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    set_seed(42)
    train_and_evaluate()