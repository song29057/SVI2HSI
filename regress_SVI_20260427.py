import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
class SVIDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_data).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_data_and_generate_folds(data_dir):
    x_path = os.path.join(data_dir, "X_SVI.csv")
    y_path = os.path.join(data_dir, "Y.csv")

    print("[Data] 正在读取 SVI 数据集...")
    # 修改点 1: 除以 255.0 将 SVI (视频色彩特征) 归一化至 [0, 1] 区间
    X_data = pd.read_csv(x_path, header=None).values / 255.0
    Y_df = pd.read_csv(y_path, header=None)

    # 验证输入维度是否为 552
    if X_data.shape[1] != 552:
        print(f"[Warning] 侦测到 X_SVI 的特征维度为 {X_data.shape[1]}，期望值为 552。")

    Y_data = Y_df.values.flatten()
    if len(Y_data) != 1550:
        raise ValueError(f"Y.csv 的行数异常: {len(Y_data)}，期望为 1550。")

    # 样本划分逻辑 (共62个样品)
    all_samples = list(range(1, 63))
    forced_train_samples = [1, 2, 61, 62]  # 极端值强制加入训练集

    # 获取除去强制样本外的剩余 58 个样本
    remaining_samples = [s for s in all_samples if s not in forced_train_samples]
    random.shuffle(remaining_samples)

    # 按照 1234512345 循环顺序分配到 5 个折中
    folds = {1: [], 2: [], 3: [], 4: [], 5: []}
    for i, sample in enumerate(remaining_samples):
        fold_idx = (i % 5) + 1
        folds[fold_idx].append(sample)

    for k in folds:
        folds[k].sort()

    return X_data, Y_data, forced_train_samples, folds


def get_row_indices(sample_list):
    indices = []
    for s in sample_list:
        start_idx = (s - 1) * 25
        end_idx = s * 25
        indices.extend(list(range(start_idx, end_idx)))
    return indices


# ==========================================
# 2. 定量分析模型构建 (微型残差架构 Micro-ResMLP)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim=32):
        super(ResidualBlock, self).__init__()
        # 采用现代大模型（如 Transformer）主流的 Pre-Norm 结构
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        # 残差连接：输入特征直接跨层与新提取的特征相加
        return x + self.block(x)


class SVI_Baseline_Net(nn.Module):
    def __init__(self, input_dim=552, hidden_dim=32):
        super(SVI_Baseline_Net, self).__init__()

        # 1. Stem (主干特征提取)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 2. Body (残差深度加工)
        self.res_blocks = nn.Sequential(
            ResidualBlock(dim=hidden_dim),
            ResidualBlock(dim=hidden_dim)
        )

        # 3. Head (预测输出)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x).squeeze(-1)


# 辅助函数: 计算 R2 分数
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


# ==========================================
# 3. 训练、评估与日志记录主进程 (5折交叉验证)
# ==========================================
def train_and_evaluate():
    data_dir = r"E:\songweiran\Imaging\Data20260427"
    results_dir = r"E:\songweiran\Imaging\Data20260427\Results"
    os.makedirs(results_dir, exist_ok=True)

    csv_log_path = os.path.join(results_dir, "loss_log_SVI_CV.csv")
    pred_save_path = os.path.join(results_dir, "predictions_SVI_CV.csv")

    X_data, Y_data, forced_train, folds = load_data_and_generate_folds(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 100
    # 用于记录 5 折的平均 Loss 以供绘图
    avg_history_train_loss = np.zeros(epochs)
    avg_history_val_loss = np.zeros(epochs)

    # 用于收集所有 5 折的 Out-of-Fold (OOF) 预测结果
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
        print(f"\n[{'=' * 15} 开始 Fold {fold}/5 交叉验证训练 {'=' * 15}]")

        # 为当前折生成保存路径
        model_save_path = os.path.join(results_dir, f"best_model_SVI_fold{fold}.pth")

        # 划分当前折的数据
        test_samples = folds[fold]
        # 训练集 = 强制样本 + 其余所有折的样本
        train_samples = forced_train + [s for f_idx, samps in folds.items() if f_idx != fold for s in samps]

        train_idx = get_row_indices(train_samples)
        test_idx = get_row_indices(test_samples)

        X_train, Y_train = X_data[train_idx], Y_data[train_idx]
        X_test, Y_test = X_data[test_idx], Y_data[test_idx]

        train_loader = DataLoader(SVIDataset(X_train, Y_train), batch_size=64, shuffle=True)
        # 在 K-Fold 中，用每一折的独立测试集作为验证集来监控泛化能力
        val_loader = DataLoader(SVIDataset(X_test, Y_test), batch_size=64, shuffle=False)

        model = SVI_Baseline_Net(input_dim=552).to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_val_mae = float('inf')

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)

            train_loss /= len(X_train)
            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    preds = model(batch_x)
                    loss = criterion(preds, batch_y)
                    val_loss += loss.item() * batch_x.size(0)

            val_loss /= len(X_test)

            avg_history_train_loss[epoch] += train_loss / 5.0
            avg_history_val_loss[epoch] += val_loss / 5.0

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(
                    f"  Fold {fold} - Epoch [{epoch + 1:03d}/{epochs}] | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}")

            if val_loss < best_val_mae:
                best_val_mae = val_loss
                torch.save(model.state_dict(), model_save_path)

        print(f"  --> [Fold {fold} Checkpoint] 最佳验证 MAE 达到 ({best_val_mae:.4f})，已保存至 {model_save_path}")

        # --- 收集当前折的最佳预测结果 ---
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                preds = model(batch_x).cpu().numpy()
                all_oof_preds.extend(preds.tolist())
                all_oof_trues.extend(batch_y.numpy().tolist())
                # 记录详细信息以便保存
                for p, t in zip(preds, batch_y.numpy()):
                    oof_details.append([fold, t, p])

    # ==========================================
    # 4. 保存 5 折平均 Loss 和 预测结果
    # ==========================================
    with open(csv_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Avg_Train_MAE", "Avg_Val_MAE"])
        for i in range(epochs):
            writer.writerow([i + 1, avg_history_train_loss[i], avg_history_val_loss[i]])

    pred_df = pd.DataFrame(oof_details, columns=["Fold_ID", "True_Label", "Predicted_Label_SVI"])
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
    print(" [Results Summary] 5折交叉验证 SVI 微型残差基线模型 评估结果:")
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
    plt.title('Average Training and Validation Loss Curve (SVI CV)')
    plt.legend()
    plt.grid(True)

    # --- 子图 2: 预测值 vs 真值 散点图 (按样品聚合并画 Error Bar) ---
    plt.subplot(1, 2, 2)

    # 交叉验证数据处理 (因为有 58 个测试样本，共 1450 条光谱)
    oof_preds_np = oof_preds_arr.reshape(-1, 25)
    oof_trues_np = oof_trues_arr.reshape(-1, 25)[:, 0]
    oof_preds_mean = oof_preds_np.mean(axis=1)
    oof_preds_std = oof_preds_np.std(axis=1)

    # 使用 errorbar 绘制带有标准差的交叉验证散点图
    plt.errorbar(oof_trues_np, oof_preds_mean, yerr=oof_preds_std, fmt='s',
                 alpha=0.8, label='Cross-Validation OOF (Mean ± SD)', color='blue', capsize=4)

    # 添加 y = x 的理想基准线
    min_val = min(oof_trues_np.min(), oof_preds_mean.min())
    max_val = max(oof_trues_np.max(), oof_preds_mean.max())
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
             'r--', linewidth=2, label='Ideal (y=x)')

    plt.xlabel('True Values (Per Sample)')
    plt.ylabel('Predicted Values (Mean of 25 Spectra)')
    plt.title('Predicted vs. True Values with Variance (SVI 5-Fold CV)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    set_seed(42)
    train_and_evaluate()