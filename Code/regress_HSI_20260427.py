import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import csv

# ==========================================
# 0. 全局设置与随机种子初始化
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
class HSIDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_data).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_data_and_generate_folds(data_dir):
    x_path = os.path.join(data_dir, "X_HSI.csv")
    y_path = os.path.join(data_dir, "Y.csv")

    print("[Data] 正在读取数据集...")

    X_data = pd.read_csv(x_path, header=None).values / 3804.0
    Y_df = pd.read_csv(y_path, header=None)
    Y_data = Y_df.values.flatten()

    if len(Y_data) != 1550:
        raise ValueError(f"Y.csv 的行数异常: {len(Y_data)}，期望为 1550。")

    all_samples = list(range(1, 63))
    forced_train_samples = [1, 2, 61, 62]

    remaining_samples = [s for s in all_samples if s not in forced_train_samples]
    random.shuffle(remaining_samples)

    folds = {1: [], 2: [], 3: [], 4: [], 5: []}

    for i, sample in enumerate(remaining_samples):
        fold_idx = (i % 5) + 1
        folds[fold_idx].append(sample)

    for k in folds:
        folds[k].sort()

    return X_data, Y_data, forced_train_samples, folds

def get_row_indices(sample_list):
    records = []

    for s in sample_list:
        for spec_id in range(25):
            row_idx = (s - 1) * 25 + spec_id
            records.append((row_idx, s, spec_id + 1))

    return records

# ==========================================
# 2. 微型残差网络 Micro-ResMLP
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class HSI_Baseline_Net(nn.Module):
    def __init__(self, input_dim=616, hidden_dim=32):
        super().__init__()

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
        feat = self.res_blocks(x)
        out = self.head(feat).squeeze(-1)
        return feat, out

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)

# ==========================================
# 3. 训练与微调主函数（内部验证 + 测试fold评估）
# ==========================================
def train_and_evaluate():
    data_dir = r"E:\songweiran\Imaging\Data20260427"
    results_dir = r"E:\songweiran\Imaging\Data20260427\Results"
    feature_dir = r"E:\songweiran\Imaging\Data20260427\Results_feature"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    csv_log_path = os.path.join(results_dir, "loss_log_HSI_CV.csv")
    pred_save_path = os.path.join(results_dir, "predictions_HSI_CV.csv")
    feature_save_path = os.path.join(feature_dir, "HSI_baseline_features_CV.csv")

    X_data, Y_data, forced_train, folds = load_data_and_generate_folds(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 100

    avg_history_train_loss = np.zeros(epochs)
    avg_history_val_loss = np.zeros(epochs)

    all_oof_preds = []
    all_oof_trues = []
    all_oof_features = []
    oof_details = []

    print("\n" + "=" * 60)
    print(" [Data Split Info] 5折交叉验证 样品分配情况:")

    for f in range(1, 6):
        print(f"Fold {f} 测试集样品: {folds[f]}")

    for fold in range(1, 6):
        print(f"\n[Fold {fold}/5 Training]")

        model_save_path = os.path.join(results_dir, f"best_model_HSI_fold{fold}.pth")

        # 获取行索引 + Sample_ID / Spectrum_ID
        test_records = get_row_indices(folds[fold])

        train_samples = forced_train + [
            s
            for f_idx, samps in folds.items()
            if f_idx != fold
            for s in samps
        ]

        train_records = get_row_indices(train_samples)

        train_idx = [r[0] for r in train_records]
        test_idx = [r[0] for r in test_records]

        test_sample_ids = [r[1] for r in test_records]
        test_spectrum_ids = [r[2] for r in test_records]

        X_train_full = X_data[train_idx]
        Y_train_full = Y_data[train_idx]

        X_test = X_data[test_idx]
        Y_test = Y_data[test_idx]

        # --- 内部验证集拆分 20% ---
        val_size = int(0.2 * len(X_train_full))
        train_size = len(X_train_full) - val_size

        train_dataset_full = HSIDataset(X_train_full, Y_train_full)

        train_dataset, val_dataset = random_split(
            train_dataset_full,
            [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False
        )

        model = HSI_Baseline_Net(input_dim=616).to(device)

        criterion = nn.L1Loss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-5
        )

        best_val_mae = float("inf")

        for epoch in range(epochs):
            model.train()

            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()

                _, preds = model(batch_x)

                loss = criterion(preds, batch_y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)

            train_loss /= len(train_loader.dataset)

            scheduler.step()

            # 内部验证集
            model.eval()

            val_loss = 0.0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    _, preds = model(batch_x)

                    val_loss += criterion(preds, batch_y).item() * batch_x.size(0)

            val_loss /= len(val_loader.dataset)

            avg_history_train_loss[epoch] += train_loss / 5.0
            avg_history_val_loss[epoch] += val_loss / 5.0

            if val_loss < best_val_mae:
                best_val_mae = val_loss
                torch.save(model.state_dict(), model_save_path)

        # ==========================================
        # 测试当前 fold
        # ==========================================
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        test_dataset = HSIDataset(X_test, Y_test)

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False
        )

        # 新增：用于保存当前 fold 的预测值和真实值
        fold_preds = []
        fold_trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.to(device)

                feat, preds = model(batch_x)

                preds_np = preds.cpu().numpy()
                trues_np = batch_y.cpu().numpy()
                feat_np = feat.cpu().numpy()

                # 新增：记录当前 fold 的预测值和真实值
                fold_preds.extend(preds_np.tolist())
                fold_trues.extend(trues_np.tolist())

                batch_size = len(preds_np)

                for j in range(batch_size):
                    global_idx = i * test_loader.batch_size + j

                    if global_idx >= len(test_sample_ids):
                        break

                    all_oof_preds.append(preds_np[j])
                    all_oof_trues.append(batch_y[j].item())
                    all_oof_features.append(feat_np[j].tolist())

                    oof_details.append([
                        fold,
                        test_sample_ids[global_idx],
                        test_spectrum_ids[global_idx],
                        batch_y[j].item(),
                        preds_np[j]
                    ])

        # ==========================================
        # 新增：打印当前 fold 的 R2、RMSE、MAE
        # ==========================================
        fold_preds_arr = np.array(fold_preds)
        fold_trues_arr = np.array(fold_trues)

        fold_rmse = np.sqrt(np.mean((fold_preds_arr - fold_trues_arr) ** 2))
        fold_mae = np.mean(np.abs(fold_preds_arr - fold_trues_arr))
        fold_r2 = calculate_r2(fold_trues_arr, fold_preds_arr)

        print(f"\n[Fold {fold}/5 Test Metrics]")
        print(f"R2   = {fold_r2:.4f}")
        print(f"RMSE = {fold_rmse:.4f}")
        print(f"MAE  = {fold_mae:.4f}")

    # ==========================================
    # 保存特征与预测
    # ==========================================
    pred_df = pd.DataFrame(
        oof_details,
        columns=[
            "Fold_ID",
            "Sample_ID",
            "Spectrum_ID",
            "True_Label",
            "Predicted_Label_HSI"
        ]
    )

    if all_oof_features:
        feat_cols = [
            f"Feature_{i + 1}"
            for i in range(len(all_oof_features[0]))
        ]

        feat_df = pd.DataFrame(
            all_oof_features,
            columns=feat_cols
        )

        final_df = pd.concat(
            [pred_df, feat_df],
            axis=1
        )

        final_df.to_csv(feature_save_path, index=False)

    pred_df.to_csv(pred_save_path, index=False)

    print(f"[Output] HSI 5折预测结果及特征已保存至: {pred_save_path} / {feature_save_path}")

    # ==========================================
    # 全局 OOF 指标
    # ==========================================
    oof_preds_arr = np.array(all_oof_preds)
    oof_trues_arr = np.array(all_oof_trues)

    cv_rmse = np.sqrt(np.mean((oof_preds_arr - oof_trues_arr) ** 2))
    cv_mae = np.mean(np.abs(oof_preds_arr - oof_trues_arr))
    cv_r2 = calculate_r2(oof_trues_arr, oof_preds_arr)

    print("\n" + "=" * 60)
    print("[Results Summary] 5折交叉验证 HSI Baseline 模型 OOF 指标:")
    print(f"全局 RMSE = {cv_rmse:.4f}")
    print(f"全局 MAE  = {cv_mae:.4f}")
    print(f"全局 R2   = {cv_r2:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    set_seed(42)
    train_and_evaluate()