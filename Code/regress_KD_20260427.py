import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import csv

# ===============================
# 0. 设置随机种子
# ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Initialization] 随机种子已统一设置为: {seed}")

# ===============================
# 1. 数据集与 5折划分
# ===============================
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
    Y_data = pd.read_csv(y_path, header=None).values.flatten()

    if len(Y_data) != 1550:
        raise ValueError(f"Y.csv 行数异常: {len(Y_data)}")

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

    return X_SVI_data, X_HSI_data, Y_data, forced_train_samples, folds

def get_row_indices(sample_list):
    records = []
    for s in sample_list:
        for spec_id in range(25):
            row_idx = (s - 1) * 25 + spec_id
            records.append((row_idx, s, spec_id + 1))
    return records

# ===============================
# 2. 网络
# ===============================
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

class Base_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
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

class HSI_Teacher_Net(Base_Net):
    def __init__(self):
        super().__init__(input_dim=616)

class SVI_Student_Net(Base_Net):
    def __init__(self):
        super().__init__(input_dim=552)

class RKDLoss(nn.Module):
    def forward(self, feat_S, feat_T):
        dist_S = torch.cdist(feat_S, feat_S, p=2)
        dist_T = torch.cdist(feat_T, feat_T, p=2)

        mean_S = dist_S.mean()
        mean_T = dist_T.mean()

        dist_S = dist_S / (mean_S + 1e-8)
        dist_T = dist_T / (mean_T + 1e-8)

        loss = F.l1_loss(dist_S, dist_T)
        return loss

def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / (ss_tot + 1e-8))

# ===============================
# 3. 蒸馏训练 + 内部验证
# ===============================
def train_and_evaluate():
    data_dir = r"E:\songweiran\Imaging\Data20260427"
    results_dir = r"E:\songweiran\Imaging\Data20260427\Results"
    feature_dir = r"E:\songweiran\Imaging\Data20260427\Results_feature"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    csv_log_path = os.path.join(results_dir, "loss_log_SVI_RKD_CV.csv")
    pred_save_path = os.path.join(results_dir, "predictions_SVI_RKD_CV.csv")
    feature_save_path = os.path.join(feature_dir, "SVI_distilled_features_CV.csv")

    X_SVI, X_HSI, Y_data, forced_train, folds = load_data_and_generate_folds(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 100

    avg_history_train_loss = np.zeros(epochs)
    avg_history_val_loss = np.zeros(epochs)

    all_oof_preds = []
    all_oof_trues = []
    all_oof_features = []
    oof_details = []

    print("[5-Fold Info]")
    for f in range(1, 6):
        print(f"Fold {f} test samples: {folds[f]}")

    for fold in range(1, 6):
        print(f"\n[Fold {fold}/5 Training]")

        model_save_path = os.path.join(results_dir, f"best_model_SVI_RKD_fold{fold}.pth")
        teacher_weight_path = os.path.join(results_dir, f"best_model_HSI_fold{fold}.pth")

        # 加载教师权重
        teacher = HSI_Teacher_Net().to(device)

        if os.path.exists(teacher_weight_path):
            teacher.load_state_dict(torch.load(teacher_weight_path, map_location=device))
            print(f"[Success] Fold {fold} HSI 教师权重已加载: {teacher_weight_path}")
        else:
            raise FileNotFoundError(
                f"找不到 Fold {fold} 的 HSI 教师权重，请确保已训练 HSI 教师模型。"
            )

        teacher.eval()

        for p in teacher.parameters():
            p.requires_grad = False

        # 数据索引
        test_samples = folds[fold]
        train_samples = forced_train + [
            s
            for f_idx, samps in folds.items()
            if f_idx != fold
            for s in samps
        ]

        train_records = get_row_indices(train_samples)
        test_records = get_row_indices(test_samples)

        train_idx = [r[0] for r in train_records]
        test_idx = [r[0] for r in test_records]

        test_sample_ids = [r[1] for r in test_records]
        test_spectrum_ids = [r[2] for r in test_records]

        SVI_tr_full = X_SVI[train_idx]
        HSI_tr_full = X_HSI[train_idx]
        Y_tr_full = Y_data[train_idx]

        SVI_ts = X_SVI[test_idx]
        HSI_ts = X_HSI[test_idx]
        Y_ts = Y_data[test_idx]

        # 内部验证集 20%
        val_size = int(0.2 * len(SVI_tr_full))
        train_size = len(SVI_tr_full) - val_size

        full_dataset = DistillationDataset(SVI_tr_full, HSI_tr_full, Y_tr_full)

        train_dataset, val_dataset = random_split(
            full_dataset,
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

        # 初始化学生网络
        student = SVI_Student_Net().to(device)

        criterion_task = nn.L1Loss()
        criterion_rkd = RKDLoss()

        optimizer = torch.optim.AdamW(
            student.parameters(),
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
            student.train()

            train_mae_total = 0.0

            for b_svi, b_hsi, b_y in train_loader:
                b_svi = b_svi.to(device)
                b_hsi = b_hsi.to(device)
                b_y = b_y.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    feat_T, _ = teacher(b_hsi)

                feat_S, pred_S = student(b_svi)

                loss_task = criterion_task(pred_S, b_y)
                loss_rkd = criterion_rkd(feat_S, feat_T)

                total_loss = loss_task + 0.5 * loss_rkd

                total_loss.backward()
                optimizer.step()

                train_mae_total += loss_task.item() * b_svi.size(0)

            train_mae = train_mae_total / len(train_loader.dataset)

            scheduler.step()

            # 内部验证集
            student.eval()

            val_mae_total = 0.0

            with torch.no_grad():
                for b_svi, b_hsi, b_y in val_loader:
                    b_svi = b_svi.to(device)
                    b_y = b_y.to(device)

                    _, p_s = student(b_svi)

                    val_mae_total += F.l1_loss(p_s, b_y).item() * b_svi.size(0)

            val_mae = val_mae_total / len(val_loader.dataset)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(student.state_dict(), model_save_path)

        # ===============================
        # 测试当前 fold
        # ===============================
        student.load_state_dict(torch.load(model_save_path, map_location=device))
        student.eval()

        test_dataset = DistillationDataset(SVI_ts, HSI_ts, Y_ts)

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False
        )

        # 新增：保存当前 fold 的预测值和真实值
        fold_preds = []
        fold_trues = []

        with torch.no_grad():
            for i, (b_svi, b_hsi, b_y) in enumerate(test_loader):
                b_svi = b_svi.to(device)

                feat, p = student(b_svi)

                preds_np = p.cpu().numpy()
                trues_np = b_y.cpu().numpy()
                feat_np = feat.cpu().numpy()

                # 新增：记录当前 fold 的预测值和真实值
                fold_preds.extend(preds_np.tolist())
                fold_trues.extend(trues_np.tolist())

                for j in range(len(preds_np)):
                    global_idx = i * test_loader.batch_size + j

                    if global_idx >= len(test_sample_ids):
                        break

                    all_oof_preds.append(preds_np[j])
                    all_oof_trues.append(b_y[j].item())
                    all_oof_features.append(feat_np[j].tolist())

                    oof_details.append([
                        fold,
                        test_sample_ids[global_idx],
                        test_spectrum_ids[global_idx],
                        b_y[j].item(),
                        preds_np[j]
                    ])

        # ===============================
        # 新增：打印当前 fold 的 R2、RMSE、MAE
        # ===============================
        fold_preds_arr = np.array(fold_preds)
        fold_trues_arr = np.array(fold_trues)

        fold_rmse = np.sqrt(np.mean((fold_preds_arr - fold_trues_arr) ** 2))
        fold_mae = np.mean(np.abs(fold_preds_arr - fold_trues_arr))
        fold_r2 = calculate_r2(fold_trues_arr, fold_preds_arr)

        print(f"\n[Fold {fold}/5 Test Metrics]")
        print(f"R²   = {fold_r2:.4f}")
        print(f"RMSE = {fold_rmse:.4f}")
        print(f"MAE  = {fold_mae:.4f}")

    # ===============================
    # 保存 CSV
    # ===============================
    pred_df = pd.DataFrame(
        oof_details,
        columns=[
            "Fold_ID",
            "Sample_ID",
            "Spectrum_ID",
            "True_Label",
            "Predicted_Label_SVI_RKD"
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

    # ===============================
    # 打印全局 OOF 指标
    # ===============================
    oof_preds_arr = np.array(all_oof_preds)
    oof_trues_arr = np.array(all_oof_trues)

    cv_rmse = np.sqrt(np.mean((oof_preds_arr - oof_trues_arr) ** 2))
    cv_mae = np.mean(np.abs(oof_preds_arr - oof_trues_arr))
    cv_r2 = calculate_r2(oof_trues_arr, oof_preds_arr)

    print("\n" + "=" * 60)
    print("[Results Summary] 5折交叉验证 SVI 蒸馏模型 OOF 指标:")
    print(f"全局 RMSE = {cv_rmse:.4f}")
    print(f"全局 MAE  = {cv_mae:.4f}")
    print(f"全局 R²   = {cv_r2:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    set_seed(42)
    train_and_evaluate()