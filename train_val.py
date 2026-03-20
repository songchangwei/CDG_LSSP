import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

multiprocessing.set_start_method('spawn', force=True)

# ======================== 1. 核心配置 ========================
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 使用设备：{DEVICE}")

# 数据配置
DATA_ROOT = "./cdg_lssp_sim_data"
MAX_LEN_THRESHOLD = 1600  # 最大序列长度
BATCH_SIZE = 8
NUM_WORKERS = 0  # 保留多线程，提升加载效率

# 训练超参数
EPOCHS = 100               # 训练轮数
LEARNING_RATE = 1e-4      # 初始学习率
WEIGHT_DECAY = 1e-5       # 权重衰减（L2正则）
GRAD_CLIP_NORM = 5.0      # 梯度裁剪阈值
PATIENCE = 20              # 早停耐心值（验证集AUC不提升则停止）
LOG_INTERVAL = 100        # 每多少批次打印一次训练日志

# 保存配置
MODEL_SAVE_DIR = "./trained_models"
LOG_SAVE_PATH = "./train_log.txt"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ======================== 2. 复用动态数据加载代码 ========================
# 数据集类
class CDGLSSPVariableDataset(Dataset):
    def __init__(self, data_dir: str, max_len_threshold: int = 256):
        self.data_dir = data_dir
        self.max_len_threshold = max_len_threshold
        self.file_list = [
            f for f in os.listdir(data_dir) 
            if f.endswith(".npy") and f.split("_")[0] in ["train", "val"]
        ]
        self.file_list.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        self._validate_dataset()

    def _validate_dataset(self):
        valid_files = []
        for f in self.file_list:
            file_path = os.path.join(self.data_dir, f)
            try:
                sample = np.load(file_path, allow_pickle=True).item()
                assert "seq_len" in sample and "f_d" in sample
                assert sample["f_d"].shape[0] == sample["seq_len"]
                valid_files.append(f)
            except:
                pass
        self.file_list = valid_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        sample = np.load(file_path, allow_pickle=True).item()

        seq_len = sample["seq_len"]
        f_d = sample["f_d"]
        f_s = sample["f_s"]
        f_n = sample["f_n"]
        sl = sample["sl"]
        label = sample["label"]

        # 截断超长序列
        if seq_len > self.max_len_threshold:
            seq_len = self.max_len_threshold
            f_d = f_d[:seq_len]
            f_s = f_s[:seq_len]
            f_n = f_n[:seq_len]
            sl = sl[:seq_len]

        # 转换为tensor（保持在CPU）
        f_d = torch.from_numpy(f_d).float()
        f_s = torch.from_numpy(f_s).float()
        f_n = torch.from_numpy(f_n).float()
        sl = torch.from_numpy(sl).float()
        label = torch.from_numpy(label).float()

        return {
            "seq_len": seq_len,
            "f_d": f_d,
            "f_s": f_s,
            "f_n": f_n,
            "sl": sl,
            "label": label
        }

# 🔴 核心修复：collate_fn中不再移到CUDA，所有张量保留在CPU
def dynamic_collate_fn(batch: list):
    batch_seq_lens = [item["seq_len"] for item in batch]
    batch_max_len = min(max(batch_seq_lens), MAX_LEN_THRESHOLD)
    batch_size = len(batch)

    # 初始化张量（CPU）
    f_d_batch = torch.zeros((batch_size, batch_max_len, 1024), dtype=torch.float32)
    f_s_batch = torch.zeros((batch_size, batch_max_len, 301), dtype=torch.float32)
    f_n_batch = torch.zeros((batch_size, batch_max_len, 1024), dtype=torch.float32)
    sl_batch = torch.zeros((batch_size, batch_max_len), dtype=torch.float32)
    mask_batch = torch.zeros((batch_size, batch_max_len), dtype=torch.float32)
    label_batch = torch.zeros((batch_size, 1), dtype=torch.float32)
    seq_len_batch = torch.tensor(batch_seq_lens, dtype=torch.int32)

    # 动态填充（CPU）
    for i, item in enumerate(batch):
        seq_len = item["seq_len"]
        f_d_batch[i, :seq_len, :] = item["f_d"]
        f_s_batch[i, :seq_len, :] = item["f_s"]
        f_n_batch[i, :seq_len, :] = item["f_n"]
        sl_batch[i, :seq_len] = item["sl"]
        mask_batch[i, :seq_len] = 1.0
        label_batch[i] = item["label"]

    return {
        "seq_lens": seq_len_batch,
        "f_d": f_d_batch,       # CPU张量
        "f_s": f_s_batch,       # CPU张量
        "f_n": f_n_batch,       # CPU张量
        "sl": sl_batch,         # CPU张量
        "mask": mask_batch,     # CPU张量
        "label": label_batch    # CPU张量
    }

# 创建DataLoader
def create_dataloader(data_type: str = "train"):
    data_dir = os.path.join(DATA_ROOT, data_type)
    dataset = CDGLSSPVariableDataset(data_dir, MAX_LEN_THRESHOLD)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=(data_type == "train"),
        collate_fn=dynamic_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # pin_memory=True加速CPU->GPU传输
        drop_last=False
    )
    print(f"✅ {data_type}集DataLoader创建完成，样本数：{len(dataset)}")
    return dataloader

# ======================== 3. 定义CDG-LSSP模型（适配变长输入） ========================
class CDGLSSPModel(nn.Module):
    """
    CDG-LSSP模型简化实现（适配变长序列+动态mask）
    核心：融合f_d/f_s/f_n/sl特征，通过注意力+全连接层完成二分类
    """
    def __init__(self):
        super().__init__()
        # 特征投影层（统一维度）
        self.f_d_proj = nn.Linear(1024, 256)
        self.f_s_proj = nn.Linear(301, 256)
        self.f_n_proj = nn.Linear(1024, 256)
        self.sl_proj = nn.Linear(1, 256)  # sl是1维，扩展到256

        # 自注意力层（捕捉序列特征）
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        # 分类头
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_d, f_s, f_n, sl, mask):
        """
        前向传播：
        - f_d: [B, L, 1024]
        - f_s: [B, L, 301]
        - f_n: [B, L, 1024]
        - sl: [B, L]
        - mask: [B, L]（1=有效，0=无效）
        """
        # 1. 特征投影（统一到256维）
        f_d_feat = self.f_d_proj(f_d)  # [B, L, 256]
        f_s_feat = self.f_s_proj(f_s)  # [B, L, 256]
        f_n_feat = self.f_n_proj(f_n)  # [B, L, 256]
        sl_feat = self.sl_proj(sl.unsqueeze(-1))  # [B, L, 256]

        # 2. 特征融合
        fused_feat = f_d_feat + f_s_feat + f_n_feat + sl_feat  # [B, L, 256]
        fused_feat = self.dropout(fused_feat)

        # 3. 自注意力（mask屏蔽无效位置）
        attn_feat, _ = self.attention(fused_feat, fused_feat, fused_feat, key_padding_mask=(1 - mask).bool())
        attn_feat = self.relu(attn_feat)

        # 4. 序列聚合（取有效位置的均值）
        mask_expanded = mask.unsqueeze(-1).expand(attn_feat.size())  # [B, L, 256]
        sum_feat = torch.sum(attn_feat * mask_expanded, dim=1)  # [B, 256]
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)  # 避免除以0
        avg_feat = sum_feat / sum_mask  # [B, 256]

        # 5. 分类
        out = self.fc1(avg_feat)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # [B, 1]
        pred = self.sigmoid(out)  # [B, 1]（0~1概率）

        return pred

# ======================== 4. 训练&验证核心函数 ========================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """训练单个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    start_time = time.time()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
        # 🔴 核心修复：训练时手动将CPU张量移到CUDA
        f_d = batch["f_d"].to(DEVICE, non_blocking=True)
        f_s = batch["f_s"].to(DEVICE, non_blocking=True)
        f_n = batch["f_n"].to(DEVICE, non_blocking=True)
        sl = batch["sl"].to(DEVICE, non_blocking=True)
        mask = batch["mask"].to(DEVICE, non_blocking=True)
        label = batch["label"].to(DEVICE, non_blocking=True)

        # 前向传播
        optimizer.zero_grad()
        pred = model(f_d, f_s, f_n, sl, mask)
        loss = criterion(pred, label)

        # 反向传播+梯度裁剪
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        # 统计（移回CPU计算指标）
        total_loss += loss.item()
        all_preds.extend(pred.detach().cpu().numpy())
        all_labels.extend(label.detach().cpu().numpy())

        # 打印批次日志
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            batch_loss = total_loss / (batch_idx + 1)
            batch_acc = accuracy_score(np.round(all_labels), np.round(all_preds))
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {batch_loss:.4f} | ACC: {batch_acc:.4f}")

    # 计算epoch指标
    avg_loss = total_loss / len(train_loader)
    all_preds = np.array(all_preds).reshape(-1)
    all_labels = np.array(all_labels).reshape(-1)
    acc = accuracy_score(np.round(all_labels), np.round(all_preds))
    # 处理AUC计算（避免单类别）
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.0
    epoch_time = time.time() - start_time

    print(f"\n📊 Train Epoch {epoch+1} | Loss: {avg_loss:.4f} | ACC: {acc:.4f} | AUC: {auc:.4f} | Time: {epoch_time:.2f}s")
    return avg_loss, acc, auc

@torch.no_grad()
def validate(model, val_loader, criterion):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    start_time = time.time()

    for batch in tqdm(val_loader, desc="Validate"):
        # 🔴 核心修复：验证时手动将CPU张量移到CUDA
        f_d = batch["f_d"].to(DEVICE, non_blocking=True)
        f_s = batch["f_s"].to(DEVICE, non_blocking=True)
        f_n = batch["f_n"].to(DEVICE, non_blocking=True)
        sl = batch["sl"].to(DEVICE, non_blocking=True)
        mask = batch["mask"].to(DEVICE, non_blocking=True)
        label = batch["label"].to(DEVICE, non_blocking=True)

        # 前向传播（无梯度）
        pred = model(f_d, f_s, f_n, sl, mask)
        loss = criterion(pred, label)

        # 统计（移回CPU计算指标）
        total_loss += loss.item()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    # 计算验证指标
    avg_loss = total_loss / len(val_loader)
    all_preds = np.array(all_preds).reshape(-1)
    all_labels = np.array(all_labels).reshape(-1)
    acc = accuracy_score(np.round(all_labels), np.round(all_preds))
    # 处理AUC计算
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.0
    val_time = time.time() - start_time

    print(f"📊 Validate | Loss: {avg_loss:.4f} | ACC: {acc:.4f} | AUC: {auc:.4f} | Time: {val_time:.2f}s")
    return avg_loss, acc, auc

def write_log(log_str):
    """写入训练日志"""
    with open(LOG_SAVE_PATH, "a", encoding="utf-8") as f:
        f.write(log_str + "\n")
    print(log_str)

# ======================== 5. 主训练流程 ========================
def main_train():
    # 1. 创建数据加载器
    train_loader = create_dataloader("train")
    val_loader = create_dataloader("val")

    # 2. 初始化模型、损失函数、优化器
    model = CDGLSSPModel().to(DEVICE)
    criterion = nn.BCELoss()  # 二分类交叉熵
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 移除旧版本不支持的verbose参数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # 3. 初始化训练记录
    best_val_auc = 0.0
    best_val_epoch = 0
    early_stop_count = 0
    log_header = "Epoch,Train_Loss,Train_ACC,Train_AUC,Val_Loss,Val_ACC,Val_AUC,LR"
    write_log(log_header)

    # 4. 训练循环
    for epoch in range(EPOCHS):
        # 训练
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        # 验证
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
        # 学习率调整
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        # 记录日志（新增学习率变化提示）
        if epoch > 0 and current_lr < optimizer.param_groups[0]["lr"]:
            write_log(f"📉 学习率调整为：{current_lr:.6f}")

        # 记录训练日志
        log_str = f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{train_auc:.4f},{val_loss:.4f},{val_acc:.4f},{val_auc:.4f},{current_lr:.6f}"
        write_log(log_str)

        # 保存最佳模型（按AUC）
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_epoch = epoch + 1
            early_stop_count = 0
            # 保存最佳模型
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "train_loss": train_loss
            }, os.path.join(MODEL_SAVE_DIR, "best_model.pth"))
            write_log(f"🎉 保存最佳模型！Epoch {epoch+1} | Val AUC: {val_auc:.4f}")
        else:
            early_stop_count += 1
            write_log(f"⚠️  验证AUC未提升，早停计数：{early_stop_count}/{PATIENCE}")

        # 早停
        if early_stop_count >= PATIENCE:
            write_log(f"🛑 早停触发！最佳模型在Epoch {best_val_epoch}，Val AUC: {best_val_auc:.4f}")
            break

    # 保存最后一轮模型
    torch.save({
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, os.path.join(MODEL_SAVE_DIR, "last_model.pth"))

    # 训练完成总结
    write_log("\n========== 训练完成 ==========")
    write_log(f"最佳验证AUC：{best_val_auc:.4f}（Epoch {best_val_epoch}）")
    write_log(f"最后一轮训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f}")
    write_log(f"模型保存路径：{MODEL_SAVE_DIR}")

# ======================== 6. 加载模型并测试 ========================
def test_best_model():
    """加载最佳模型，测试验证集性能"""
    # 创建验证集加载器
    val_loader = create_dataloader("val")
    # 初始化模型
    model = CDGLSSPModel().to(DEVICE)
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, "best_model.pth"), map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    # 验证
    criterion = nn.BCELoss()
    val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
    # 打印结果
    print("\n========== 最佳模型测试结果 ==========")
    print(f"加载Epoch {checkpoint['epoch']}的模型")
    print(f"验证损失：{val_loss:.4f}")
    print(f"验证准确率：{val_acc:.4f}")
    print(f"验证AUC：{val_auc:.4f}")

# ======================== 主函数 ========================
if __name__ == "__main__":
    # 清空日志文件
    with open(LOG_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(f"训练开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最大序列长度：{MAX_LEN_THRESHOLD}\n")
        f.write(f"批次大小：{BATCH_SIZE}\n")
        f.write(f"初始学习率：{LEARNING_RATE}\n\n")
    
    # 执行训练
    main_train()
    
    # 测试最佳模型
    test_best_model()