import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ======================== 核心配置 ========================
# 数据根目录
DATA_ROOT = "./cdg_lssp_sim_data"
# 动态收集配置
BATCH_SIZE = 8               # 批次大小（可根据显存调整）
NUM_WORKERS = 4              # 多线程加载数（建议=CPU核心数）
MAX_LEN_THRESHOLD = 1600      # 动态长度阈值（超过则截断，防止显存溢出）
PIN_MEMORY = True            # 锁页内存（加速GPU数据传输）
SHUFFLE = True               # 训练集是否打乱（验证集设为False）

# ======================== 1. 定义变长数据集类 ========================
class CDGLSSPVariableDataset(Dataset):
    """
    变长序列数据集类（加载无mask的.npy变长数据）
    支持动态加载、数据校验、异常处理
    """
    def __init__(self, data_dir: str, max_len_threshold: int = 256):
        self.data_dir = data_dir
        self.max_len_threshold = max_len_threshold
        # 过滤并排序数据文件
        self.file_list = [
            f for f in os.listdir(data_dir) 
            if f.endswith(".npy") and f.split("_")[0] in ["train", "val"]
        ]
        # 按数字排序（train_0, train_1...）
        self.file_list.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        # 预校验数据有效性（可选，耗时但能提前发现坏数据）
        self._validate_dataset()
        
    def _validate_dataset(self):
        """预校验数据集，过滤损坏文件"""
        valid_files = []
        for f in tqdm(self.file_list, desc="预校验数据集"):
            file_path = os.path.join(self.data_dir, f)
            try:
                sample = np.load(file_path, allow_pickle=True).item()
                # 校验核心字段和长度
                assert "seq_len" in sample, f"缺失seq_len字段：{f}"
                assert "f_d" in sample, f"缺失f_d字段：{f}"
                assert sample["f_d"].shape[0] == sample["seq_len"], f"f_d长度不匹配：{f}"
                valid_files.append(f)
            except Exception as e:
                print(f"跳过损坏文件 {f}：{str(e)}")
        self.file_list = valid_files
        print(f"数据集校验完成，有效文件数：{len(self.file_list)}")
    
    def __len__(self):
        """返回数据集总长度"""
        return len(self.file_list)
    
    def __getitem__(self, idx: int):
        """
        动态加载单条数据（核心）
        返回：未Padding的原始变长张量 + 序列长度
        """
        # 加载数据文件
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        sample = np.load(file_path, allow_pickle=True).item()
        
        # 1. 提取核心字段
        seq_len = sample["seq_len"]
        f_d = sample["f_d"]
        f_s = sample["f_s"]
        f_n = sample["f_n"]
        sl = sample["sl"]
        label = sample["label"]
        
        # 2. 动态截断超长序列（防止显存溢出）
        if seq_len > self.max_len_threshold:
            seq_len = self.max_len_threshold
            f_d = f_d[:seq_len]
            f_s = f_s[:seq_len]
            f_n = f_n[:seq_len]
            sl = sl[:seq_len]
        
        # 3. 转换为torch张量（float32，适配模型）
        f_d = torch.from_numpy(f_d).float()
        f_s = torch.from_numpy(f_s).float()
        f_n = torch.from_numpy(f_n).float()
        sl = torch.from_numpy(sl).float()
        label = torch.from_numpy(label).float()
        
        # 4. 返回原始数据（不做Padding，交给collate_fn动态处理）
        return {
            "file_name": file_name,    # 可选：记录文件名，方便调试
            "seq_len": seq_len,        # 序列长度
            "f_d": f_d,                # [seq_len, 1024]
            "f_s": f_s,                # [seq_len, 301]
            "f_n": f_n,                # [seq_len, 1024]
            "sl": sl,                  # [seq_len]
            "label": label             # [1]
        }

# ======================== 2. 定义动态Collate函数（核心） ========================
def dynamic_collate_fn(batch: list):
    """
    动态收集函数（Dynamic Collation）
    功能：
    1. 动态计算当前批次的最大长度
    2. 动态Padding到该长度（仅Pad到批次最大，而非全局最大）
    3. 动态生成mask（1=有效，0=无效）
    4. 保持张量维度一致，适配模型输入
    """
    # 1. 提取批次内所有样本的序列长度，动态计算批次最大长度
    batch_seq_lens = [item["seq_len"] for item in batch]
    batch_max_len = max(batch_seq_lens)
    batch_size = len(batch)
    
    # 2. 初始化动态Padding后的张量容器
    # 维度：[batch_size, batch_max_len, feat_dim]
    f_d_batch = torch.zeros((batch_size, batch_max_len, 1024), dtype=torch.float32)
    f_s_batch = torch.zeros((batch_size, batch_max_len, 301), dtype=torch.float32)
    f_n_batch = torch.zeros((batch_size, batch_max_len, 1024), dtype=torch.float32)
    sl_batch = torch.zeros((batch_size, batch_max_len), dtype=torch.float32)
    mask_batch = torch.zeros((batch_size, batch_max_len), dtype=torch.float32)  # 动态mask
    label_batch = torch.zeros((batch_size, 1), dtype=torch.float32)
    seq_len_batch = torch.tensor(batch_seq_lens, dtype=torch.int32)
    file_names = []
    
    # 3. 动态填充每个样本（仅Pad到批次最大长度）
    for i, item in enumerate(batch):
        seq_len = item["seq_len"]
        # 填充特征（有效部分保留，无效部分Pad 0）
        f_d_batch[i, :seq_len, :] = item["f_d"]
        f_s_batch[i, :seq_len, :] = item["f_s"]
        f_n_batch[i, :seq_len, :] = item["f_n"]
        sl_batch[i, :seq_len] = item["sl"]
        # 动态生成mask：有效位置=1，无效位置=0
        mask_batch[i, :seq_len] = 1.0
        # 填充标签和文件名
        label_batch[i] = item["label"]
        file_names.append(item["file_name"])
    
    # 4. 返回动态收集后的批次数据
    return {
        "file_names": file_names,       # 可选：调试用
        "seq_lens": seq_len_batch,      # 每个样本的原始长度
        "batch_max_len": batch_max_len, # 当前批次的最大长度（动态）
        "f_d": f_d_batch,               # [B, L, 1024]
        "f_s": f_s_batch,               # [B, L, 301]
        "f_n": f_n_batch,               # [B, L, 1024]
        "sl": sl_batch,                 # [B, L]
        "mask": mask_batch,             # [B, L] 动态生成的mask
        "label": label_batch            # [B, 1]
    }

# ======================== 3. 封装动态DataLoader生成函数 ========================
def create_dynamic_dataloader(
    data_type: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    创建动态数据加载器
    参数：
        data_type: "train" / "val"，选择训练集/验证集
        batch_size: 批次大小
        shuffle: 是否打乱（训练集True，验证集False）
        num_workers: 多线程数
        pin_memory: 是否使用锁页内存
    返回：
        配置好的动态DataLoader
    """
    # 确定数据目录
    data_dir = os.path.join(DATA_ROOT, data_type)
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在：{data_dir}")
    
    # 创建数据集
    dataset = CDGLSSPVariableDataset(
        data_dir=data_dir,
        max_len_threshold=MAX_LEN_THRESHOLD
    )
    
    # 创建动态DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dynamic_collate_fn,  # 核心：动态收集函数
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # 不丢弃最后一个不完整批次
        prefetch_factor=2  # 预取批次，提升加载效率
    )
    
    print(f"✅ 动态DataLoader创建完成（{data_type}集）")
    print(f"   - 批次大小：{batch_size}")
    print(f"   - 样本总数：{len(dataset)}")
    print(f"   - 批次总数：{len(dataloader)}")
    print(f"   - 多线程数：{num_workers}")
    print(f"   - 动态长度阈值：{MAX_LEN_THRESHOLD}")
    
    return dataloader

# ======================== 4. 测试动态数据收集 ========================
def test_dynamic_collation():
    """测试动态收集效果，验证每个批次的动态长度和维度"""
    # 创建训练集动态DataLoader
    train_loader = create_dynamic_dataloader(
        data_type="train",
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # 创建验证集动态DataLoader（不打乱）
    val_loader = create_dynamic_dataloader(
        data_type="val",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # 遍历训练集前3个批次，验证动态收集效果
    print("\n===================== 测试训练集动态收集 ======================")
    for idx, batch in enumerate(train_loader):
        if idx >= 3:  # 只看前3个批次
            break
        print(f"\n【训练集批次 {idx+1}】")
        print(f"  批次文件名：{batch['file_names']}")
        print(f"  批次最大长度：{batch['batch_max_len']}")
        print(f"  各样本原始长度：{batch['seq_lens'].numpy()}")
        print(f"  f_d维度：{batch['f_d'].shape} (B={BATCH_SIZE}, L={batch['batch_max_len']}, D=1024)")
        print(f"  mask维度：{batch['mask'].shape} (B={BATCH_SIZE}, L={batch['batch_max_len']})")
        print(f"  label维度：{batch['label'].shape}")
        # 验证mask有效性（有效位置和原始长度一致）
        for i in range(BATCH_SIZE):
            seq_len = batch['seq_lens'][i].item()
            mask_sum = batch['mask'][i, :seq_len].sum().item()
            assert mask_sum == seq_len, f"mask生成错误：样本{i}有效长度{seq_len}，mask有效数{mask_sum}"
    print("✅ 训练集动态收集测试通过！")
    
    # 遍历验证集1个批次
    print("\n===================== 测试验证集动态收集 ======================")
    batch = next(iter(val_loader))
    print(f"【验证集批次】")
    print(f"  批次最大长度：{batch['batch_max_len']}")
    print(f"  f_s维度：{batch['f_s'].shape}")
    print(f"  sl维度：{batch['sl'].shape}")
    print("✅ 验证集动态收集测试通过！")

# ======================== 主函数 ========================
if __name__ == "__main__":
    # 设置CUDA（如果可用）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🔧 使用GPU：{torch.cuda.get_device_name(0)}")
    else:
        print(f"🔧 使用CPU")
    
    # 测试动态数据收集
    test_dynamic_collation()