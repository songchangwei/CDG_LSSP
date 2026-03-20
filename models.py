import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import LayerNorm

# ======================== 全局参数配置（与论文保持一致） ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_MAX = 1600  # 最大句子序列长度
FEAT_DIM_UNIFIED = 256  # 特征统一维度
GCN_LAYERS = 2  # GCN层数
TOP_K = 2  # 扩展边top-k
DROPOUT_RATE = 0.2  # dropout率
ATTN_QK_DIM = 64  # 注意力Q/K维度
ATTN_V_DIM = FEAT_DIM_UNIFIED  # 注意力V维度
FFN_HIDDEN = 512  # FFN隐藏层维度
CLASSIFIER_HIDDEN = 512  # 分类器隐藏层维度
EPS = 1e-7  # 数值稳定性常量，避免0/1极端值

# ======================== 基础组件 ========================
class PositionalEncoding(nn.Module):
    """正弦余弦位置编码（仅对有效位置编码）"""
    def __init__(self, d_model, max_len=T_MAX):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 不参与训练

    def forward(self, x, mask):
        """
        x: [B, T, D] 特征序列
        mask: [B, T] 有效位置mask（1=有效，0=无效）
        return: [B, T, D] 加位置编码后的特征
        """
        pe = self.pe[:x.size(1)].transpose(0, 1)  # [1, T, D]
        x = x + pe * mask.unsqueeze(-1)  # 仅有效位置加编码
        return x

class FeatureUnifier(nn.Module):
    """特征维度统一模块（1×1Conv1D+PE+LayerNorm）"""
    def __init__(self, in_dim):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=FEAT_DIM_UNIFIED, 
                                kernel_size=1, stride=1, padding=0)  # 1×1卷积
        self.pe = PositionalEncoding(FEAT_DIM_UNIFIED)
        self.norm = LayerNorm(FEAT_DIM_UNIFIED)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x, mask):
        """
        x: [B, T, in_dim] 原始视图特征
        mask: [B, T] 有效位置mask
        return: [B, T, FEAT_DIM_UNIFIED] 统一维度后的特征
        """
        x = x.transpose(1, 2)  # [B, in_dim, T] -> Conv1D输入格式
        x = self.conv1d(x).transpose(1, 2)  # [B, T, FEAT_DIM_UNIFIED]
        x = self.pe(x, mask)  # 加位置编码
        x = self.norm(x)
        x = self.dropout(x)
        return x

class GCNLayer(nn.Module):
    """基础GCN层（图卷积+ELU+LayerNorm）"""
    def __init__(self, in_dim=FEAT_DIM_UNIFIED, out_dim=FEAT_DIM_UNIFIED):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.elu = nn.ELU()
        self.norm = LayerNorm(out_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x, adj):
        """
        x: [B, T, D] 节点特征
        adj: [B, T, T] 归一化后的邻接矩阵
        return: [B, T, D] 图卷积后特征
        """
        x = torch.bmm(adj, x)  # 邻接矩阵乘节点特征：B*T*T @ B*T*D = B*T*D
        x = self.linear(x)
        x = self.elu(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

# ======================== 核心模块 ========================
class NEGCN(nn.Module):
    """NEGCN模块（D/S视图，负情绪软标签引导的图卷积）"""
    def __init__(self):
        super().__init__()
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(GCN_LAYERS)])
        self.eps = 1e-6  # 避免除零

    def cal_cos_sim(self, x, mask):
        """计算节点余弦相似度，屏蔽无效位置"""
        x_norm = F.normalize(x, p=2, dim=-1)  # [B, T, D]
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, T, T]
        mask_mat = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, T, T]
        sim = sim * mask_mat
        return sim

    def cal_sl_sim(self, sl, mask):
        """计算负情绪软标签相似度"""
        # 关键修复1：裁剪软标签到[0,1]
        sl = torch.clamp(sl, min=0.0, max=1.0)
        sl = sl.unsqueeze(1)  # [B, 1, T]
        sl_i = sl.expand(-1, sl.size(2), -1)  # [B, T, T]
        sl_j = sl.transpose(1, 2)  # [B, T, T]
        sl_diff = 1 - torch.abs(sl_i - sl_j)
        sl_sq = (sl_i ** 2 + sl_j ** 2) / 2
        sl_sim = sl_diff * sl_sq
        mask_mat = mask.unsqueeze(1) * mask.unsqueeze(2)
        sl_sim = sl_sim * mask_mat
        return sl_sim

    def build_adj(self, sim_m, sl_sim, mask):
        """构建自适应邻接矩阵（时间基础边+TOP-K扩展边）"""
        B, T, _ = sim_m.shape
        # 综合相关性
        corel = 0.5 * sim_m + 0.5 * sl_sim  # [B, T, T]
        # 时间基础边
        base_adj = torch.eye(T, device=DEVICE).unsqueeze(0).repeat(B, 1, 1)  # 自环
        for i in range(T-1):
            base_adj[:, i, i+1] = 1.0
            base_adj[:, i+1, i] = 1.0
        # TOP-K扩展边
        ext_adj = torch.zeros_like(corel)
        corel = corel.masked_fill(base_adj == 1, -1.0)  # 排除基础边节点
        topk_vals, topk_inds = torch.topk(corel, k=TOP_K, dim=-1)  # [B, T, K]
        for b in range(B):
            for i in range(T):
                if mask[b, i] == 0:
                    continue
                for j in topk_inds[b, i]:
                    if mask[b, j] == 1:
                        ext_adj[b, i, j] = 1.0
                        ext_adj[b, j, i] = 1.0
        # 合并+加权+归一化
        adj = base_adj + ext_adj
        adj = adj.clamp(0, 1) * corel
        adj = adj + torch.eye(T, device=DEVICE).unsqueeze(0).repeat(B, 1, 1)  # 加自环
        row_sum = adj.sum(dim=-1, keepdim=True) + self.eps
        col_sum = adj.sum(dim=1, keepdim=True) + self.eps
        adj = adj / torch.sqrt(row_sum * col_sum)  # 对称归一化
        # 屏蔽无效位置
        mask_mat = mask.unsqueeze(1) * mask.unsqueeze(2)
        adj = adj * mask_mat
        return adj

    def forward(self, x, sl, mask):
        """
        x: [B, T, FEAT_DIM_UNIFIED] 统一维度后的D/S视图特征
        sl: [B, T] 负情绪软标签
        mask: [B, T] 有效位置mask
        return: [B, T, FEAT_DIM_UNIFIED] NEGCN增强后的特征
        """
        cos_sim = self.cal_cos_sim(x, mask)
        sl_sim = self.cal_sl_sim(sl, mask)
        adj = self.build_adj(cos_sim, sl_sim, mask)
        for gcn in self.gcn_layers:
            x = gcn(x, adj)
        return x

class VanillaGCN(nn.Module):
    """常规GCN模块（N视图，仅声学特征相似度）"""
    def __init__(self):
        super().__init__()
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(GCN_LAYERS)])
        self.eps = 1e-6

    def cal_cos_sim(self, x, mask):
        """计算余弦相似度"""
        x_norm = F.normalize(x, p=2, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
        mask_mat = mask.unsqueeze(1) * mask.unsqueeze(2)
        sim = sim * mask_mat
        return sim

    def build_adj(self, sim_m, mask):
        """构建图（时间基础边+TOP-K扩展边）"""
        B, T, _ = sim_m.shape
        # 时间基础边
        base_adj = torch.eye(T, device=DEVICE).unsqueeze(0).repeat(B, 1, 1)
        for i in range(T-1):
            base_adj[:, i, i+1] = 1.0
            base_adj[:, i+1, i] = 1.0
        # TOP-K扩展边
        ext_adj = torch.zeros_like(sim_m)
        corel = sim_m.masked_fill(torch.eye(T, device=DEVICE).unsqueeze(0).repeat(B,1,1)==1, -1.0)
        topk_vals, topk_inds = torch.topk(corel, k=TOP_K, dim=-1)
        for b in range(B):
            for i in range(T):
                if mask[b, i] == 0:
                    continue
                for j in topk_inds[b, i]:
                    if mask[b, j] == 1:
                        ext_adj[b, i, j] = 1.0
                        ext_adj[b, j, i] = 1.0
        # 合并+加权+归一化
        adj = base_adj + ext_adj
        adj = adj.clamp(0, 1) * sim_m
        adj = adj + torch.eye(T, device=DEVICE).unsqueeze(0).repeat(B,1,1)
        row_sum = adj.sum(dim=-1, keepdim=True) + self.eps
        col_sum = adj.sum(dim=1, keepdim=True) + self.eps
        adj = adj / torch.sqrt(row_sum * col_sum)
        # 屏蔽无效位置
        mask_mat = mask.unsqueeze(1) * mask.unsqueeze(2)
        adj = adj * mask_mat
        return adj

    def forward(self, x, mask):
        """
        x: [B, T, FEAT_DIM_UNIFIED] 统一维度后的N视图特征
        mask: [B, T] 有效位置mask
        return: [B, T, FEAT_DIM_UNIFIED] GCN增强后的特征
        """
        cos_sim = self.cal_cos_sim(x, mask)
        adj = self.build_adj(cos_sim, mask)
        for gcn in self.gcn_layers:
            x = gcn(x, adj)
        return x

class AttentionBlock(nn.Module):
    """注意力模块（自注意力/交叉注意力+FFN+残差）"""
    def __init__(self):
        super().__init__()
        # 注意力投影层
        self.w_q = nn.Linear(FEAT_DIM_UNIFIED, ATTN_QK_DIM)
        self.w_k = nn.Linear(FEAT_DIM_UNIFIED, ATTN_QK_DIM)
        self.w_v = nn.Linear(FEAT_DIM_UNIFIED, ATTN_V_DIM)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(FEAT_DIM_UNIFIED, FFN_HIDDEN),
            nn.GELU(),
            nn.Linear(FFN_HIDDEN, FEAT_DIM_UNIFIED),
            nn.Dropout(DROPOUT_RATE)
        )
        self.norm1 = LayerNorm(FEAT_DIM_UNIFIED)
        self.norm2 = LayerNorm(FEAT_DIM_UNIFIED)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.eps = 1e-6

    def self_attn(self, x, mask):
        """自注意力"""
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        attn_score = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(ATTN_QK_DIM)
        mask_mat = mask.unsqueeze(1).repeat(1, x.size(1), 1)
        attn_score = attn_score.masked_fill(mask_mat == 0, -1e9)
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_out = torch.bmm(attn_prob, v)
        return self.dropout(attn_out)

    def cross_attn(self, q_x, k_v_x, q_mask, k_v_mask):
        """交叉注意力"""
        q = self.w_q(q_x)
        k = self.w_k(k_v_x)
        v = self.w_v(k_v_x)
        attn_score = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(ATTN_QK_DIM)
        q_mask_mat = q_mask.unsqueeze(1).repeat(1, q_x.size(1), 1)
        k_v_mask_mat = k_v_mask.unsqueeze(1).repeat(1, q_x.size(1), 1)
        mask_mat = q_mask_mat * k_v_mask_mat
        attn_score = attn_score.masked_fill(mask_mat == 0, -1e9)
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_out = torch.bmm(attn_prob, v)
        return self.dropout(attn_out)

    def forward(self, x, mask, cross_x=None, cross_mask=None):
        """
        自注意力：仅输入x, mask
        交叉注意力：输入x(q), cross_x(k/v), mask(q), cross_mask(k/v)
        return: [B, T, FEAT_DIM_UNIFIED] 注意力输出
        """
        residual = x
        if cross_x is None:
            attn_out = self.self_attn(x, mask)
        else:
            attn_out = self.cross_attn(x, cross_x, mask, cross_mask)
        x = self.norm1(residual + attn_out)
        residual = x
        ffn_out = self.ffn(x)
        x = self.norm2(residual + ffn_out)
        return x

class NEKI(nn.Module):
    """NEKI模块（负情绪知识注入）"""
    def __init__(self):
        super().__init__()
        self.n_proj = nn.ModuleList([nn.Linear(FEAT_DIM_UNIFIED, FEAT_DIM_UNIFIED) for _ in range(4)])
        self.attn = AttentionBlock()
        self.norm = LayerNorm(FEAT_DIM_UNIFIED)

    def forward(self, x_list, f_n, sl, mask):
        """
        x_list: [4, B, T, D] 跨视图基础融合特征
        f_n: [B, T, D] GCN增强后的N视图特征
        sl: [B, T] 负情绪软标签
        mask: [B, T] 有效位置mask
        return: [4, B, T, D] 知识注入后的特征
        """
        # 关键修复2：再次校验软标签
        sl = torch.clamp(sl, min=0.0, max=1.0)
        B, T, D = f_n.shape
        inj_x_list = []
        sl = sl.unsqueeze(-1)
        for i, x in enumerate(x_list):
            f_n_proj = self.n_proj[i](f_n)
            self_attn_x = self.attn(x, mask)
            cross_attn_x = self.attn(x, mask, cross_x=f_n_proj, cross_mask=mask)
            inj_x = (1 - sl) * self_attn_x + sl * cross_attn_x
            inj_x = self.norm(inj_x)
            inj_x_list.append(inj_x)
        return inj_x_list

class NEGAPooling(nn.Module):
    """NEGA Pooling模块（负情绪引导的注意力池化）"""
    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear(FEAT_DIM_UNIFIED, ATTN_QK_DIM)
        self.w_k = nn.Linear(FEAT_DIM_UNIFIED, ATTN_QK_DIM)
        self.w_v = nn.Linear(FEAT_DIM_UNIFIED, ATTN_V_DIM)
        self.eps = 1e-6

    def forward(self, x, sl, mask):
        """
        x: [B, T, D] 知识注入后的特征
        sl: [B, T] 负情绪软标签
        mask: [B, T] 有效位置mask
        return: [B, D] 全局池化特征
        """
        # 关键修复3：校验软标签
        sl = torch.clamp(sl, min=0.0, max=1.0)
        B, T, D = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        attn_score = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(ATTN_QK_DIM)
        # 融合软标签先验
        sl_j = sl.unsqueeze(1).repeat(1, T, 1)
        attn_score = attn_score * (1 + sl_j ** 2)
        # 屏蔽无效位置
        mask_mat = mask.unsqueeze(1).repeat(1, T, 1)
        attn_score = attn_score.masked_fill(mask_mat == 0, -1e9)
        attn_prob = F.softmax(attn_score, dim=-1)
        # 全局池化
        global_feat = torch.bmm(attn_prob, v).mean(dim=1)
        return global_feat

class Classifier(nn.Module):
    """分类器模块（改用BCEWithLogitsLoss，数值更稳定）"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(FEAT_DIM_UNIFIED * 4, CLASSIFIER_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(CLASSIFIER_HIDDEN, 1)  # 移除Sigmoid，交给BCEWithLogitsLoss
        )
        # 关键修复4：改用BCEWithLogitsLoss（合并Sigmoid+Loss，避免数值溢出）
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, feat_list, label=None):
        """
        feat_list: [4, B, D] 4个视图的全局池化特征
        label: [B, 1] 真实标签（可选）
        return: pred [B,1] 预测概率；loss（可选）
        """
        feat_concat = torch.cat(feat_list, dim=-1)
        logits = self.fc(feat_concat)  # 输出logits（未经过Sigmoid）
        
        # 推理时转换为概率（训练时不需要，Loss内部会处理）
        if label is None:
            pred = torch.sigmoid(logits)
            # 关键修复5：裁剪概率到[EPS, 1-EPS]，避免0/1
            pred = torch.clamp(pred, min=EPS, max=1-EPS)
            return pred
        
        # 训练时计算损失（直接用logits）
        loss = self.loss_fn(logits, label.float())
        # 同时返回概率（用于打印）
        pred = torch.sigmoid(logits)
        pred = torch.clamp(pred, min=EPS, max=1-EPS)
        return pred, loss

# ======================== 整体CDG-LSSP模型 ========================
class CDG_LSSP(nn.Module):
    """完整的CDG-LSSP模型"""
    def __init__(self):
        super().__init__()
        # 特征维度统一
        self.unifier_D = FeatureUnifier(in_dim=1024)
        self.unifier_S = FeatureUnifier(in_dim=301)
        self.unifier_N = FeatureUnifier(in_dim=1024)
        # 图卷积
        self.negcn_D = NEGCN()
        self.negcn_S = NEGCN()
        self.gcn_N = VanillaGCN()
        # 跨视图注意力
        self.attn_D = AttentionBlock()
        self.attn_S = AttentionBlock()
        self.cross_attn_D2S = AttentionBlock()
        self.cross_attn_S2D = AttentionBlock()
        # NEKI
        self.neki = NEKI()
        # NEGA Pooling
        self.nega_pool = NEGAPooling()
        # 分类器
        self.classifier = Classifier()

    def create_mask(self, x):
        """生成变长序列mask"""
        B, T, _ = x.shape
        mask = torch.ones(B, T, device=DEVICE)
        return mask
    
    def check_nan_inf(self, tensor, name):
        """调试用：检查张量是否有NaN/Inf"""
        if torch.isnan(tensor).any():
            print(f"⚠️ 张量 {name} 包含NaN值！")
        if torch.isinf(tensor).any():
            print(f"⚠️ 张量 {name} 包含Inf值！")

    def forward(self, f_d, f_s, f_n, sl, label=None):
        """
        输入：
            f_d: [B, T, 1024] D视图特征
            f_s: [B, T, 301]  S视图特征
            f_n: [B, T, 1024] N视图特征
            sl:  [B, T]      负情绪软标签
            label: [B,1]     真实标签（训练时用）
        输出：
            pred: [B,1] 预测概率；loss（训练时返回）
        """
        # 关键修复6：输入特征校验（调试用）
        self.check_nan_inf(f_d, "f_d")
        self.check_nan_inf(f_s, "f_s")
        self.check_nan_inf(f_n, "f_n")
        self.check_nan_inf(sl, "sl")
        
        # 生成mask
        mask = self.create_mask(f_d)
        # 特征统一
        f_d_unif = self.unifier_D(f_d, mask)
        f_s_unif = self.unifier_S(f_s, mask)
        f_n_unif = self.unifier_N(f_n, mask)
        # 图卷积增强
        f_d_gcn = self.negcn_D(f_d_unif, sl, mask)
        f_s_gcn = self.negcn_S(f_s_unif, sl, mask)
        f_n_gcn = self.gcn_N(f_n_unif, mask)
        # 跨视图基础融合
        f_d_self = self.attn_D(f_d_gcn, mask)
        f_s_self = self.attn_S(f_s_gcn, mask)
        f_d2s = self.cross_attn_D2S(f_d_gcn, mask, cross_x=f_s_gcn, cross_mask=mask)
        f_s2d = self.cross_attn_S2D(f_s_gcn, mask, cross_x=f_d_gcn, cross_mask=mask)
        base_x_list = [f_d_self, f_s_self, f_d2s, f_s2d]
        # NEKI知识注入
        inj_x_list = self.neki(base_x_list, f_n_gcn, sl, mask)
        # NEGA Pooling
        global_feat_list = [self.nega_pool(x, sl, mask) for x in inj_x_list]
        # 分类/损失
        if label is None:
            pred = self.classifier(global_feat_list)
            return pred
        else:
            pred, loss = self.classifier(global_feat_list, label)
            return pred, loss

# ======================== 模型使用示例 ========================
if __name__ == "__main__":
    # 关键修复7：启用CUDA同步，定位具体错误
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # 初始化模型
    model = CDG_LSSP().to(DEVICE)
    # 生成模拟输入（替换为实际提取的特征）
    B = 4  # 批次大小
    T = 64 # 句子数（≤T_MAX=256）
    f_d = torch.randn(B, T, 1024).to(DEVICE)
    f_s = torch.randn(B, T, 301).to(DEVICE)
    f_n = torch.randn(B, T, 1024).to(DEVICE)
    sl = torch.rand(B, T).to(DEVICE)  # 软标签[0,1]
    label = torch.randint(0, 2, (B, 1)).to(DEVICE)  # 真实标签

    # 训练模式
    model.train()
    pred, loss = model(f_d, f_s, f_n, sl, label)
    print(f"训练模式 - 预测概率：{pred.squeeze().detach().cpu().numpy()}, 损失：{loss.item():.4f}")

    # 推理模式
    model.eval()
    with torch.no_grad():
        pred = model(f_d, f_s, f_n, sl)
    print(f"推理模式 - 预测概率：{pred.squeeze().detach().cpu().numpy()}")

    # 模型保存/加载
    torch.save(model.state_dict(), "cdg_lssp_model.pth")
    model.load_state_dict(torch.load("cdg_lssp_model.pth", map_location=DEVICE))