import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. 配置
# =========================================================
@dataclass
class Config:
    seq_len: int = 20
    pred_horizon: int = 5
    batch_size: int = 32
    hidden_dim: int = 64
    num_heads: int = 4
    temporal_layers: int = 2
    gnn_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 2. 示例数据格式说明
# =========================================================
"""
原始长表建议格式：

date, commodity, close, volume, oi
2020-01-02, AU, ...
2020-01-02, CU, ...
2020-01-03, AU, ...
...

你需要把它整理成 panel:
X:    [T, N, F]
mask: [T, N]      1表示该商品该日有数据，0表示无数据
close_panel: [T, N] 用于构造未来5日max/min收益率标签
"""


# =========================================================
# 3. 数据预处理
# =========================================================
def build_panels(
    df: pd.DataFrame,
    date_col: str = "date",
    commodity_col: str = "commodity",
    close_col: str = "close",
    volume_col: str = "volume",
    oi_col: str = "oi",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    """
    返回:
    features: [T, N, F]
    mask:     [T, N]
    close_panel: [T, N]
    dates:    长度T
    commodities: 长度N
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([commodity_col, date_col])

    dates = sorted(df[date_col].unique())
    commodities = sorted(df[commodity_col].unique())

    date_to_idx = {d: i for i, d in enumerate(dates)}
    com_to_idx = {c: i for i, c in enumerate(commodities)}

    T = len(dates)
    N = len(commodities)

    close_panel = np.full((T, N), np.nan, dtype=np.float32)
    volume_panel = np.full((T, N), np.nan, dtype=np.float32)
    oi_panel = np.full((T, N), np.nan, dtype=np.float32)

    for row in df.itertuples(index=False):
        d = getattr(row, date_col)
        c = getattr(row, commodity_col)
        t = date_to_idx[d]
        n = com_to_idx[c]

        close_panel[t, n] = float(getattr(row, close_col))
        volume_panel[t, n] = float(getattr(row, volume_col))
        oi_panel[t, n] = float(getattr(row, oi_col))

    mask = (~np.isnan(close_panel)).astype(np.float32)

    # 核心特征：收益率 / 成交量变化率 / 持仓变化率
    ret = np.full_like(close_panel, np.nan)
    vol_chg = np.full_like(volume_panel, np.nan)
    oi_chg = np.full_like(oi_panel, np.nan)

    # 按商品算
    for n in range(N):
        cp = pd.Series(close_panel[:, n])
        vp = pd.Series(volume_panel[:, n])
        op = pd.Series(oi_panel[:, n])

        ret[:, n] = cp.pct_change().values.astype(np.float32)
        vol_chg[:, n] = vp.pct_change().replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
        oi_chg[:, n] = op.pct_change().replace([np.inf, -np.inf], np.nan).values.astype(np.float32)

    features = np.stack([ret, vol_chg, oi_chg], axis=-1)  # [T, N, 3]

    # 用0填充缺失，真正有效性靠mask控制
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    close_panel = np.nan_to_num(close_panel, nan=0.0)

    return features, mask, close_panel, dates, commodities


def compute_targets(close_panel: np.ndarray, mask: np.ndarray, pred_horizon: int = 5):
    """
    计算未来5日最高/最低收益率:
    y_max[t, n] = max(close[t+1:t+H] / close[t] - 1)
    y_min[t, n] = min(close[t+1:t+H] / close[t] - 1)

    返回:
    y: [T, N, 2]
    y_mask: [T, N]   该位置是否有有效标签
    """
    T, N = close_panel.shape
    y = np.zeros((T, N, 2), dtype=np.float32)
    y_mask = np.zeros((T, N), dtype=np.float32)

    for t in range(T):
        if t + pred_horizon >= T:
            continue
        future_slice = close_panel[t + 1:t + 1 + pred_horizon]  # [H, N]
        current = close_panel[t]  # [N]

        valid_current = (mask[t] > 0) & (current > 0)

        for n in range(N):
            if not valid_current[n]:
                continue

            future_prices = future_slice[:, n]
            future_valid = future_prices > 0
            if future_valid.sum() == 0:
                continue

            future_returns = future_prices[future_valid] / current[n] - 1.0
            y[t, n, 0] = future_returns.max()
            y[t, n, 1] = future_returns.min()
            y_mask[t, n] = 1.0

    return y, y_mask


def build_adjacency_from_correlation(
    close_panel: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.15,
) -> np.ndarray:
    """
    用历史收益率相关性构建简单图。
    """
    T, N = close_panel.shape
    ret = np.zeros((T, N), dtype=np.float32)

    for n in range(N):
        s = pd.Series(close_panel[:, n])
        ret[:, n] = s.pct_change().fillna(0.0).values

    corr = np.corrcoef(ret.T)
    corr = np.nan_to_num(corr, nan=0.0)

    adj = np.where(np.abs(corr) >= threshold, np.abs(corr), 0.0).astype(np.float32)

    # 加自环
    np.fill_diagonal(adj, 1.0)

    # 归一化 A_hat = D^{-1/2} A D^{-1/2}
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(np.clip(deg, 1e-8, None), -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    return adj_norm.astype(np.float32)


# =========================================================
# 4. 数据集
# =========================================================
class CommodityDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,   # [T, N, F]
        mask: np.ndarray,       # [T, N]
        targets: np.ndarray,    # [T, N, 2]
        y_mask: np.ndarray,     # [T, N]
        seq_len: int,
        start_idx: int,
        end_idx: int,
    ):
        """
        取样本时刻 t:
        输入  [t-seq_len+1, ..., t]
        标签  targets[t]
        """
        self.X = features
        self.mask = mask
        self.y = targets
        self.y_mask = y_mask
        self.seq_len = seq_len

        self.indices = []
        for t in range(start_idx, end_idx):
            if t - seq_len + 1 < 0:
                continue
            if self.y_mask[t].sum() == 0:
                continue
            self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.X[t - self.seq_len + 1:t + 1]       # [L, N, F]
        x_mask = self.mask[t - self.seq_len + 1:t + 1]  # [L, N]
        y = self.y[t]                                # [N, 2]
        y_mask = self.y_mask[t]                      # [N]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(x_mask, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(y_mask, dtype=torch.float32),
        )


# =========================================================
# 5. 模型组件
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1).unsqueeze(1)  # [L,1,1,D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [L, B, N, D]
        L = x.size(0)
        return x + self.pe[:L]


class TemporalTransformerBlock(nn.Module):
    """
    对每个商品，沿时间维做 self-attention
    输入: [B, L, N, D]
    输出: [B, L, N, D]
    """
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask=None):
        """
        x: [B, L, N, D]
        valid_mask: [B, L, N]  1有效, 0无效
        """
        B, L, N, D = x.shape

        # reshape 成 [B*N, L, D]
        x2 = x.permute(0, 2, 1, 3).reshape(B * N, L, D)

        key_padding_mask = None
        if valid_mask is not None:
            m = valid_mask.permute(0, 2, 1).reshape(B * N, L)  # [B*N, L]
            key_padding_mask = (m <= 0)  # True 表示要mask

        attn_out, _ = self.attn(x2, x2, x2, key_padding_mask=key_padding_mask)
        x2 = self.norm1(x2 + self.dropout(attn_out))
        ffn_out = self.ffn(x2)
        x2 = self.norm2(x2 + self.dropout(ffn_out))

        x = x2.reshape(B, N, L, D).permute(0, 2, 1, 3)
        return x


class GraphLayer(nn.Module):
    """
    简单图传播层:
    H' = GELU(A @ H W)
    输入: [B, N, D]
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [B, N, D]
        # adj: [N, N]
        h = self.linear(x)
        h = torch.einsum("ij,bjd->bid", adj, h)
        h = torch.gelu(h)
        h = self.norm(x + self.dropout(h))
        return h


class TemporalGraphTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_commodities: int,
        adj: np.ndarray,
        hidden_dim: int = 64,
        num_heads: int = 4,
        temporal_layers: int = 2,
        gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_commodities = num_commodities
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(num_features, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=512)

        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(temporal_layers)
        ])

        self.graph_layers = nn.ModuleList([
            GraphLayer(hidden_dim, dropout)
            for _ in range(gnn_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # 两个输出：未来5日最高收益率、最低收益率
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        adj_tensor = torch.tensor(adj, dtype=torch.float32)
        self.register_buffer("adj", adj_tensor)

    def forward(self, x, x_mask):
        """
        x: [B, L, N, F]
        x_mask: [B, L, N]
        return: [B, N, 2]
        """
        B, L, N, F = x.shape
        assert N == self.num_commodities

        h = self.input_proj(x)  # [B, L, N, D]

        # positional encoding 需要 [L, B, N, D]
        h = h.permute(1, 0, 2, 3)
        h = self.pos_enc(h)
        h = h.permute(1, 0, 2, 3)

        for block in self.temporal_blocks:
            h = block(h, x_mask)

        # 取最后时刻表示
        h_last = h[:, -1]  # [B, N, D]

        # 图传播
        for gnn in self.graph_layers:
            h_last = gnn(h_last, self.adj)

        out = self.head(self.dropout(h_last))  # [B, N, 2]
        return out


# =========================================================
# 6. 损失函数
# =========================================================
def masked_mse_loss(pred, target, mask):
    """
    pred:   [B, N, 2]
    target: [B, N, 2]
    mask:   [B, N]
    """
    mask = mask.unsqueeze(-1)  # [B, N, 1]
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = mask.sum() * pred.size(-1)
    denom = torch.clamp(denom, min=1.0)
    return diff2.sum() / denom


# =========================================================
# 7. 训练与验证
# =========================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, x_mask, y, y_mask in loader:
        x = x.to(device)
        x_mask = x_mask.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)

        optimizer.zero_grad()
        pred = model(x, x_mask)
        loss = masked_mse_loss(pred, y, y_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x, x_mask, y, y_mask in loader:
        x = x.to(device)
        x_mask = x_mask.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)

        pred = model(x, x_mask)
        loss = masked_mse_loss(pred, y, y_mask)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def fit_model(
    features: np.ndarray,
    mask: np.ndarray,
    close_panel: np.ndarray,
    config: Config,
):
    T, N, F = features.shape

    targets, y_mask = compute_targets(close_panel, mask, config.pred_horizon)
    adj = build_adjacency_from_correlation(close_panel, mask)

    # 时间切分：训练 / 验证 / 测试
    train_end = int(T * 0.7)
    valid_end = int(T * 0.85)

    train_ds = CommodityDataset(features, mask, targets, y_mask, config.seq_len, config.seq_len - 1, train_end)
    valid_ds = CommodityDataset(features, mask, targets, y_mask, config.seq_len, train_end, valid_end)
    test_ds  = CommodityDataset(features, mask, targets, y_mask, config.seq_len, valid_end, T - config.pred_horizon)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = TemporalGraphTransformer(
        num_features=F,
        num_commodities=N,
        adj=adj,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        temporal_layers=config.temporal_layers,
        gnn_layers=config.gnn_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_valid = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, config.device)
        valid_loss = evaluate(model, valid_loader, config.device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = evaluate(model, test_loader, config.device)
    print(f"Best valid_loss={best_valid:.6f}, test_loss={test_loss:.6f}")

    return model, adj, targets, y_mask


# =========================================================
# 8. 最新一天做预测
# =========================================================
@torch.no_grad()
def predict_latest(
    model: nn.Module,
    features: np.ndarray,   # [T, N, F]
    mask: np.ndarray,       # [T, N]
    commodities: List[str],
    seq_len: int,
    device: str,
) -> pd.DataFrame:
    """
    用最后 seq_len 天，预测“从最新这一天起算”的未来5日最高/最低收益率
    """
    T, N, F = features.shape
    assert T >= seq_len

    x = features[-seq_len:]     # [L, N, F]
    x_mask = mask[-seq_len:]    # [L, N]

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)         # [1, L, N, F]
    x_mask = torch.tensor(x_mask, dtype=torch.float32).unsqueeze(0).to(device)

    pred = model(x, x_mask)[0].cpu().numpy()  # [N, 2]

    out = pd.DataFrame({
        "commodity": commodities,
        "pred_max_return_5d": pred[:, 0],
        "pred_min_return_5d": pred[:, 1],
    }).sort_values("pred_max_return_5d", ascending=False)

    return out


# =========================================================
# 9. 示例主程序
# =========================================================
def main():
    # 你自己的数据文件
    # 列格式：
    # date, commodity, close, volume, oi
    df = pd.read_csv("commodity_daily_data.csv")

    features, mask, close_panel, dates, commodities = build_panels(
        df,
        date_col="date",
        commodity_col="commodity",
        close_col="close",
        volume_col="volume",
        oi_col="oi",
    )

    print("features.shape =", features.shape)
    print("mask.shape     =", mask.shape)
    print("close.shape    =", close_panel.shape)
    print("num_dates      =", len(dates))
    print("num_commodities=", len(commodities))

    config = Config(
        seq_len=20,
        pred_horizon=5,
        batch_size=32,
        hidden_dim=64,
        num_heads=4,
        temporal_layers=2,
        gnn_layers=2,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=20,
    )

    model, adj, targets, y_mask = fit_model(features, mask, close_panel, config)

    # 保存模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "adj": adj,
        "commodities": commodities,
        "config": config.__dict__,
    }, "temporal_graph_transformer.pt")

    # 最新预测
    latest_pred = predict_latest(
        model=model,
        features=features,
        mask=mask,
        commodities=commodities,
        seq_len=config.seq_len,
        device=config.device,
    )

    print("\nLatest 5-day prediction:")
    print(latest_pred.head(20))
    latest_pred.to_csv("latest_prediction.csv", index=False)


if __name__ == "__main__":
    main()
