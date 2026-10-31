"""Đọc hockey.csv, tạo nhãn và chuẩn hóa chung cho ba bài."""
from pathlib import Path
import numpy as np


def load_data():
    path = Path(__file__).resolve().parent / "hockey.csv"
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8-sig"
    )

    # Không dùng Win_Pct hoặc số trận thắng/thua làm đặc trưng.
    X = np.column_stack([data["Goals_For"], data["Goals_Against"]]).astype(float)
    win_pct = data["Win_Pct"].astype(float)
    assert np.isfinite(X).all() and np.isfinite(win_pct).all()

    # Hai lớp: dưới 50% / từ 50%.
    y_binary = (win_pct >= 0.5).astype(int)

    # Ba lớp: dưới 40% / từ 40% đến dưới 60% / từ 60%.
    y_multi = np.digitize(win_pct, bins=[0.4, 0.6])

    # Chia theo mùa giải; validation dùng để chọn mô hình.
    train = data["Year"] < 2006
    val = (data["Year"] >= 2006) & (data["Year"] < 2009)
    test = data["Year"] >= 2009

    mean = X[train].mean(axis=0)
    std = X[train].std(axis=0)
    std[std == 0] = 1
    X = (X - mean) / std

    # Tên khóa giúp mỗi bài lấy đúng nhãn, không phải nhớ thứ tự trả về.
    return {
        "X_train": X[train], "X_val": X[val], "X_test": X[test],
        "yb_train": y_binary[train], "yb_val": y_binary[val], "yb_test": y_binary[test],
        "ym_train": y_multi[train], "ym_val": y_multi[val], "ym_test": y_multi[test]
    }
