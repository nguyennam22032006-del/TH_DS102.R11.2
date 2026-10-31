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

    X = np.column_stack([data["Goals_For"], data["Goals_Against"]]).astype(float)
    win_pct = data["Win_Pct"].astype(float)
    assert np.isfinite(X).all() and np.isfinite(win_pct).all()

    y_binary = (win_pct >= 0.5).astype(int)

    y_multi = np.digitize(win_pct, bins=[0.4, 0.6])

    train = data["Year"] < 2006
    val = (data["Year"] >= 2006) & (data["Year"] < 2009)
    test = data["Year"] >= 2009

    mean = X[train].mean(axis=0)
    std = X[train].std(axis=0)
    std[std == 0] = 1
    X = (X - mean) / std

    return {
        "X_train": X[train], "X_val": X[val], "X_test": X[test],
        "yb_train": y_binary[train], "yb_val": y_binary[val], "yb_test": y_binary[test],
        "ym_train": y_multi[train], "ym_val": y_multi[val], "ym_test": y_multi[test]
    }
