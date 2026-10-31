import numpy as np


def tinh_chi_so(y_that, y_du_doan, so_lop):
    chi_so = []
    for lop in range(so_lop):
        TP = np.sum((y_that == lop) & (y_du_doan == lop))
        FP = np.sum((y_that != lop) & (y_du_doan == lop))
        FN = np.sum((y_that == lop) & (y_du_doan != lop))

        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        denominator = 2 * TP + FP + FN
        f1 = 2 * TP / denominator if denominator > 0 else 0.0
        chi_so.append([precision, recall, f1])
    return np.array(chi_so)


def danh_gia(y_that, y_du_doan, ten_lop):
    chi_so = tinh_chi_so(y_that, y_du_doan, len(ten_lop))
    print(f"{'Lớp':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for ten, (p, r, f) in zip(ten_lop, chi_so):
        print(f"{ten:<16} {p:10.4f} {r:10.4f} {f:10.4f}")
    p, r, f = chi_so.mean(axis=0)
    print(f"{'Macro':<16} {p:10.4f} {r:10.4f} {f:10.4f}")

