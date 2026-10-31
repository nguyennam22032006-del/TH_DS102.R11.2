from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_data
from evaluation import danh_gia, tinh_chi_so


class SoftmaxRegression:
    def __init__(self, epoch=2000, lr=0.05, balance_power=0.0):
        self.epoch, self.lr, self.balance_power = epoch, lr, balance_power

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def loss_fn(self, y, p):
        return -(y * np.log(np.clip(p, 1e-15, 1.0))).sum(axis=1).mean()

    def predict(self, X):
        return self.softmax(X @ self.w + self.b)

    def predict_label(self, X):
        return self.predict(X).argmax(axis=1)

    def fit(self, X, y, X_val, y_val):
        N, d = X.shape
        K = y.shape[1]
        self.w, self.b = np.zeros((d, K)), np.zeros(K)
        self.losses, self.val_losses = [], []
        self.best_f1, self.best_val_loss, self.best_epoch = -1.0, np.inf, 0

        counts = y.sum(axis=0)
        if np.any(counts == 0):
            raise ValueError("Tập train phải có mẫu của mọi lớp.")
        class_weights = (N / (K * counts)) ** self.balance_power
        sw = class_weights[y.argmax(axis=1), None]
        wsum = sw.sum()
        val_labels = y_val.argmax(axis=1)

        for e in range(self.epoch):
            err = (self.predict(X) - y) * sw
            self.w -= self.lr * X.T @ err / wsum
            self.b -= self.lr * err.sum(axis=0) / wsum

            self.losses.append(self.loss_fn(y, self.predict(X)))
            val_proba = self.predict(X_val)
            val_loss = self.loss_fn(y_val, val_proba)
            self.val_losses.append(val_loss)
            f1 = tinh_chi_so(val_labels, val_proba.argmax(axis=1), K)[:, 2].mean()

            if (f1, -val_loss) > (self.best_f1, -self.best_val_loss):
                self.best_f1, self.best_val_loss, self.best_epoch = f1, val_loss, e + 1
                best_w, best_b = self.w.copy(), self.b.copy()

        self.w, self.b = best_w, best_b
        return self


def main():
    d = load_data()
    X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
    y_train, y_val, y_test = d["ym_train"], d["ym_val"], d["ym_test"]

    K = 3
    Y_train, Y_val = np.eye(K)[y_train], np.eye(K)[y_val]

    models = [SoftmaxRegression(balance_power=p).fit(X_train, Y_train, X_val, Y_val)
              for p in [0.0, 0.5, 1.0]]
    model = max(models, key=lambda m: (m.best_f1, -m.best_val_loss))

    danh_gia(y_test, model.predict_label(X_test), ["Thap", "Trung binh", "Cao"])

    epochs = np.arange(1, len(model.losses) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, model.losses, label="Train")
    plt.plot(epochs, model.val_losses, label="Validation")
    plt.title("Softmax Regression - Hockey")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy (unweighted)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    results = Path(__file__).resolve().parent / "results"
    results.mkdir(exist_ok=True)
    plt.savefig(results / "bai_2_loss.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()