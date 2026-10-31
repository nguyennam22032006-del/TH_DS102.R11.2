from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_data
from evaluation import danh_gia


class LogisticRegression:
    def __init__(self, epoch=2000, lr=0.05, l2=0.0, patience=100):
        self.epoch, self.lr, self.l2, self.patience = epoch, lr, l2, patience

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def loss_fn(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def predict(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict_label(self, X):
        return (self.predict(X) >= 0.5).astype(int).ravel()

    def fit(self, X, y, X_val, y_val):
        N, d = X.shape
        y, y_val = np.reshape(y, (-1, 1)), np.reshape(y_val, (-1, 1))
        self.w, self.b = np.zeros((d, 1)), 0.0
        self.losses, self.val_losses = [], []
        self.best_val_loss, self.best_epoch, wait = np.inf, 0, 0
        best_w, best_b = self.w.copy(), self.b

        for e in range(self.epoch):
            err = self.predict(X) - y
            self.w -= self.lr * (X.T @ err / N + self.l2 * self.w)
            self.b -= self.lr * err.mean()

            self.losses.append(self.loss_fn(y, self.predict(X)))
            val_loss = self.loss_fn(y_val, self.predict(X_val))
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss, self.best_epoch = val_loss, e + 1
                best_w, best_b, wait = self.w.copy(), self.b, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.w, self.b = best_w, best_b
        return self


def main():
    d = load_data()
    X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
    y_train, y_val, y_test = d["yb_train"], d["yb_val"], d["yb_test"]

    models = [LogisticRegression(l2=l2).fit(X_train, y_train, X_val, y_val)
              for l2 in [0.0, 0.001, 0.01, 0.1, 1.0]]
    model = min(models, key=lambda m: m.best_val_loss)

    danh_gia(y_test, model.predict_label(X_test), ["Duoi 50%", "Tu 50%"])

    epochs = np.arange(1, len(model.losses) + 1)
    plt.plot(epochs, model.losses, label="Train")
    plt.plot(epochs, model.val_losses, label="Validation")
    plt.axvline(model.best_epoch, color="gray", linestyle="--", label="Best validation")
    plt.title(f"Logistic Regression - Hockey (L2={model.l2:g})")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    results = Path(__file__).resolve().parent / "results"
    results.mkdir(exist_ok=True)
    plt.savefig(results / "bai_1_loss.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()