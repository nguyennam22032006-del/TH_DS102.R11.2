from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_data
from evaluation import danh_gia


def main():
    d = load_data()
    tasks = [
        ("LOGISTIC", "yb", ["Duoi 50%", "Tu 50%"]),
        ("SOFTMAX", "ym", ["Thap", "Trung binh", "Cao"]),
    ]
    for title, key, names in tasks:
        model = LogisticRegression(max_iter=2000).fit(d["X_train"], d[f"{key}_train"])
        print(f" {title} SCIKIT-LEARN")
        danh_gia(d[f"{key}_test"], model.predict(d["X_test"]), names)


if __name__ == "__main__":
    main()