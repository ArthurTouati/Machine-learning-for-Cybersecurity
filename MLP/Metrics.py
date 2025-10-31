# MLP→SVM "FAST" : 15 epochs, courbes propres, exécution rapide
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, log_loss,
    precision_recall_fscore_support
)

NPZ = "images_malware.npz"

# ---- utilitaires images ----
def resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h, w = img.shape
    if (h, w) == (new_h, new_w): return img
    r = (np.linspace(0, h - 1, new_h)).astype(np.int64)
    c = (np.linspace(0, w - 1, new_w)).astype(np.int64)
    return img[r[:, None], c[None, :]]

def ensure_gray2d(img):
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[-1] == 1: arr = arr.squeeze(-1)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        arr = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    if arr.ndim != 2: raise ValueError(f"Image non 2D: shape={arr.shape}")
    return arr.astype(np.float32)

def load_Xy_from_arr(npz_path, target_hw=(32,32)):
    d = np.load(npz_path, allow_pickle=True)
    arr = d["arr"]  # (N,2) -> [image,label]
    imgs = [ensure_gray2d(x) for x in arr[:,0]]
    labels_raw = np.asarray(arr[:,1])
    H0,W0 = target_hw
    imgs = [resize_nn(im,H0,W0) for im in imgs]
    X = np.stack(imgs)
    if X.max()>1.0: X = X/255.0
    N,H,W = X.shape
    X = X.reshape(N,H*W).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(labels_raw.ravel())
    return X,y,le,(H0,W0)

def last_hidden_feats(mlp: MLPClassifier, X: np.ndarray) -> np.ndarray:
    def relu(z): return np.maximum(0, z)
    A = X
    sizes = mlp.hidden_layer_sizes if isinstance(mlp.hidden_layer_sizes, tuple) else (mlp.hidden_layer_sizes,)
    for i in range(len(sizes)):
        Z = A @ mlp.coefs_[i] + mlp.intercepts_[i]
        A = relu(Z)
    return A

def print_metrics_table(acc, prec, rec, f1, n_train, n_test, epochs):
    header = "Metrics Table:"
    rows = [
        ("Metrics","Value"),
        ("Accuracy",  f"{acc:.4f}"),
        ("Precision", f"{prec:.4f}"),
        ("Recall",    f"{rec:.4f}"),
        ("F1 Score",  f"{f1:.4f}"),
        (f"Data Points {n_train} (train) / {n_test} (test)",""),
        ("Epochs", str(epochs))
    ]
    col1 = max(len(r[0]) for r in rows)+2
    col2 = max(len(r[1]) for r in rows)+2
    print("\n"+header)
    for a,b in rows: print(f"{a:<{col1}}{b:>{col2}}")

def main():
    # 1) data → 32x32 → flatten
    X,y,le,used = load_Xy_from_arr(NPZ, target_hw=(32,32))
    print(f"[OK] X={X.shape}, classes={len(le.classes_)} size={used}")

    # 2) splits
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr_full,y_tr_full,test_size=0.20,random_state=42,stratify=y_tr_full)

    # 3) standardize + PCA (randomized → rapide)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr); X_va = scaler.transform(X_va); X_te = scaler.transform(X_te)

    pca = PCA(n_components=128, whiten=True, svd_solver="randomized", random_state=42)
    X_tr = pca.fit_transform(X_tr); X_va = pca.transform(X_va); X_te = pca.transform(X_te)

    # 4) MLP compact, régul' modérée, LR plus haut → converge vite
    n_epochs = 15
    classes = np.unique(y)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,64),
        activation="relu",
        solver="adam",
        alpha=1e-3,                 # L2
        batch_size=512,
        learning_rate_init=1e-3,    # plus rapide
        max_iter=1, warm_start=True,
        shuffle=True, random_state=42, verbose=False
    )

    tr_loss=[]; va_loss=[]; tr_acc=[]; va_acc=[]
    for ep in range(n_epochs):
        if ep==0: mlp.partial_fit(X_tr,y_tr,classes=classes)
        else:     mlp.partial_fit(X_tr,y_tr)

        ytrp = mlp.predict_proba(X_tr); yvap = mlp.predict_proba(X_va)
        tr_loss.append(log_loss(y_tr,ytrp,labels=classes))
        va_loss.append(log_loss(y_va,yvap,labels=classes))
        tr_acc.append(accuracy_score(y_tr, mlp.predict(X_tr)))
        va_acc.append(accuracy_score(y_va, mlp.predict(X_va)))

    # 5) Graphes
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.plot(tr_loss,label="Training Loss"); plt.plot(va_loss,label="Validation Loss")
    plt.title("MLP Model Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.subplot(1,2,2); plt.plot(tr_acc,label="Training Accuracy"); plt.plot(va_acc,label="Validation Accuracy")
    plt.title("MLP Model Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout(); plt.show()

    # 6) Features → SVM (un peu régularisé, équilibré)
    X_trte = np.vstack([X_tr,X_va]); y_trte = np.concatenate([y_tr,y_va])
    X_trte_f = last_hidden_feats(mlp, X_trte)
    X_te_f   = last_hidden_feats(mlp, X_te)

    svm = SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=42)
    svm.fit(X_trte_f, y_trte)

    y_pred = svm.predict(X_te_f)
    acc = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)

    print("\nClassification report (par classe) :")
    print(classification_report(y_te, y_pred, digits=4, target_names=[str(c) for c in le.classes_]))
    cm = confusion_matrix(y_te, y_pred)
    print("Confusion matrix:\n", cm)

    print_metrics_table(acc, prec, rec, f1, X_trte.shape[0], X_te.shape[0], n_epochs)

    plt.figure(figsize=(7,6))
    plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.colorbar(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()


