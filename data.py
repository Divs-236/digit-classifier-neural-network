import os, glob
import numpy as np

def _find(folder, key):
    hits = [f for f in glob.glob(os.path.join(folder, "*")) 
            if key in os.path.basename(f).lower() and os.path.isfile(f)]

    if not hits:
        raise FileNotFoundError

    return hits[0]

def _read_idx(folder):
    img_path, lbl_path = _find(folder,"images"), _find(folder, "labels")

    if img_path.endswith(".gz"):
        import gzip; op = gzip.open
    else:
        op = open

    with op(img_path, "rb") as f:
        X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with op(lbl_path, "rb") as f:
        y = np.frombuffer(f.read(), np.uint8, offset = 8)
    
    return X, y

def _read_png_folders(folder):
    from PIL import Image
    X, y = [], []
    for d in range(10):
        for p in glob.glob(os.path.join(folder, str(d), "*.*")):
            X.append(np.array(Image.open(p).convert("L").resize((28,28))))
            y.append(d)

    return np.array(X, dtype = np.uint8), np.array(y)

def _read(folder):
    if os.path.isdir(os.path.join(folder, "0")):
        return _read_png_folders(folder)

    return _read_idx(folder)


def load(num_train=2000, num_test=500, pool=False, seed=0, root="."):
    rng = np.random.default_rng(seed)

    def prep(split, n):
        X, y = _read(os.path.join(root, split))
        idx = rng.permutation(len(X))[:n]

        X = X[idx].astype(np.float32) / 255.0
        X = (X - 0.1307) / 0.3081
        y = y[idx]

        if pool:
            X = X.reshape(-1, 14, 2, 14, 2).mean(axis=(2,4))
        return X.reshape(len(X), -1), y.astype(np.int64)

    Xtr, ytr = prep("train", num_train)
    Xte, yte = prep("test", num_test)

    return Xtr, ytr, Xte, yte
