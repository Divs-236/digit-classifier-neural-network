import os, glob
import numpy as np


#Finds first instance of a keyword in a filename in a given folder
def _find(folder, key):
    hits = [f for f in glob.glob(os.path.join(folder, "*")) 
            if key in os.path.basename(f).lower() and os.path.isfile(f)]

    if not hits:
        raise FileNotFoundError #Error if no instance of keyword is found

    return hits[0]


def _read(folder):
    img_path, lbl_path = _find(folder,"images"), _find(folder, "labels")

    with open(img_path, "rb") as f:
        #skips 16-byte header for image files
        X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)    

    with open(lbl_path, "rb") as f:
        #skips 8-byte header for label files
        y = np.frombuffer(f.read(), np.uint8, offset = 8)
    
    return X, y

def load(num_train=2000, num_test=500, pool=False, seed=0, root="."): #root : directory containing train and test
    rng = np.random.default_rng(seed)

    def prep(split, n):
        X, y = _read(os.path.join(root, split))  #retrieves images from respective split
        idx = rng.permutation(len(X))[:n]  #takes a random subset of 'n' samples   

        X = X[idx].astype(np.float32) / 255.0 
        X = (X - 0.1307) / 0.3081  #normalized using MNIST mean and std.
        y = y[idx]

        if pool:  #OPTIONAL: downscaling factor (14x14) for cheaper compute
            X = X.reshape(-1, 14, 2, 14, 2).mean(axis=(2,4))
        return X.reshape(len(X), -1), y.astype(np.int64)

    Xtr, ytr = prep("train", num_train)  #returns normalized images along with their int64 labels
    Xte, yte = prep("test", num_test)

    return Xtr, ytr, Xte, yte
