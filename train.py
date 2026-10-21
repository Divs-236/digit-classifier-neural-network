import json, math, time, sys, random
import numpy as np
sys.setrecursionlimit(10000)
from nn import MLP
from data import load

random.seed(0)

def cross_entropy(logits, y):
    n = len(y)
    m = logits.data.max(axis=1, keepdims=True)
    z = logits - m

    logp = z - z.exp().sum(axis=1, keepdims=True).log()
    eps = 0.1
    onehot = np.full((n, 10), eps / 9, dtype=np.float32)
    onehot[np.arange(n), y] = 1.0 - eps
    return -(logp * onehot).sum() / n

def accuracy(model, X, y, bs=1000):
    correct = 0

    for i in range(0, len(X), bs):
        pred = model(X[i:i+bs]).data.argmax(axis=1)
        correct += (pred == y[i:i+bs]).sum()
    return correct / len(X)

def shift(X, max_px=2):
    imgs = X.reshape(-1, 28, 28)
    out = np.empty_like(imgs)
    bg = imgs.min()

    for i in range(len(imgs)):
        dx, dy = np.random.randint(-max_px, max_px + 1, 2)
        out[i] = np.roll(imgs[i], (dy, dx), axis=(0,1))

        if dy > 0: out[i, :dy, :] = bg
        elif dy < 0: out[i, dy:, :] = bg
        if dx > 0: out[i, :, :dx] = bg
        elif dx < 0: out[i, :, dx:] = bg

    return out.reshape(len(X), -1)

def train(num_train, num_test, epochs, learn_rate, batch=16, hidden=32):
    Xtr, ytr, Xte, yte = load(num_train, num_test)
    model = MLP(784, [hidden, 64, 10])
    history = []

    for epoch in range(epochs):
        lr = 0.005 + 0.5 * (learn_rate - 0.005) * (1 +  math.cos(math.pi * epoch / epochs))
        t0 = time.time()
        model.train_mode(True)
        order = list(range(len(Xtr))); random.shuffle(order)
        epoch_loss, nb = 0.0, 0
        params = model.parameters()

        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            loss = cross_entropy(model(shift(Xtr[idx])), ytr[idx])
            model.zero_grad()
            loss.backward()

            wd = 1e-4
            for p in params:
                p.data -= lr * (p.grad + wd * p.data)

            epoch_loss += float(loss.data); nb += 1

        test_acc = accuracy(model, Xte, yte)
        history.append({"epoch" : epoch, "loss" : epoch_loss / nb, "test_acc" : test_acc})
        print(f"epoch {epoch}, loss {epoch_loss/nb:.4f}, test_acc {test_acc:.3f} ({time.time() - t0:.0f}s)")

        json.dump([p.data.tolist() for p in params], open("weights.json", "w"))
        json.dump(history, open("history.json", "w"))
    return model

if __name__ == "__main__":
    train(num_train=60000, num_test=10000, epochs=12, learn_rate= 0.1, batch=64, hidden=128) 