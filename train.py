import json, math, time, random
import numpy as np
from nn import MLP
from data import load

random.seed(0)

def cross_entropy(logits, y):
    n = len(y)
    m = logits.data.max(axis=1, keepdims=True)
    z = logits - m

    logp = z - z.exp().sum(axis=1, keepdims=True).log()  #log softmax z - log(sum(exp(z)))
    eps = 0.1  #label smoothing 
    onehot = np.full((n, 10), eps / 9, dtype=np.float32)  #wrong values share 0.1 all together
    onehot[np.arange(n), y] = 1.0 - eps  #correct value shares 0.9 alone
    return -(logp * onehot).sum() / n  #mean over the batch

#Accuracy calculator; calculates in chunks to reduce memory consumption
def accuracy(model, X, y, cap=1000):
    correct = 0

    for i in range(0, len(X), cap):
        pred = model(X[i:i+cap]).data.argmax(axis=1)
        correct += (pred == y[i:i+cap]).sum()
    return correct / len(X)

#Data augmentation; randomly moves each image by up to 2 pixels in x and y, to improve training and prevent overfitting
def shift(X, max_px=2):
    imgs = X.reshape(-1, 28, 28)
    out = np.empty_like(imgs)
    bg = imgs.min()  #background colour after normalization

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
    model = MLP(784, [hidden, 64, 10])  # 784 -> 128 -> 64 -> 10
    history = []

    for epoch in range(epochs):
        #cosine decay; starts at learn_rate, to a minimum of 0.005
        lr = 0.005 + 0.5 * (learn_rate - 0.005) * (1 +  math.cos(math.pi * epoch / epochs))
        t0 = time.time()
        model.train_mode(True)
        order = list(range(len(Xtr))); random.shuffle(order)
        epoch_loss, nb = 0.0, 0
        params = model.parameters()

        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            loss = cross_entropy(model(shift(Xtr[idx])), ytr[idx])  #augment only the training batch
            model.zero_grad()  #clear gradients before another backward pass
            loss.backward()

            wd = 1e-4  #weight decay

            for p in params:  #Gradient Descent
                p.data -= lr * (p.grad + wd * p.data)

            epoch_loss += float(loss.data); nb += 1

        test_acc = accuracy(model, Xte, yte)
        history.append({"epoch" : epoch, "loss" : epoch_loss / nb, "test_acc" : test_acc})
        print(f"epoch {epoch}, loss {epoch_loss/nb:.4f}, test_acc {test_acc:.3f} ({time.time() - t0:.0f}s)")

        #stored weights and history after every epoch
        json.dump([p.data.tolist() for p in params], open("weights.json", "w"))
        json.dump(history, open("history.json", "w"))
    return model

if __name__ == "__main__":
    train(num_train=60000, num_test=10000, epochs=12, learn_rate= 0.1, batch=64, hidden=128) 