import json, math, time, sys, random
sys.setrecursionlimit(10000)
from nn import MLP
from data import load

random.seed(0)

def softmax(logits, target):
    m = max(l.data for l in logits)
    exps = [(l-m).exp() for l in logits]

    total = sum(exps)
    return -(exps[target] / total).log()

def accuracy(model, X, y):
    correct = 0

    for xi, yi in zip(X,y):
        logits = model(xi)
        pred = max(range(10), key=lambda k: logits[k].data)
        correct += (pred == yi)

    return correct / len(X)

def train(num_train, num_test, epochs, learn_rate, batch=16, hidden=32):
    Xtr, ytr, Xte, yte = load(num_train, num_test)
    model = MLP(196, [hidden, 10])
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        order = list(range(len(Xtr))); random.shuffle(order)
        epoch_loss, nb = 0.0, 0
        params = model.parameters()

        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            loss = (sum(softmax(model(Xtr[j]), ytr[j]) for j in idx) / len(idx))
            model.zero_grad()
            loss.backward()

            for p in params:
                p.data -= learn_rate * p.grad

            epoch_loss += loss.data; nb += 1

        test_acc = accuracy(model, Xte, yte)
        history.append({"epoch" : epoch, "loss" : epoch_loss / nb, "test_acc" : test_acc})
        print(f"epoch {epoch}, loss {epoch_loss/nb:.4f}, test_acc {test_acc:.3f} ({time.time() - t0:.0f}s)")

        json.dump([p.data for p in params], open("weights.json", "w"))
        json.dump(history, open("history.json", "w"))
    return model

if __name__ == "__main__":
    train(num_train=60000, num_test=10000, epochs=8, learn_rate= 0.05, batch=16, hidden=32)