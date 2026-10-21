import json
import numpy as np
from nn import MLP
from data import load


model = MLP(784, [128, 64, 10])  #rebuilds model as it was implemented in training
for p, w in zip(model.parameters(), json.load(open("weights.json"))):  #loads the saved weights from training
    p.data = np.array(w, dtype=np.float32)

def predict(x):  #returns predicted class per row, all softmax probabilities
    logits = model(x).data
    z = logits - logits.max(axis=1, keepdims=True)  
    e = np.exp(z)

    probs = e / e.sum(axis=1, keepdims=True)  #softmax; each row sums to 1
    return probs.argmax(axis=1), probs

if __name__ == "__main__":
    _, _, Xte, yte = load(num_test=10000)

    preds, probs = predict(Xte)

    for y, p, pr in zip(yte, preds, probs):
        print(f"true {y} predicted {p} confidence {pr[p]:.2f}")

    print()
    print("Accuracy:", (preds == yte).mean())