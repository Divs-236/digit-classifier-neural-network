import json, sys
import numpy as np
from nn import MLP
from data import load

model = MLP(196, [32, 10])

for p, w in zip(model.parameters(), json.load(open("weights.json"))):
    p.data = w

def predict(x):
    logits = model(x)
    vals = [l.data for l in logits]

    m = max(vals)

    exps = [np.exp(v - m) for v in vals]
    probs = [e / sum(exps) for e in exps]

    return int(np.argmax(probs)), probs

if __name__ == "__main__":
    _, _, Xte, yte = load(num_train=20, num_test=5000)

    correct = 0
    incorrect = 0

    for x, y in zip(Xte, yte):
        pred, probs = predict(x)

        if pred == y:
            correct += 1
        else:
            incorrect += 1

        
        print(f"true {y} predicted {pred} confidence {probs[pred]:.2f}")

    print()
    print("Accuracy: ",correct / (correct + incorrect))