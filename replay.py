# быстрая проверка без переобучения: берет сохраненные предсказания стека (cache/stack_v2.npz),
# подбирает порог по OOF так же, как в pipeline.py, и собирает submission_replay.csv
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

y = pd.read_csv("train_6dd2c989.csv").result.values
d = np.load("cache/stack_v2.npz")
oof, test = d["oof"], d["test"]

ts = np.linspace(0.1, 0.9, 161)
f1s = np.array([f1_score(y, oof > t) for t in ts])
th = ts[int(np.argmax(f1s))]
print(f"порог {th:.3f}, OOF F1 {f1s.max():.4f}")

pred = (test > th).astype(int)
pd.DataFrame({"Id": np.arange(len(pred)), "Predicted": pred}).to_csv("submission_replay.csv", index=False)

ref = pd.read_csv("submission_v2.csv").Predicted.values
print(f"отличается от submission_v2.csv в {(pred != ref).sum()} строках из {len(ref)}")
