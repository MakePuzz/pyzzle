#%%
import json

import numpy as np
import matplotlib.pyplot as plt

from pyzzle import Puzzle

fname = "json/MakePuzz_Season1.json"

with open(fname, "r") as f:
    data = json.load(f)

#%% std
def get_std(name):
    evals = []
    for k, v in data[name]["evaluation"].items():
        evals += [int(k)]*v
    return np.array(evals).std()

std_mean = np.mean(list(map(get_std, data)))
print("std mean: ", std_mean)
# %% cover_sum
def get_cover(name):
    return Puzzle.get_cover(np.array(eval(data[name]["list"])))

covers = np.array(list(map(get_cover, data))).astype(bool).astype(int)
cover_sum = covers.sum(axis=0)

fig, ax = plt.subplots()
mp = ax.imshow(cover_sum)
ax.set(title="counts", xticks=np.arange(15), yticks=np.arange(15))
fig.colorbar(mp)
fig.savefig("counts.png")
# %%
def get_eval_mean(name):
    keys = np.array(list(data[name]["evaluation"].keys())).astype(int)
    vals = list(data[name]["evaluation"].values())
    return sum(keys*vals)/sum(vals)

evals = np.array(list(map(get_eval_mean, data)))
# %%
covers = np.array(list(map(get_cover, data))).astype(bool)
cover_sum = covers[evals >= 7].sum(axis=0)

fig, ax = plt.subplots()
mp = ax.imshow(cover_sum)
ax.set(title="Counts (Eval >= 7)", xticks=np.arange(15), yticks=np.arange(15))
fig.colorbar(mp)
fig.savefig("counts_high.png")
# %%
covers = np.array(list(map(get_cover, data))).astype(int)
cv_flat = np.array(list(map(lambda cover: cover.ravel(), covers)))

from sklearn.decomposition import PCA
pca = PCA(n_components=5, copy=True, whiten=False, svd_solver='auto', random_state=None)
pca.fit(cv_flat)
# %%
cv_flat_de = pca.transform(cv_flat)

# %%
cv_flat_de.shape