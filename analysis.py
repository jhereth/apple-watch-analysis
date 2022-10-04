# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import httpx
from bs4 import BeautifulSoup

r = httpx.get("https://www.apple.com/watch/compare/")
html_doc = r.text
soup = BeautifulSoup(html_doc, "html.parser")
devices_src = soup.find_all("div", class_="device-content with-list-bullet")

# %%
from typing import List

from attrs import define
from pyrsistent import PVector
from pyrsistent import pvector


@define(frozen=True)
class Watch:
    name: str
    features: PVector

    def __lt__(self, other: "Watch") -> bool:
        if len(self.features) < len(other.features):
            return True
        if (len(self.features) == len(other.features)) and (self.name < other.name):
            return True
        return False


# %%
from collections import defaultdict
from pprint import pprint

watches = []

for device in devices_src:
    watches.append(
        Watch(
            name=device.h3.text,
            features=PVector(li.text for li in device.find_all("li")),
        )
    )

watches.sort()
pprint([(_.name, len(_.features)) for _ in watches])

# %%
w = watches[0]
type(w.features)

# %%
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = defaultdict()

for watch in watches:
    print(watch.name)
    embeddings[watch] = model.encode(watch.features, convert_to_tensor=True)

# %%

# %%
pprint(embeddings["Apple\xa0Watch SE"])


# %%
def dd():
    return defaultdict(dd)


scores = defaultdict(dd)
key_list = list(features.keys())

for i in range(len(key_list)):
    key_i = key_list[i]
    for j in range(i + 1, len(key_list)):
        key_j = key_list[j]
        score = util.cos_sim(embeddings[key_i], embeddings[key_j])
        scores[key_i][key_j] = score

# %%
first = key_list[0]
second = key_list[2]
print(
    (matrix := scores[first][second]).size(),
    len(features[first]),
    len(features[second]),
)


# %% tags=[]
@define(frozen=True)
class FeatureNode:
    watch: Watch
    index: int

    def name(self):
        return watch.features[index]


feature_nodes = [
    FeatureNode(watch, index)
    for watch in devices
    for index in range(len(watch.features))
]
print(len(feature_nodes))

# %%
import torch

argmax_vector = torch.argmax(matrix, dim=1)
for i in range(len(argmax_vector)):
    j = argmax_vector[i].item()
    if (s := matrix[i][j].item()) < 0.999:
        print(i, j, s, features[first][i], "|", features[second][j])

# %%
argmax_vector = torch.argmax(matrix, dim=0)
for j in range(len(argmax_vector)):
    i = argmax_vector[j].item()
    if (s := matrix[i][j].item()) < 0.999:
        print(i, j, s, features[first][i], "|", features[second][j])

# %%
print("|", (one := features[first][14]), "|", (two := features[second][13]), "|")
print(one == two)

# %%
import itertools

print(len(key_list))
pprint(combs := list(itertools.combinations(key_list, 2)))
print(len(combs))

# %%
t = torch.Tensor([[1, -7, 3], [8, -15, 6]])
print(t)
print(t.size())

# %%
torch.max(t, dim=0)

# %%
torch.max(t, dim=1)

# %%
torch.argmax(t, dim=0)

# %%
torch.argmax(t, dim=1)

# %%
torch.max(t)

# %%
torch.argmax(t)

# %%
a = [2, 1, 3]
pprint(sorted(a))
pprint(a.sort())
pprint(a)

# %%
