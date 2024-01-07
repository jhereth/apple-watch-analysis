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

# %% [markdown]
# # Analysis of Apple Watches

# %% [markdown]
# The [Apple website](https://www.apple.com/watch/compare/) shows all available Apple Watch models with their features. We scrape this information.

# %%
import httpx
from bs4 import BeautifulSoup

r = httpx.get("https://www.apple.com/watch/compare/")
html_doc = r.text
soup = BeautifulSoup(html_doc, "html.parser")
devices_html = soup.find_all("div", class_="device-content with-list-bullet")

# %% [markdown]
# We extract the features and store them with the watch models in `Watch` objects.
#
# Later we want to match features between differen watch models. For this purpose we define a linear order on the watches: A watch is considered "smaller" if it has less features than the other watch. If they have both the same number of features, the watch the the alphabetical smaller name is considered smaller.

# %%
from collections import defaultdict
from pprint import pprint
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


watches = []

for device in devices_html:
    watches.append(
        Watch(
            name=device.h3.text,
            features=pvector(li.text for li in device.find_all("li")),
        )
    )

watches.sort()
pprint([(_.name, len(_.features)) for _ in watches])

# %% [markdown]
# The features are written as free text. While the same feature uses the same text, it's harder to find what feature of a newer model matches to which feature of an older model.
#
# Using an NLP model from [huggingface](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) we compute embeddings for the features of all watches.

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = defaultdict()

for watch in watches:
    # print(watch.name)
    embeddings[watch] = model.encode(watch.features, convert_to_tensor=True)

# %%
import itertools
from collections import Counter
from typing import Dict

import torch
from sentence_transformers import util


def dd():
    return defaultdict(dd)


def derive_mappings(matrix: torch.Tensor) -> Dict[int, int]:
    l1, l2 = matrix.size()
    maxs, indices = torch.max(matrix, dim=1)
    naive_map = {i: indices[i].item() for i in range(l1)}
    # pprint(naive_map)
    if max((vals := (c := Counter(naive_map.values())).values())) > 1:
        # print(c)
        _too_complicated = f"Too complicated: {c}"
        if len(dups := [_ for _ in c if c[_] > 1]) > 1:
            raise ValueError("More than one conflict. " + _too_complicated)
        target = dups[0]
        if c[target] > 2:
            raise ValueError(
                "More than two features point to same feature. " + _too_complicated
            )
        # print(f"{dups=}")
        features = [i for i in naive_map if naive_map[i] == dups[0]]
        i, j = features
        # print(i, j, maxs[i], maxs[j])
        if maxs[i] > maxs[j]:
            i, j = j, i
        # target_2 = (row:=matrix[i].values()).index(max(v for v in row if v < maxs[i]))
        # print(target_2)
        val_2, target_2 = (top2 := matrix[i].topk(2)).values[1].item(), top2.indices[
            1
        ].item()
        if (
            len(
                [val for i in range(l1) if (val := matrix[i][target_2].item()) >= val_2]
            )
            > 1
        ):
            raise ValueError(
                f"Second best mapping for feature {i_2} is also a good (or better) mapping for other features. "
                + _too_complicated
            )
        if target_2 in vals:
            raise ValueError(
                f"New target {target_2} for feature {i_2} has already been mapped. "
                + _too_complicated
            )
        naive_map[i] = target_2
    return naive_map


scores = defaultdict(dd)
mappings = defaultdict(dd)

for (w1, w2) in itertools.combinations(watches, 2):
    assert w1 < w2
    scores[(w1, w2)] = (matrix := util.cos_sim(embeddings[w1], embeddings[w2]))
    mappings[(w1, w2)] = derive_mappings(matrix)

# %%
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

features = set()
for w in watches:
    features |= set(w.features)
# pprint(features)
G = nx.DiGraph()
G.add_nodes_from(features)
edge_labels = defaultdict(lambda: "")
for (w1, w2) in itertools.combinations(watches, 2):
    for (s, t) in mappings[(w1, w2)].items():
        # print(w1.name, w2.name, s, t)
        G.add_edge(*(e:=(w1.features[s], w2.features[t])))
        # print(e)
        # edge_labels[e]+=f"-({w1.name[12:]}, {w2.name[12:]})"
        if e[0] != e[1]:
            edge_labels[e]+=f"-({w1.name[12:]}, {w2.name[12:]})"
GC = list(nx.connected_components(G.to_undirected()))
print(G.number_of_nodes(), G.number_of_edges(), len(list(GC)))

# print("edges", nx.get_edge_attributes(G, "weight"))
# print(edge_labels)
for (i, component) in enumerate(GC):
    print(i, component)
    sG = G.subgraph(component)
    # print("type", type(sG))
    # print("edges", sG.edges)
    pos = nx.spring_layout(sG)
    # print(pos)
    plt.figure()
    nx.draw(sG, pos, with_labels=True, connectionstyle='arc3, rad = 0.1', font_size=6)
    nx.draw_networkx_edge_labels(sG, pos, edge_labels={e:l for (e,l) in edge_labels.items() if any(n in e for n in component)})
    # plt.axis_off()
    # fig.tight_layout()
    plt.show()
    if i >= 0:
        # break
        pass


# %% tags=[]
@define(frozen=True)
class FeatureNode:
    watch: Watch
    index: int

    def name(self):
        return watch.features[self.index]


feature_nodes = [
    FeatureNode(watch, index)
    for watch in watches
    for index in range(len(watch.features))
]
print(len(feature_nodes))

# %% [markdown]
# ## Transitivity
#
# The same feature `i` of watch `w1` should be mapped to the same feature `j` of `w3` directly or via feature `k` of watch `w2` (if `i` is mapped to `k`).

# %%
for (w1, w2, w3) in itertools.combinations(watches, 3):
    assert w1 < w2 < w3
    m1, m2, m12 = mappings[w1,w2], mappings[w2,w3], mappings[w1,w3]
    # print(m1, m2, m12)
    for i in range(len(w1.features)):
        # print(m1[i])
        assert (l:=m2[m1[i]])==(r:=m12[i]), f"({w1.name}, {w2.name}, {w3.name}): {w1.features[i]} maps to {w3.features[l]} and {w3.features[r]}"



# %%
import pandas as pd

i = 0

for w in watches:
    for c in GC:
        assert len(c.intersection(w.features)) <= 1
        # print(c.intersection(w.features))
        i+=1
print(f"{i} tests successful")


def pop_or_none(s: set):
    if len(s) >= 1:
        return s.pop()
    return None

df = pd.DataFrame(
    data=[{w.name: pop_or_none(c.intersection(w.features)) for w in watches} for c in GC]
)

# %%

# %%

# %%
from itables import init_notebook_mode

init_notebook_mode(all_interactive=True)
df

# %%
