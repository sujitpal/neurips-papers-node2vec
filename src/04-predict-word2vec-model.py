import gensim
import os

DATA_DIR = "../data"
MODEL_FILE = os.path.join(DATA_DIR, "w2v-nips-papers.model")
ID_TITLE_MAPPING_FILE = os.path.join(DATA_DIR, "NeurIPS_id-title.csv")

pid2title = {}
fitm = open(ID_TITLE_MAPPING_FILE, "r")
for line in fitm:
    pid, title = line.strip().split('\t')
    pid = int(pid) - 1      # pids provided are 1-based
    pid2title[pid] = title
fitm.close()

model = gensim.models.Word2Vec.load(MODEL_FILE)

# elements of the embedding
print("10 random papers")
print("")
pids = model.wv.vocab.keys()
sample_pids = [int(p) for p in pids][0:10]
for pid in sample_pids:
    print("{:d} {:s}".format(pid, pid2title[pid]))
print("---")

# check for most similar to given
# 1479 Ensemble Learning for Multi-Layer Networks (1997)
source_paper_id = 1479
print("Papers most similar to {:d} {:s}".format(
    source_paper_id, pid2title[source_paper_id]
))
print("")
for rpid, score in model.most_similar(str(source_paper_id)):
    print("{:.3f} {:s} {:s}".format(
        score, rpid, pid2title[int(rpid)]
    )
)
print("---")

# paper vector arithmetic. Given following papers:
# A: 1180    Representing Face Images for Emotion Classification (1996)
# B: 5209    Transfer Learning in a Transductive Setting (2013)
# C: 3488    Semi-supervised Learning with Weakly-Related Unlabeled Data : Towards Better Text Categorization (2008)
# we want to find D, such that A : B :: C : D
# D should be B - A + C
print("Arithmetic in paper space")
paper_d = model.most_similar(
    positive=["5209", "3488"],
    negative=["1180"])[0:1][0][0]
print("{:s} {:s}".format(paper_d, pid2title[int(paper_d)]))
print("---")

# vector representation
vec = model["7077"]
print("Paper vector shape:", vec.shape)
print("---")