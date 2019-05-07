import csv
import numpy as np
import os

from scipy.sparse import csr_matrix


DATA_DIR = "../data"
WORD_COUNTS_FILE = os.path.join(DATA_DIR, "NeurIPS_1987-2017.csv")

RANDOM_WALKS_FILE = os.path.join(DATA_DIR, "random-walks.txt")

NUM_WALKS = 32
WALK_LENGTH = 40

def build_col_pid_mapping(header_row):
    # produce mapping of column id to paper code
    col2pid = {}
    for colid, header_col in enumerate(header_row[1:]):
        col2pid[colid] = header_col
    return col2pid


def random_walk(start_rowid, S, path_length, col2pid):
    walk = []
    start = start_rowid
    # walk.append(col2pid[start])
    walk.append(start)
    while len(walk) < path_length:
        trans_probs = np.squeeze(np.asarray(S[start]))
        start = np.random.choice(
            np.arange(len(trans_probs)),
            p=trans_probs)
        # walk.append(col2pid[start])
        walk.append(start)
    return walk


# parse the word counts file and produce a sparse term document
# matrix (11463 terms x 5811 documents)
row_idxs, col_idxs, data = [], [], []
with open(WORD_COUNTS_FILE, "r") as f:
    reader = csv.reader(f)
    
    # skip header
    header_row = next(reader)
    col2pid = build_col_pid_mapping(header_row)

    num_cols = len(header_row) - 1
    for row_id, row in enumerate(reader):
        if row_id % 1000 == 0:
            print("{:d} rows converted".format(row_id))
        # skip the word, convert elements to ints
        counts = np.array([int(x) for x in row[1:]])
        # compute non-zero elements for current row
        nz_col_ids = np.nonzero(counts)[0]
        nz_data = counts[nz_col_ids]
        nz_row_ids = np.repeat(row_id, len(nz_col_ids))
        # append data to big list
        row_idxs.extend(nz_row_ids.tolist())
        col_idxs.extend(nz_col_ids.tolist())
        data.extend(nz_data.tolist())

print("{:d} rows converted, COMPLETE".format(row_id+1))

X = csr_matrix((np.array(data), 
        (
            np.array(row_idxs), 
            np.array(col_idxs)
        )
    ), shape=(row_id+1, num_cols))
print("X.shape:", X.shape)

# Construct a similarity matrix, this is going to be our
# graph represented as an adjacency matrix
S = X.T * X
print("S.shape:", S.shape)

# Normalize along rows, so each (row, col) value indicates
# the transition probability of moving from paper indicated
# by row to paper indicated by col.
S_colsum = np.sum(S, axis=1)
S = S / S_colsum

# starting from each node, construct 10 random walks
# of size 128 each. Each random walk will be a sentence and
# will be written out to the file
frand = open(RANDOM_WALKS_FILE, "w")
for i in range(S.shape[0]):
    if i % 1000 == 0:
        print("Random walks generated for {:d} nodes".format(i))
    for p in range(NUM_WALKS):
        walk = random_walk(0, S, WALK_LENGTH, col2pid)
        frand.write(" ".join([str(w) for w in walk]) + "\n")

print("Random walks generated for {:d} nodes, COMPLETE".format(i))
frand.close()
