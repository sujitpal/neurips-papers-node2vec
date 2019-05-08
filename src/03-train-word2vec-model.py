import gensim
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = "../data"
RANDOM_WALKS_FILE = os.path.join(DATA_DIR, "random-walks.txt")
MODEL_FILE = os.path.join(DATA_DIR, "w2v-nips-papers.model")

class Documents(object):
    def __init__(self, input_file):
        self.input_file = input_file

    def __iter__(self):
        with open(self.input_file, "r") as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    if i % 1000 == 0:
                        logging.info("{:d} random walks extracted".format(i))
                yield line.strip().split()


docs = Documents(RANDOM_WALKS_FILE)
model = gensim.models.Word2Vec(
    docs,
    size=128,    # size of embedding vector
    window=10,   # window size
    sg=1,        # skip-gram model
    min_count=2,
    workers=4
)
model.train(
    docs, 
    total_examples=model.corpus_count,
    epochs=10)
model.save(MODEL_FILE)
