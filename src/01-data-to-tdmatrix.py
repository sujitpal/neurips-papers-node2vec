import collections
import csv
import gensim
import nltk
import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = "../data"

PAPERS_FILE = os.path.join(DATA_DIR, "papers.csv")

ID_TITLE_FILE = os.path.join(DATA_DIR, "NeurIPS_id-title.csv")
TDMATRIX_FILE = os.path.join(DATA_DIR, "NeurIPS_1987-2017.csv")

english_stopwords = set(stopwords.words("english"))
id2title = {}
td_matrix = collections.Counter()
num_processed = 0
with open(PAPERS_FILE, "rU") as f:
    reader = csv.reader((line.replace('\0', '') for line in f),
                        delimiter=',')
    for row in reader:
        id, pub_yr, title, _, _, abs_text, full_text = row
        if id == "id":
            continue
        if num_processed % 1000 == 0:
            print("{:d} papers processed".format(num_processed))
        id2title[int(id)] = "{:s} ({:s})".format(title, pub_yr)
        # join words from title*5, abstract*2, full_text
        contents = " ".join([title, title, title, title, title,
            abs_text, abs_text, full_text])
        tokens = gensim.utils.simple_preprocess(contents, deacc=True)
        tokens = [t for t in tokens if t not in english_stopwords]
        for token in tokens:
            td_key = " ".join([id, token])
            td_matrix[td_key] += 1
        num_processed += 1

print("{:d} papers processed, COMPLETE".format(num_processed))

print("raw vocab size: {:d}".format(len(td_matrix)))        

# minimum term frequency > 5
td_matrix = {k:v for k, v in td_matrix.items() 
                 if v > 5}
print("after pruning tf > {:d}, vocab size: {:d}".format(
    5, len(td_matrix))
)

# maximum doc frequency < 0.2 of corpus size
num_docs = len(id2title)
docfreqs = {}
for k in td_matrix.keys():
    id, term = k.split()
    if term in docfreqs.keys():
        docfreqs[term].append(id)
    else:
        docfreqs[term] = [id]

docfreqs = [(k, len(v)) for k, v in docfreqs.items()]
freq_terms = [k for (k, v) in docfreqs if v > 0.2 * num_docs]
remove_keys = []
for freq_term in freq_terms:
    for k in td_matrix.keys():
        if k.endswith(" " + freq_term):
            remove_keys.append(k)
for k in remove_keys:
    del td_matrix[k]
print("after pruning df > {:d}%, vocab size: {:d}".format(
    20, len(td_matrix))
)

# reformat to same format as NIPS_1987-2015.csv file from UCI ML repo
ids = sorted(list(id2title.keys()))
tokens = set()
for k in td_matrix.keys():
    _, token = k.split()
    tokens.add(token)
tokens = sorted(list(tokens))

num_written, num_total = 0, len(tokens)
fout = open(TDMATRIX_FILE, "w")
for token in tokens:
    if num_written % 100000 == 0:
        print("{:d}/{:d} token counts written".format(
            num_written, num_total
        ))
    cols = [token]
    for id in ids:
        key = " ".join([str(id), token])
        try:
            count = td_matrix[key]
        except KeyError:
            count = 0
        cols.append(str(count))
    fout.write("{:s}\n".format(",".join(cols)))
    num_written += 1

print("{:d}/{:d} token counts written, COMPLETE".format(
    num_written, num_total
))
fout.close()

fout = open(ID_TITLE_FILE, "w")
for id, title in id2title.items():
    fout.write("{:d}\t{:s}\n".format(id, title))
fout.close()
