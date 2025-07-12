import tensorflow as tf
import pathlib
import random
import unicodedata
import re
import pickle
from tensorflow.keras.layers import TextVectorization


text_file = tf.keras.utils.get_file(
    fname = 'fra-eng.zip',
    origin = "http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zib",
    extract = True,
    )
text_file = pathlib.Path(text_file).parent / 'fra.txt'
print(text_file)
with open(text_file) as fp:
    text_pair = [line for line in fp]
for _ in range(5):
    print(random.choice(text_pair))
    
def normalize(line):
    line = unicodedata.normalize("NFKC", line.strip().lower())
    line = re.sub(r"^[^ \w](?!\s)",r"\1",line)
    line = re.sub(r"(\s[^ \w])(?!\s)",r"\1",line)
    line = re.sub(r"(?!\s)([^ \w]$)",r"\1",line)
    line = re.sub(r"(?!\s)([^ \w]\s)",r"\1",line)
    eng, fre = line.split("\t")
    fre = '[start] ' + fre +' [end]'
    return eng, fre

with open(text_file) as fp:
    text_pairs = [normalize(line) for line in fp]

for _ in range(5):
    print(random.choice(text_pairs))

eng_tokens, fre_tokens = set(), set()
eng_maxlen, fre_maxlen = 0, 0
for eng, fre in text_pairs:
    eng_token, fre_token = eng.split(), fre.split()
    eng_maxlen = max(eng_maxlen, len(eng_token))
    fre_maxlen = max(eng_maxlen, len(fre_token))
    eng_tokens.update(eng_token)
    fre_tokens.update(fre_token)

print(f"total tokens in english {len(eng_tokens)}")
print(f"total tokens in french {len(fre_tokens)}")
print(f"maximum length of line is {eng_maxlen}")
print(f"maximum length of line is {fre_maxlen}")

with open("text_pairs,pickle", "wb") as fp:
    pickle.dump(text_pairs,fp)

# Embadding layer
# positional layer
# attention layer

with open("text_pairs.picle", 'rb') as fp:
    text_pairs = pickle.load(fp)

random.shuffle(text_pairs)
n_val = int(.15*len(text_pairs))
n_train = len(text_pairs) - 2*n_val
train_pair = text_pairs[:n_train]
text_pair = text_pair[n_train+n_val:]

vocab_en = 10000 
vocab_fr = 20000
seq_length = 25

eng_vect = TextVectorization(
    max_tokens = vocab_en,
    standardize = None,
    split = 'whitespace',
    output_mode = 'int',
    output_sequece_length = seq_length 
)

fre_vect = TextVectorization(
    max_tokens = vocab_en,
    standardize = None,
    split = 'whitespace',
    output_mode = 'int',
    output_sequece_length = seq_length 
)