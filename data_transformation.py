import tensorflow as tf
import pathlib
import random
import unicodedata
import re
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np

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
text_pair = text_pair[n_train:n_train+n_val]

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

train_eng = [pair[0] for pair in train_pair]
train_fre = [pair[1] for pair in train_pair]

eng_vect.adapt(train_eng)
eng_vect.adapt(train_fre)

with open('vectorize.pickle', 'wb') as fp:
    data = {'train' : train_pair,
            'test' : test_pair,
            'eng_vect' : eng_vect.get_config(),
            'fre_vect' : fre_vect.get_config(),
            'eng_weight' : eng_vect.get_weights(),
            'fre_weight' : fre_vect.get_weights(),
            }
    pickle.dump(data, fp)

with open("vectorize.pickle" , 'rb') as fp:
    data = pickle.load(fp)

train_pair  = data['train']
test_pair = data['test']

eng_vect = TextVectorization.from_config(data['eng_vect'])
eng_vect.set_weights(data['eng_weights'])
fre_vect = TextVectorization.frrom_config(data['fre_vect'])
fre_vect.set_weights(data['fre_weights'])

def format_dataset(eng, fre):
    eng = eng_vect(eng)
    fre = fre_vect(fre)

    source = {'encode_inp' : eng,
            'decode_inp' : fre[:, :-1],
            }
    target = fre[:, 1:]
    return (source, target)

def make_dataset(pairs, batchsize = 64): 
    eng_text, fre_text = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices(list(eng_text),list(fre_text))
    
    return dataset.shuffle(2048).batch(batchsize).map(format_dataset).prefetch(16).cache()

train_ds = make_dataset(train_pair)
for inputs,target in train_ds.take(1):
    print(inputs['encode_inp'].shape)
    print(inputs['encode_inp'][0])
    print(inputs['encode_inp'].shape)
    print(inputs['encode_inp'][0])
    print(target.shape)
    print(target[0])

test_ds = make_dataset(test_pair)

# positional embedding

def pos_enc_matrix(L, d, n = 10000):
    assert d%2 == 0
    d2 = d//2

    P = np.zeros(L,d)
    k = np.arrange(L).reshape(-1,1)
    i = np.arrange(d2).reshape(-1,1)

    denom = np.power(n, -i/d2)
    args = k *denom

    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)

class positionalEmbedding(tf.keras.layers.layer):
    def __init__(self, seq_length, vocab_size, embed_dim, **kwargs):
        super.__init__(**kwargs)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeddings = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embed_dim, mask_zero = True)
        matrix = post_enc_matrix(seq_lenght, embed_dim)
        self.positional_embedding = tf.constant(matrix,dtype='float32')

def call(self, input):
    embedded_tokens = self.token_embeddings(inputs)
    return embedded_tokens + self.positional_embedding

def compute_mask(self, *args, **kwargs):
    return self.token_embeddings.computer_mask(*args, **kwargs)

