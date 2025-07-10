import tensorflow as tf
import pathlib
import random
import unicodedata
import re


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
    line = re.sub(r"^[^\w](?!\s)",r"\1",line)
    line = re.sub(r"(\s[^\w])(?!\s)",r"\1",line)
    line = re.sub(r"(?!\s)([^\w]$)",r"\1",line)
    line = re.sub(r"(?!\s)([^\w]\s)",r"\1",line)
