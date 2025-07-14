# 🇫🇷➡️🇬🇧 French to English Translation using TensorFlow

This project implements a French-to-English machine translation pipeline using TensorFlow. It includes preprocessing, tokenization, dataset preparation, and vectorization of sentence pairs from a bilingual dataset. This serves as a foundational step toward building transformer-based or seq2seq models.

---

## 📁 Dataset

The dataset used is the [French-English sentence pairs](http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip) provided by TensorFlow. The text file (`fra.txt`) contains over 100,000 sentence pairs for training and evaluation.

---

## 🧠 Features

- ✅ Download and extract the dataset
- ✅ Text normalization using Unicode and Regex
- ✅ Token statistics: vocabulary size, max sentence length
- ✅ Data serialization with `pickle`
- ✅ Text vectorization using `TextVectorization` layers
- ✅ Dataset batching and formatting for model training

---

## 🧾 Preprocessing Workflow

1. **Download Dataset**
2. **Normalize Sentences**
   - Lowercasing
   - Unicode normalization
   - Regex cleanup
   - Add `[start]` and `[end]` tokens for target language
3. **Token Stats**
   - Vocabulary size
   - Sentence length
4. **Vectorization**
   - TensorFlow `TextVectorization` with a defined vocabulary size and sequence length
5. **Serialization**
   - Store processed data and vectorizer configs using `pickle`

---

## 🧪 Sample Code Snippets

### ✅ Normalize and Tokenize

```python
def normalize(line):
    line = unicodedata.normalize("NFKC", line.strip().lower())
    line = re.sub(r"^[^ \w](?!\s)", r"\1", line)
    eng, fre = line.split("\t")
    fre = "[start] " + fre + " [end]"
    return eng, fre
```

## 📦 Dependencies
```bash
tensorflow
numpy
pickle
pathlib
random
unicodedata
re
```

## 🚀 How to Run
> Clone this repository:
```bash
git clone https://github.com/jaiganesh2108/Data-Transformation-Python.git
cd Data-Transformation-Python
```
> Run the preprocessing script:
```bash
python data_transformation.py
```
You’ll get:

> text_pairs.pickle
> vectorize.pickle


