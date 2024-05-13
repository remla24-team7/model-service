from joblib import dump
from tensorflow.keras.preprocessing.text import Tokenizer

# check data locations with extraction from drive
with open("dataset/train.txt", "r", encoding="utf-8") as file:
    train = [line.strip() for line in file.readlines()[1:]]
raw_x_train = [line.split("\t")[1] for line in train]
raw_y_train = [line.split("\t")[0] for line in train]

with open("dataset/test.txt", "r", encoding="utf-8") as file:
    test = [line.strip() for line in file.readlines()]
raw_x_test = [line.split("\t")[1] for line in test]
raw_y_test = [line.split("\t")[0] for line in test]

with open("dataset/val.txt", "r", encoding="utf-8") as file:
    val = [line.strip() for line in file.readlines()]
raw_x_val = [line.split("\t")[1] for line in val]
raw_y_val = [line.split("\t")[0] for line in val]
tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
char_index = tokenizer.word_index
SEQUENCE_LENGTH = 200

dump(tokenizer, 'tokenizer.joblib')