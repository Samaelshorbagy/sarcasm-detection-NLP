import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential

# Intialize Data
sent=[]
head=[]
url=[]


# load dataset
with open("news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", "r") as f:
    # each line is a JSON object
    for line in f:
        item = json.loads(line)
        sent.append(item['headline'])
        head.append(item['is_sarcastic'])
        url.append(item['article_link'])


# 80% for training
train_size = int(len(sent) * 0.8)   

train_sent=sent[:train_size]
test_sent=sent[train_size:]

train_labels=head[:train_size]
test_labels=head[train_size:]
# convert labels to np arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# set token parameters
trunc_type='post'
pad_type='post'
oov_token="<OOV>"
max_len=100
embedding_dim=16
vocab_size=10000

tokenizer = Tokenizer(oov_token= "<OOV>", num_words=vocab_size)
tokenizer.fit_on_texts(train_sent)
word_index = tokenizer.word_index

# convert al headlines le sequence of tokens
train_seq=tokenizer.texts_to_sequences(train_sent)
train_padded=pad_sequences(train_seq, padding=pad_type, truncating=trunc_type, maxlen=max_len)

test_seq=tokenizer.texts_to_sequences(test_sent)
test_padded=pad_sequences(test_seq, padding=pad_type, truncating=trunc_type, maxlen=max_len)

# Model
model=Sequential([
    # convert words into dense vectors for input into neural networks
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    # capture sequential dependencies in both back & forward directions
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )
model.summary()

# Train
history = model.fit(
    train_padded, 
    train_labels, 
    epochs=10, 
    validation_data=(test_padded, test_labels), 
    verbose=2
)

# test_loss, test_acc = model.evaluate(test_padded, test_labels, verbose=2)
# print(f"Test Accuracy: {test_acc:.4f}")


def predict_sarcasm(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, padding='post', maxlen=train_padded.shape[1])

    prediction = model.predict(padded)
    predicted_value = prediction[0][0]

    if predicted_value >= 0.5:
        print(f"Sarcastic: {sentence}")
    else:
        print(f"Not Sarcastic: {sentence}")

test_sentence_1 = "Oh great, another rainy day. Just what I needed!"
test_sentence_2 = "I love spending hours stuck in traffic."

# Make predictions on test sentences
predict_sarcasm(test_sentence_1)
predict_sarcasm(test_sentence_2)        

# print(train_padded[0])
# print(train_padded.shape)