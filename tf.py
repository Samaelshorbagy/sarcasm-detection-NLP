import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

    
sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!',
    'What do you love about dog and cat?'
]

tokenizer = Tokenizer(num_words = 100, oov_token= "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

seq=tokenizer.texts_to_sequences(sentences)
padded=pad_sequences(seq, padding='post',truncating='post', maxlen=3)

# test_data=[
#     'do you love my dog?',
#     'all i do is love'
# ]
# test_seq=tokenizer.texts_to_sequences(test_data)

print(word_index)
print(seq)
# print(test_seq)
print(padded)