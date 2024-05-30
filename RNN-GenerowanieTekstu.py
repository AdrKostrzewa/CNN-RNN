import numpy as np
import tensorflow as tf


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Przykładowy tekst
text = """Once upon a time, in a land far, far away, there was a small village. In this village lived a ..."""

# Parametry
seq_length = 40  # długość sekwencji wejściowej
step = 3  # krok przesunięcia sekwencji

# Przygotowanie sekwencji wejściowych i docelowych
sentences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

# Tokenizacja
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(sentences)
next_char_indices = tokenizer.texts_to_sequences(next_chars)

# One-hot encoding
X = to_categorical(sequences, num_classes=len(tokenizer.word_index) + 1)
y = to_categorical(next_char_indices, num_classes=len(tokenizer.word_index) + 1)


# Parametry modelu
input_shape = (seq_length, len(tokenizer.word_index) + 1)

# Budowanie modelu
model = Sequential()
model.add(LSTM(128, input_shape=input_shape))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Trening modelu
model.fit(X, y, epochs=10, batch_size=128)

import random
def generate_text(model, tokenizer, seed_text, length):
    result = seed_text
    for _ in range(length):
        # Przygotowanie danych wejściowych
        encoded = tokenizer.texts_to_sequences([result[-seq_length:]])[0]
        encoded = to_categorical([encoded], num_classes=len(tokenizer.word_index) + 1)

        # Przewidywanie następnego znaku
        predicted_index = np.argmax(model.predict(encoded), axis=-1)[0]

        # Dodanie przewidywanego znaku do wyniku
        for char, index in tokenizer.word_index.items():
            if index == predicted_index:
                result += char
                break

    return result


# Przykład generowania tekstu
seed_text = "Once upon a time"
generated_text = generate_text(model, tokenizer, seed_text, 100)
print(generated_text)
