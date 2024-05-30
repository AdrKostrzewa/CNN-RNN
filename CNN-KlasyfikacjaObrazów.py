import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical



# Ładowanie i przygotowanie danych
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Budowanie modelu CNN
model = Sequential()

# Warstwa konwolucyjna 1 i poolingowa 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))

# Warstwa konwolucyjna 2 i poolingowa 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Warstwa konwolucyjna 3
model.add(Conv2D(128, (3, 3), activation='relu'))

# Warstwa spłaszczająca
model.add(Flatten())

# Warstwa w pełni połączona
model.add(Dense(128, activation='relu'))

# Warstwa wyjściowa
model.add(Dense(10, activation='softmax'))





# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




# Trening modelu
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))




# Ocena modelu
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)