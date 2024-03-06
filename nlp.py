import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

# Load dataset (e.g., sentiment analysis dataset)
data = pd.read_csv('sentiment_data.csv')

# Preprocess dataset
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['label'], test_size=0.2)

# Define LSTM model
model = models.Sequential([
    layers.Embedding(10000, 16, input_length=100),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)
