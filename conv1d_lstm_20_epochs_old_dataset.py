# -*- coding: utf-8 -*-
"""Conv1d_Lstm_20_epochs_Old_Dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xzVXXstrC_wJ6qx4gPapLwguixJG222D
"""

pip install keras tensorflow

from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import logging
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaxPooling1D, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

data = pd.read_csv('/content/deceptive-opinion.csv')

data.head()

df = data[['deceptive','polarity','text']]                            #df[['C', 'D', 'E']]

df1 = df.sample(frac=1)

df1.shape

import matplotlib.pyplot as plt
my_tags = ['deceptive','truthful']
plt.figure(figsize=(10,4))
df1.deceptive.value_counts().plot(kind='bar');

fig, ax = plt.subplots()

ax.scatter(df1['deceptive'], df1['polarity'])
# set a title and labels
ax.set_title('Deceptive Opinion Spam Dataset')
ax.set_xlabel('Deceptive')
ax.set_ylabel('Polarity')

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

df1.info()

df1.head()

df2=df1.drop(columns=['polarity'],axis=1)

df2.describe()

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df2['deceptive']= label_encoder.fit_transform(df2['deceptive'])

df2['deceptive'].unique()

# prompt: in the above label encoder code extart what is 1, 0 and their corresponding labels

print(list(label_encoder.inverse_transform([0,1])))

df2.head()

df2.describe()

from nltk.corpus import stopwords

nltk.download("stopwords")

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower().split()
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+\-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text

df2['text'] = df2['text'].apply(clean_text)

X = df2.text
y = df2.deceptive

X = X.map(lambda a: clean_text(a))

def text_to_wordlist(text):
    #Remove Special Characters
    text = re.sub(r'[^a-z\d ]', " ", text)
    text = re.sub(r'\d+', '_num_', text)
    return(text)

X = X.map(lambda a: text_to_wordlist(a))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(len(x_test))
print(len(y_test))

print(len(x_train))
print(len(y_train))

tokenizer = Tokenizer(num_words=None,lower=True,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',split=' ',char_level=False)

tokenizer.fit_on_texts(X)

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(X)

vocab_size = len(word_index)
print('Vocab size: {}'.format(vocab_size))
longest = max(len(seq) for seq in X)
print("Longest comment size: {}".format(longest))
average = np.mean([len(seq) for seq in X])
print("Average comment size: {}".format(average))
stdev = np.std([len(seq) for seq in X])
print("Stdev of comment size: {}".format(stdev))
max_len = int(average + stdev * 3)
print('Max comment size: {}'.format(max_len))

from tensorflow.keras.preprocessing.sequence import pad_sequences

processed_x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
processed_x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')

processed_x_train.shape

print(processed_x_train)
print(processed_x_test)

print('x_train shape:', processed_x_train.shape)
print('x_test shape:', processed_x_test.shape)

print(len(y_train))

import tensorflow.keras.backend
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D, BatchNormalization, LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Embedding
# from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

import os

embeddings_index = {}
f = open(os.path.join('../input/glove-global-vectors-for-word-representation', '/content/drive/MyDrive/yelp_review_polarity_csv/yelp_review_polarity_csv/glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
k = 0
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        k += 1
        embedding_matrix[i] = embedding_vector

"""2 LSTM + 2 CONV1D"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Initialize model
model_2lstm = Sequential()

# Add layers
model_2lstm.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim,
                    weights=[embedding_matrix], input_length=max_len, trainable=True))
model_2lstm.add(LSTM(60, return_sequences=True, name='lstm_layer1',
               kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
model_2lstm.add(LSTM(30, return_sequences=True, name='lstm_layer2',
               kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
model_2lstm.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model_2lstm.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model_2lstm.add(MaxPooling1D(pool_size=3))
model_2lstm.add(GlobalMaxPooling1D())
model_2lstm.add(BatchNormalization())
model_2lstm.add(Dense(64, activation='relu', kernel_regularizer=l2(0.17), bias_regularizer=l2(0.01)))
model_2lstm.add(Dropout(0.3))
model_2lstm.add(Dense(32, activation='sigmoid', kernel_regularizer=l2(0.19), bias_regularizer=l2(0.01)))
model_2lstm.add(Dropout(0.3))
model_2lstm.add(Dense(1, activation='sigmoid'))

# Compile and summarize
model_2lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_2lstm.build(input_shape=(None, max_len))  # Explicitly define input shape
model_2lstm.summary()

model_2lstm.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

history = model_2lstm.fit(processed_x_train,y_train,validation_data=(processed_x_test,y_test),epochs=20,batch_size=32,verbose=1)

loss, accuracy = model_2lstm.evaluate(processed_x_test, y_test, batch_size=32)

from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = model_2lstm.predict(processed_x_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

from sklearn.metrics import roc_curve, auc
y_pred = model_2lstm.predict(processed_x_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_keras = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re

def predict_truthfulness(text):

    cleaned_text = clean_text(text)
    cleaned_text = text_to_wordlist(cleaned_text)

    x_input = tokenizer.texts_to_sequences([cleaned_text])
    x_input = pad_sequences(x_input, maxlen=max_len, padding='post', truncating='post')

    # Make prediction
    prediction = model_2lstm.predict(x_input)[0][0]

    if prediction >= 0.5:
      predicted_class = "truthful"
    else:
      predicted_class = "deceptive"


    return f"Prediction: {predicted_class}"

input_text = input("Enter the text to analyze: ")
result = predict_truthfulness(input_text)
result

"""# **2 LAYER LSTM AND CONV1d**"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Initialize model
model = Sequential()

# Add layers
model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim,
                    weights=[embedding_matrix], input_length=max_len, trainable=True))
model.add(LSTM(60, return_sequences=True, name='lstm_layer1',
               kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
model.add(LSTM(30, return_sequences=True, name='lstm_layer2',
               kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(GlobalMaxPooling1D())
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.17), bias_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='sigmoid', kernel_regularizer=l2(0.19), bias_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile and summarize
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, max_len))  # Explicitly define input shape
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

history = model.fit(processed_x_train,y_train,validation_data=(processed_x_test,y_test),epochs=20,batch_size=32,verbose=1)

loss, accuracy = model.evaluate(processed_x_test, y_test, batch_size=32)

from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = model.predict(processed_x_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

from sklearn.metrics import roc_curve, auc
y_pred = model.predict(processed_x_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_keras = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re

def predict_truthfulness(text):

    cleaned_text = clean_text(text)
    cleaned_text = text_to_wordlist(cleaned_text)

    x_input = tokenizer.texts_to_sequences([cleaned_text])
    x_input = pad_sequences(x_input, maxlen=max_len, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(x_input)[0][0]

    if prediction >= 0.5:
      predicted_class = "truthful"
    else:
      predicted_class = "deceptive"


    return f"Prediction: {predicted_class}"

input_text = input("Enter the text to analyze: ")
result = predict_truthfulness(input_text)
result

"""# **1 LAYER LTSM AND CONV1D**"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Initialize the model
model_1lstm = Sequential()

# Add layers
model_1lstm.add(Embedding(input_dim=vocab_size + 1,
                          output_dim=embedding_dim,
                          weights=[embedding_matrix],
                          input_length=max_len,
                          trainable=True))
model_1lstm.add(LSTM(60, return_sequences=True, name='lstm_layer1'))
model_1lstm.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model_1lstm.add(MaxPooling1D(pool_size=3))
model_1lstm.add(GlobalMaxPooling1D())
model_1lstm.add(BatchNormalization())
model_1lstm.add(Dense(64, activation='relu', kernel_regularizer=l2(0.17), bias_regularizer=l2(0.01)))
model_1lstm.add(Dropout(0.3))
model_1lstm.add(Dense(32, activation='sigmoid', kernel_regularizer=l2(0.19), bias_regularizer=l2(0.01)))
model_1lstm.add(Dropout(0.3))
model_1lstm.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
model_1lstm.add(Dropout(0.3))
model_1lstm.add(Dense(1, activation='sigmoid'))


model_1lstm.build(input_shape=(None, max_len))
model_1lstm.summary()

model_1lstm.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

history = model_1lstm.fit(processed_x_train,y_train,validation_data=(processed_x_test,y_test),epochs=20,batch_size=32,verbose=1)

loss, accuracy = model_1lstm.evaluate(processed_x_test, y_test, batch_size=32)

from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = model_1lstm.predict(processed_x_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

from sklearn.metrics import roc_curve, auc
y_pred = model_1lstm.predict(processed_x_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_keras = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re

def predict_truthfulness(text):

    cleaned_text = clean_text(text)
    cleaned_text = text_to_wordlist(cleaned_text)

    x_input = tokenizer.texts_to_sequences([cleaned_text])
    x_input = pad_sequences(x_input, maxlen=max_len, padding='post', truncating='post')

    # Make prediction
    prediction = model_1lstm.predict(x_input)[0][0]

    if prediction >= 0.5:
      predicted_class = "truthful"
    else:
      predicted_class = "deceptive"


    return f"Prediction: {predicted_class}"

input_text = input("Enter the text to analyze: ")
result = predict_truthfulness(input_text)
result

# prompt: make auc roc curve f all models in one diagram to compare

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming y_test is your true labels and y_pred is the predicted probabilities from model_2lstm
y_pred_2lstm = model_2lstm.predict(processed_x_test).ravel()
fpr_2lstm, tpr_2lstm, thresholds_2lstm = roc_curve(y_test, y_pred_2lstm)
auc_2lstm = auc(fpr_2lstm, tpr_2lstm)

# Assuming y_pred is the predicted probabilities from model
y_pred = model.predict(processed_x_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_model = auc(fpr, tpr)


# Assuming y_pred is the predicted probabilities from model_1lstm
y_pred_1lstm = model_1lstm.predict(processed_x_test).ravel()
fpr_1lstm, tpr_1lstm, thresholds_1lstm = roc_curve(y_test, y_pred_1lstm)
auc_1lstm = auc(fpr_1lstm, tpr_1lstm)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_2lstm, tpr_2lstm, label='2 LSTM + 2 Conv1D (area = {:.3f})'.format(auc_2lstm))
plt.plot(fpr, tpr, label='2 LSTM + 1 Conv1D (area = {:.3f})'.format(auc_model))
plt.plot(fpr_1lstm, tpr_1lstm, label='1 LSTM + 1 Conv1D (area = {:.3f})'.format(auc_1lstm))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

