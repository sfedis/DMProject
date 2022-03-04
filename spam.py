import pandas as pd
import numpy as np
import gensim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer   #stemming
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


def recall_m(
    y_true, y_pred
) -> float:
    """Calculates Recall given the actual value and the predicted value
    
    Parameters:
        y_true: Actual value
        y_pred: Predicted value
        
    Retruns:
        recall: Returns float value of Recall
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(
    y_true, y_pred
) -> float:
    """Calculates Precision given the actual value and the predicted value
    
    Parameters:
        y_true: Actual value
        y_pred: Predicted value
        
    Retruns:
        recall: Returns float value of Precision
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(
    y_true, y_pred
) -> float:
    """Calculates F1 score given the actual value and the predicted value
    
    Parameters:
        y_true: Actual value
        y_pred: Predicted value
        
    Retruns:
        recall: Returns float value of F1 score
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Import data
data = pd.read_csv('spam_or_not_spam.csv')
data = data.dropna()

X =  data['email'].values
y = data['label'].values

#Encoding labels spam (1) and not spam (0)
lebel = LabelEncoder()
y = lebel.fit_transform(y)
y = y.reshape(-1,1)

#Constructing corpus from stemmed data
corpus = []
for i in range(len(X)):
    stemmer = PorterStemmer()
    X[i] = X[i].lower()
    X[i] = X[i].split()
    email =  [stemmer.stem(J) for J in X[i] ]
    email = ' '.join(email)
    corpus.append(email)

#Spliting our data
X_train, X_test, y_train, y_test = train_test_split(corpus,y,test_size=0.25,random_state=2)

#documents is an array with all the words in our stemmed data
documents = [text.split() for text in X_train]

#Building and training word to vec model
w2v_model = gensim.models.Word2Vec(vector_size=300, window=3, min_count=3, workers=4)
w2v_model.build_vocab(documents)
w2v_model.train(documents, total_examples=len(documents), epochs=32)

#Tokenizing our stemmed data for training
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1

#Convert our stemmed data to array-like object for training our neural network
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=300)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=300)
print(X_train)

#Constructing embedding_matrix
embedding_matrix = np.zeros((vocab_size, 300))
print(embedding_matrix)
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

#embedding_layer is the first layer of our neural network
embedding_layer = Embedding(vocab_size, 300, 
                            weights=[embedding_matrix], 
                            input_length=300, 
                            trainable=False)

#Building the neural network
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

#Compiling model, defining metrics from custom functions
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy', f1_m, precision_m, recall_m])

callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='accuracy', min_delta=1e-4, patience=3)]

#Training the model and getting the evaluations
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=8,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)

loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print(f'F1 score: {f1_score} \nPrecesion: {precision} \nRecall: {recall}')

