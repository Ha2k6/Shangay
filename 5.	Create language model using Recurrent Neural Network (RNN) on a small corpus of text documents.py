from tensorflow.keras.models import *

from tensorflow.keras.layers import *

from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np



text=["i love ai","i love ml","ai loves data"]

t=Tokenizer(); t.fit_on_texts(text)

seq=np.array(t.texts_to_sequences(text))



X,Y=seq[:,:-1], seq[:,-1]

m=Sequential([Embedding(10,8),SimpleRNN(16),Dense(10,activation="softmax")])

m.compile(loss="sparse_categorical_crossentropy",optimizer="adam")

m.fit(X,Y,epochs=50,verbose=0)



print(t.index_word[np.argmax(m.predict(X[0:1]))])

