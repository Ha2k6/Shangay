from tensorflow.keras.models import Sequential; from tensorflow.keras.layers import *

from tensorflow.keras.preprocessing.text import Tokenizer; from tensorflow.keras.preprocessing.sequence import pad_sequences; import numpy as np



t=["real news","gov update","fake scam","hoax fake"]; y=np.array([1,1,0,0])

tok=Tokenizer(); tok.fit_on_texts(t); X=pad_sequences(tok.texts_to_sequences(t),5)



m=Sequential([Embedding(input_dim=30, output_dim=8, input_length=5), LSTM(8), Dense(1,activation="sigmoid")])

m.compile("adam","binary_crossentropy"); m.fit(X,y,epochs=20,verbose=0)



s="fake scam"; S=pad_sequences(tok.texts_to_sequences([s]),5)

print("REAL" if m.predict(S)[0][0]>0.5 else "FAKE")

