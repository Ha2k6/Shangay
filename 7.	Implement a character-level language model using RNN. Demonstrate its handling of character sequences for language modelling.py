from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import SimpleRNN, Dense

import numpy as np



text = "hello world"

chars = sorted(set(text))

c2i = {c:i for i,c in enumerate(chars)}

i2c = {i:c for c,i in c2i.items()}



seq = np.array([[c2i[c] for c in text[:-1]]])

Y   = np.array([c2i[text[-1]]])



m = Sequential([SimpleRNN(16,input_shape=(len(text)-1,1)), Dense(len(chars),activation="softmax")])

m.compile("adam","sparse_categorical_crossentropy")

m.fit(seq.reshape(1,-1,1),Y,epochs=100,verbose=0)



print("Predicted char:", i2c[np.argmax(m.predict(seq.reshape(1,-1,1)))])

