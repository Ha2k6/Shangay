import tensorflow as tf, numpy as np



words=["I","eat","rice"]; tags=["PRON","VERB","NOUN"]

X=np.array([[0,1,2]]); Y=np.array([[0,1,2]])



enc=tf.keras.layers.LSTM(8, return_state=True)

dec=tf.keras.layers.LSTM(8, return_sequences=True)

dense=tf.keras.layers.Dense(len(tags))



_,h,c = enc(tf.one_hot(X,10))

out = dense(dec(tf.one_hot(Y,10), initial_state=[h,c])).numpy()   # (1,seq,ntags)

pred = np.argmax(out, axis=-1)[0]

for w,p in zip(words, pred): print(w, "→", tags[p])

