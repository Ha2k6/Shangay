import tensorflow as tf

(X,y),_ = tf.keras.datasets.mnist.load_data()

X = X[...,None]/255.0



m = tf.keras.Sequential([tf.keras.layers.Conv2D(8,3,activation='relu',input_shape=(28,28,1)),

                         tf.keras.layers.Flatten(), tf.keras.layers.Dense(10,activation='softmax')])



m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

m.fit(X,y,epochs=1)

print("Prediction:", tf.argmax(m.predict(X[:1])))

