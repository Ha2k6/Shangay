from sklearn.datasets import fetch_lfw_people

import tensorflow as tf

d = fetch_lfw_people(min_faces_per_person=20, resize=0.3)

X, y = d.images[...,None]/255.0, d.target



m = tf.keras.Sequential([tf.keras.layers.Conv2D(8,3,activation='relu',input_shape=X[0].shape),

                         tf.keras.layers.Flatten(), tf.keras.layers.Dense(len(d.target_names),activation='softmax')])



m.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

m.fit(X, y, epochs=1);  print(d.target_names[tf.argmax(m.predict(X[:1]))])

