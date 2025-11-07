import tensorflow as tf, matplotlib.pyplot as plt

(x,_),_=tf.keras.datasets.mnist.load_data(); x=(x/255.).reshape(-1,28,28,1)

G=tf.keras.Sequential([tf.keras.layers.Dense(784,activation='relu',input_shape=(100,)),tf.keras.layers.Reshape((28,28,1))])

D=tf.keras.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(1,activation='sigmoid')])

D.compile('adam','binary_crossentropy'); D.fit(x,tf.ones((len(x),1)),epochs=1,verbose=0)

z=tf.random.normal([16,100]); fake=G(z).numpy()

for i in fake: plt.imshow(i[:,:,0],cmap='gray'); plt.show()

