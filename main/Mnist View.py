import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('Train', x_train.shape, y_train.shape)
print('Test', x_test.shape, y_test.shape)

plt.imshow(x_train[1], cmap='gray')
plt.show()