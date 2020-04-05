from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime

import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model.summary()
print("\n")

class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {} \n'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))
callbacks = [ PrintLR() ]

print("Training Set Size: ", len(train_images))
start = datetime.now()
model.fit(train_images, train_labels, epochs=20, callbacks=callbacks,
                    validation_data=(test_images, test_labels))
print("Training Time: ", datetime.now()-start)
print("\n")

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print("Test Set Size: ", len(test_images))
print("Loss: ", test_loss)
print("Accuracy %: ", test_acc * 100.)
