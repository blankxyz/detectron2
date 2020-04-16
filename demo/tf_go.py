import tensorflow as tf

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
print(dataset)
dataset = dataset.flat_map(lambda window: window.batch(5))
print(dataset)
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
print(dataset)