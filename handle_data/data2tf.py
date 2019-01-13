import tensorflow as tf
import numpy as np

print(tf.__version__)

tfrecords_filename = "tfrecords/train.tfrecords"

#writer = tf.python_io.TFRecordWriter(tfrecords_filename)



# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))




# the number of observations in the dataset
n_observations = int(1e4)

# boolean feature, encoded as False or True
feature0 = np.random.choice([False, True], n_observations)

# integer feature, random between -10000 and 10000
feature1 = np.random.randint(0, 5, n_observations)

# bytes feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)


def serialize_example(feature0, feature1, feature2, feature3):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




# This is an example observation from the dataset.

example_observation = []

x = np.random.randn(100).astype(np.float32)
x=x.tostring()
serialized_example = serialize_example(False, 4,x, 0.9876)
print(serialized_example)
example_proto = tf.train.Example.FromString(serialized_example)

print(example_proto)

