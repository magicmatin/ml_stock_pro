
import tensorflow as tf
from handle_data.stock_calculator_metric import metric_calculator
from handle_data.stock_utils import pro_get_all_codes

print(tf.__version__)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(features):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    # feature = {
    #     'feature0': _int64_feature(feature0),
    #     'feature1': _int64_feature(feature1),
    #     'feature2': _bytes_feature(feature2),
    #     'feature3': _float_feature(feature3),
    # }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()

def stock_data_convert_tf(filename,step=30):
    with tf.python_io.TFRecordWriter(filename) as writer:
        for code in pro_get_all_codes():
            metric_data = metric_calculator(code)
            metric_data.drop(["trade_date"], axis=1, inplace=True)
            list_values = metric_data.values[19:].astype(np.float32)
            list_len = len(list_values)

            for index in range(list_len):
                start = index
                end = index + step
                if end<=list_len:
                    window_value = list_values[start:end]
                    value_bytes = window_value.tostring()
                    labels_bytes = window_value[:, 1].tostring()
                    features = {
                        "stock_feature": _bytes_feature(value_bytes),
                        "label": _bytes_feature(labels_bytes)

                    }
                    serialize_data = serialize_example(features)
                    writer.write(serialize_data)



if __name__ == '__main__':
    import numpy as np
    stock_data_convert_tf("stock_tf.tfrecode")
    # codes = pro_get_all_codes()
    # for code in codes:
    #     print(code)
    #     metric_data = metric_calculator(code)
    #     metric_data.drop(["trade_date"], axis=1, inplace=True)
    #     vs = metric_data.values[18:].astype(np.float32)
    #     step = 20
    #     filename = "test.tf"
    #     for index in range(len(vs)):
    #         if((step+index)<=len(vs)):
    #             window_value = vs[index:index+step]
    #             print(window_value)
    #             print(window_value.shape)
    #             value_bytes = window_value.tostring()
    #             labels_bytes = window_value[:, 1].tostring()
    #             print(value_bytes)
    #             features = {
    #                 "stock_feature": _bytes_feature(value_bytes),
    #                 "label": _bytes_feature(labels_bytes)
    #
    #             }
    #             serialize_data = serialize_example(features)
    #             with tf.python_io.TFRecordWriter(filename) as writer:
    #                 writer.write(serialize_data)
    #             break
    #
    #     break

