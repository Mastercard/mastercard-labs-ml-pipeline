"""
Training module using TF Boosted Trees Classifier to predict if Customer will make specific
transaction in the future or not!
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd

import json
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--transformed-data-dir',
                        type=str,
                        required=True,
                        help='GCS path containing tf-transformed training and eval data.')

    parser.add_argument('--job-dir',
                        type=str,
                        required=True,
                        help='GCS or local directory.')

    # parser.add_argument('--tf-export-dir',
    #                     type=str,
    #                     default='export/',
    #                     help='GCS path or local directory to export model')

    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.01,
                        help='Learning rate for training.')

    parser.add_argument('--n-trees',
                        type=int,
                        default=100,
                        help='Number of trees.')

    parser.add_argument('--max-depth',
                        type=int,
                        default=6,
                        help='Maximum depth for the trees.')

    parser.add_argument('--train-start',
                        type=int,
                        default=0,
                        help='Start index of train examples within the data.')

    parser.add_argument('--train-count',
                        type=int,
                        default=200,
                        help='Number of train examples within the data.')

    parser.add_argument('--eval-start',
                        type=int,
                        default=200,
                        help='Start index of eval examples within the data.')

    parser.add_argument('--eval-count',
                        type=int,
                        default=200,
                        help='Number of eval examples within the data.')

    parser.add_argument('--optimizer',
                        choices=['Adam', 'SGD', 'Adagrad'],
                        default='Adagrad',
                        help='Optimizer for training. If not provided, '
                             'tf.estimator default will be used.')
    parser.add_argument('--hidden-layer-size',
                        type=str,
                        default='100',
                        help='comma separated hidden layer sizes. For example "200,100,50".')
    parser.add_argument('--steps',
                        type=int,
                        help='Maximum number of training steps to perform. If unspecified, will '
                             'honor epochs.')
    parser.add_argument('--epochs',
                        type=int,
                        help='Maximum number of training data epochs on which to train. If '
                             'both "steps" and "epochs" are specified, the training '
                             'job will run for "steps" or "epochs", whichever occurs first.')
    parser.add_argument('--preprocessing-module',
                        type=str,
                        required=False,
                        help=('GCS path to a python file defining '
                              '"preprocess" and "get_feature_columns" functions.'))
    parser.add_argument('--schema',
                        type=str,
                        required=True,
                        help='GCS json schema file path.')
    parser.add_argument('--target',
                        type=str,
                        required=True,
                        help='The name of the column to predict in training data.')

    args = parser.parse_args()

    return args


def read_ct_data(train_start, train_count, eval_start, eval_count, data_path):
    """
    Read a Santander training data
    :param train_start: Start index for the training set
    :param train_count: Number of instances to be used for training
    :param eval_start: Start index for the eval set
    :param eval_count: Number of instances to be used for evaluation
    :return: The required division between train and test set
    """
    data = pd.read_csv(data_path)

    # Dropping the id column
    data.drop(['ID_code'], axis=1, inplace=True)

    data = data.values
    return (data[train_start:train_start + train_count],
            data[eval_start:eval_start + eval_count])


def make_inputs_from_np_arrays(features_np, label_np):
    """Makes and returns input_fn and feature_columns from numpy arrays.
    The generated input_fn will return tf.data.Dataset of feature dictionary and a
    label, and feature_columns will consist of the list of
    tf.feature_column.BucketizedColumn.
    Note, for in-memory training, tf.data.Dataset should contain the whole data
    as a single tensor. Don't use batch.
    Args:
      features_np: A numpy ndarray (shape=[batch_size, num_features]) for
          float32 features.
      label_np: A numpy ndarray (shape=[batch_size, 1]) for labels.
    Returns:
      input_fn: A function returning a Dataset of feature dict and label.
      feature_names: A list of feature names.
      feature_column: A list of tf.feature_column.BucketizedColumn.
    """
    num_features = features_np.shape[1]
    features_np_list = np.split(features_np, num_features, axis=1)
    # 1-based feature names.
    feature_names = ["feature_%02d" % (i + 1) for i in range(num_features)]

    # Create source feature_columns and bucketized_columns.
    def get_bucket_boundaries(feature):
        """Returns bucket boundaries for feature by percentiles."""
        return np.unique(np.percentile(feature, range(0, 100))).tolist()

    source_columns = [
        tf.feature_column.numeric_column(
            feature_name, dtype=tf.float32,
            default_value=0.0)
        for feature_name in feature_names
    ]

    bucketized_columns = [
        tf.feature_column.bucketized_column(
            source_columns[i],
            boundaries=get_bucket_boundaries(features_np_list[i]))
        for i in range(num_features)
    ]

    # Make an input_fn that extracts source features.
    def input_fn():
        """Returns features as a dictionary of numpy arrays, and a label."""
        features = {
            feature_name: tf.constant(features_np_list[i])
            for i, feature_name in enumerate(feature_names)
        }
        return tf.data.Dataset.zip((tf.data.Dataset.from_tensors(features),
                                    tf.data.Dataset.from_tensors(label_np),))

    return input_fn, feature_names, bucketized_columns


def make_eval_inputs_from_np_arrays(features_np, label_np):
    """Makes eval input as streaming batches."""
    num_features = features_np.shape[1]
    features_np_list = np.split(features_np, num_features, axis=1)

    # 1-based feature names.
    feature_names = ["feature_%02d" % (i + 1) for i in range(num_features)]

    def input_fn():
        features = {
            feature_name: tf.constant(features_np_list[i])
            for i, feature_name in enumerate(feature_names)
        }
        return tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(features),
            tf.data.Dataset.from_tensor_slices(label_np),)).batch(1000)

    return input_fn


def _make_csv_serving_input_receiver_fn(column_names, column_defaults):
    """Returns serving_input_receiver_fn for csv.
    The input arguments are relevant to `tf.decode_csv()`.
    Args:
      column_names: a list of column names in the order within input csv.
      column_defaults: a list of default values with the same size of
          column_names. Each entity must be either a list of one scalar, or an
          empty list to denote the corresponding column is required.
          e.g. [[""], [2.5], []] indicates the third column is required while
              the first column must be string and the second must be float/double.
    Returns:
      a serving_input_receiver_fn that handles csv for serving.
    """

    def serving_input_receiver_fn():
        csv = tf.placeholder(dtype=tf.string, shape=[None], name="csv")
        features = dict(zip(column_names, tf.decode_csv(csv, column_defaults)))
        receiver_tensors = {"inputs": csv}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    return serving_input_receiver_fn


def main():
    # tf.logging.set_verbosity(tf.logging.INFO)

    args = parse_arguments()

    train_data, eval_data = read_ct_data(args.train_start, args.train_count, args.eval_start, args.eval_count,
                                         'gs://kubeflow-pipelines-demo/dataset/train.csv')

    train_input_fn, feature_names, feature_columns = make_inputs_from_np_arrays(
        features_np=train_data[:, 1:], label_np=train_data[:, 0:1])

    eval_input_fn = make_eval_inputs_from_np_arrays(
        features_np=eval_data[:, 1:], label_np=eval_data[:, 0:1])

    print("Training starting...")
    classifier = tf.estimator.BoostedTreesClassifier(
        feature_columns,
        n_batches_per_layer=1,
        model_dir=args.job_dir,
        n_trees=args.n_trees,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate)

    classifier.train(train_input_fn)

    print("Training Finished Successfully...")

    eval_results = classifier.evaluate(eval_input_fn)

    print("Export saved model...")

    # export_dir = args.tf_export_dir

    export_dir = os.path.join(args.job_dir, 'export/export')

    classifier.export_saved_model(
        export_dir,
        _make_csv_serving_input_receiver_fn(
            column_names=feature_names,
            # columns are all floats.
            column_defaults=[[0.0]] * len(feature_names)))

    print("Done exporting the model...")

    metadata = {
        'outputs': [{
            'type': 'tensorboard',
            'source': args.job_dir,
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    with open('/output.txt', 'w') as f:
        f.write(args.job_dir)


if __name__ == '__main__':
    main()
