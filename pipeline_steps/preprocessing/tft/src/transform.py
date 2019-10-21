# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import apache_beam as beam
import argparse
import datetime
import csv
import json
import logging
import os
import pandas as pd
from tensorflow.python.lib.io import file_io

# Inception Checkpoint
INCEPTION_V3_CHECKPOINT = 'gs://cloud-ml-data/img/flower_photos/inception_v3_2016_08_28.ckpt'
INCEPTION_EXCLUDED_VARIABLES = ['InceptionV3/AuxLogits', 'InceptionV3/Logits', 'global_step']

DELIMITERS = '.,!?() '
VOCAB_SIZE = 100000


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='GCS or local directory.')
    parser.add_argument('--train',
                        type=str,
                        required=True,
                        help='GCS path of train file patterns.')
    parser.add_argument('--eval',
                        type=str,
                        required=True,
                        help='GCS path of eval file patterns.')
    parser.add_argument('--schema',
                        type=str,
                        required=True,
                        help='GCS json schema file path.')
    parser.add_argument('--project',
                        type=str,
                        required=True,
                        help='The GCP project to run the dataflow job.')
    parser.add_argument('--mode',
                        choices=['local', 'cloud'],
                        help='whether to run the job locally or in Cloud Dataflow.')
    parser.add_argument('--preprocessing-module',
                        type=str,
                        required=False,
                        help=('GCS path to a python file defining '
                              '"preprocess" and "get_feature_columns" functions.'))

    args = parser.parse_args()
    return args


def run_transform(output_dir, train_data_file, eval_data_file):
    """Writes a tft transform fn, and metadata files.
    Args:
      output_dir: output folder
      schema: schema list.
      train_data_file: training data file pattern.
      eval_data_file: eval data file pattern.
      project: the project to run dataflow in.
      local: whether the job should be local or cloud.
      preprocessing_fn: a function used to preprocess the raw data. If not
                        specified, a function will be automatically inferred
                        from the schema.
    """

    data = pd.read_csv(train_data_file)

    output_file_prefix = os.path.join(output_dir, 'preprocess_data.csv')

    print(data.head(10))

    # with file_io.FileIO(output_file_prefix, 'w') as f:
    #     data.to_csv(f, columns=list(data.columns.values), header=False, index=False)


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()

    run_transform(args.output, args.train, args.eval)

    with open('/output.txt', 'w') as f:
        f.write(args.output)


if __name__ == "__main__":
    main()
