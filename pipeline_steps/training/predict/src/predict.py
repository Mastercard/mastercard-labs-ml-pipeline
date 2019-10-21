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


# TODO: Add Unit or Integration Test
import apache_beam as beam
import argparse
import datetime
import json
import logging
import os
from tensorflow.python.lib.io import file_io
from tensorflow.contrib import predictor
import numpy as np

import pandas as pd
from google.cloud import storage


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='GCS or local directory.')
    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help='GCS or local path of test file patterns.')
    parser.add_argument('--schema',
                        type=str,
                        required=True,
                        help='GCS or local json schema file path.')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='GCS or local path of model trained with tft preprocessed data.')
    parser.add_argument('--target',
                        type=str,
                        required=True,
                        help='Name of the column for prediction target.')
    parser.add_argument('--project',
                        type=str,
                        required=True,
                        help='The GCP project to run the dataflow job.')
    parser.add_argument('--mode',
                        choices=['local', 'cloud'],
                        help='whether to run the job locally or in Cloud Dataflow.')
    parser.add_argument('--batchsize',
                        type=int,
                        default=32,
                        help='Batch size used in prediction.')

    args = parser.parse_args()
    return args


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))


def run_predict(output_dir, data_path, model_export_dir):
    """Run predictions with given model using DataFlow.
    Args:
      output_dir: output folder
      data_path: test data file path.
      schema: schema list.
      target_name: target column name.
      model_export_dir: GCS or local path of exported model trained with tft preprocessed data.
      project: the project to run dataflow in.
      local: whether the job should be local or cloud.
      batch_size: batch size when running prediction.
    """
    predict_fn = predictor.from_saved_model(model_export_dir)

    df = pd.read_csv(data_path)

    data = df.drop(['ID_code', 'target'], axis=1)

    columns = list(data.columns.values)

    output_schema = []

    for column in columns:
        output_schema.append({'name': column, 'type': 'NUMBER'})

    output_schema.append({'name': 'target', 'type': 'CATEGORY'})
    output_schema.append({'name': 'predicted', 'type': 'CATEGORY'})
    output_schema.append({'name': 'false', 'type': 'NUMBER'})
    output_schema.append({'name': 'true', 'type': 'NUMBER'})

    columns = columns + ['target', 'predicted', 'false', 'true']

    results = pd.DataFrame(columns=columns)

    for index, row in data.iterrows():
        target = df.iloc[index]['target']
        tnx = data.iloc[index].values
        tnx_info = [','.join(str(e) for e in tnx)]
        prediction = predict_fn({"inputs": tnx_info})
        prediction['source'] = target
        print(prediction)
        print(target)

        ob = list(row.values)

        ob.append(target)

        ob.append((str(prediction['scores'][0][0] < prediction['scores'][0][1])).lower())
        ob.append(prediction['scores'][0][0])
        ob.append(prediction['scores'][0][1])

        results.loc[index] = ob

    print(results.head())

    print(output_schema)

    schema_dir = os.path.join(output_dir, 'schema.json')

    # with file_io.FileIO(schema_dir, 'w') as f:
    #     json.dumps(output_schema, f)

    file_io.write_string_to_file(schema_dir, json.dumps(output_schema))

    output_file_prefix = os.path.join(output_dir, 'prediction_results')

    with file_io.FileIO(output_file_prefix, 'w') as f:
        results.to_csv(f, columns=columns, header=False, index=False)

    return columns


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()
    # Models trained with estimator are exported to base/export/export/123456781 directory.
    # Our trainer export only one model.
    export_parent_dir = os.path.join(args.model, 'export', 'export')
    model_export_dir = os.path.join(export_parent_dir, file_io.list_directory(export_parent_dir)[0])

    columns = run_predict(args.output, args.data, model_export_dir)

    prediction_results = os.path.join(args.output, 'prediction_results')

    with open('/output.txt', 'w') as f:
        f.write(prediction_results)
    #
    # with open("/output.txt", "w") as text_file:
    #     text_file.write("{0}".format(prediction_results))

    # np.savetxt('/output.txt', ["%s" % prediction_results], fmt='%s')

    metadata = {
        'outputs': [{
            'type': 'table',
            'storage': 'gcs',
            'format': 'csv',
            'header': columns,
            'source': prediction_results
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()
