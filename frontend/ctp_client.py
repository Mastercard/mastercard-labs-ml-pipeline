# !/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with ctp model.
"""

from __future__ import print_function

from __future__ import print_function

from grpc.beta import implementations

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import pandas as pd

import random

def get_prediction(tnx_info, server_host='127.0.0.1', server_port=8500, timeout=100.0, server_name='kfdemo'):
    """
    Retrieve a prediction from a TensorFlow model server

    :param tnx_info:       an annonymised transaction details
    :param server_host: the address of the TensorFlow server
    :param server_port: the port used by the server
    :param server_name: the name of the server
    :param timeout:     the amount of time to wait for a prediction to complete
    :return 0:          the integer predicted in the transaction
    :return 1:          the confidence scores for all classes
    """

    print("connecting to:%s:%i" % (server_host, server_port))
    # initialize to server connection
    channel = implementations.insecure_channel(server_host, server_port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # build request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = server_name
    request.model_spec.signature_name = 'predict'
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(tnx_info))

    # retrieve results
    result = stub.Predict(request, timeout)
    resultVal = result.outputs["classes"].string_val[0]
    scores = result.outputs['probabilities'].float_val
    return resultVal, scores


def random_transaction():
    """
    Generating a random transaction from test file
    :return: Strin value of a Random Transaction
    """
    data = pd.read_csv('/home/test.csv')

    # Dropping ID and target columns
    data.drop(['ID_code', 'target'], axis=1, inplace=True)

    # Generating random number between 0 and 200
    tnx_index = random.randint(0,200)

    # Getting the required row in the data
    tnx = data.iloc[tnx_index].values

    # Convert the row into CS String value
    tnx_info = [','.join(str(e) for e in tnx)]

    return tnx_info
