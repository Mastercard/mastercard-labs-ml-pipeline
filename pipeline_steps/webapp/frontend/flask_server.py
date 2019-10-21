import logging

from flask import Flask, render_template, request
from ctp_client import get_prediction, random_transaction

app = Flask(__name__)


# handle requests to the server
@app.route("/")
def main():
    # get url parameters for HTML template
    name_arg = request.args.get('name', 'kfdemo-service')
    addr_arg = request.args.get('addr', 'kfdemo-service')
    port_arg = request.args.get('port', '9000')
    args = {"name": name_arg, "addr": addr_arg, "port": port_arg}
    logging.info("Request args: %s", args)

    output = None
    connection = {"text": "", "success": False}
    try:

        # Get random transaction
        tnx_info, target = random_transaction()
        # get prediction from TensorFlow server
        pred, scores = get_prediction(tnx_info, server_host=addr_arg,
                                      server_port=int(port_arg),
                                      timeout=10, server_name=name_arg)
        # if no exceptions thrown, server connection was a success
        connection["text"] = "Connected (model version: " + str(1) + ")"
        connection["success"] = True
        # parse class confidence scores from server prediction
        scores_dict = []
        for i in range(0, 2):
            scores_dict += [{"index": str(i), "val": scores[i]}]
        output = {"truth": target, "prediction": pred,
                  "tnx_info": tnx_info, "scores": scores_dict}
    except Exception as e:  # pylint: disable=broad-except
        logging.info("Exception occured: %s", e)
        # server connection failed
        connection["text"] = "Exception making request: {0}".format(e)
    # render results using HTML template
    return render_template('index.html', output=output,
                           connection=connection, args=args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format=('%(levelname)s|%(asctime)s'
                                '|%(pathname)s|%(lineno)d| %(message)s'),
                        datefmt='%Y-%m-%dT%H:%M:%S',
                        )
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting flask.")
    app.run(debug=False, port=8080, host='0.0.0.0')
