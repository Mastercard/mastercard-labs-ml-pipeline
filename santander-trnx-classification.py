# !/usr/bin/env python3

import kfp
from kfp import components
from kfp import dsl
from kfp import gcp
from kfp import onprem

platform = 'GCP'

dataflow_tf_transform_op = components.load_component_from_file('pipeline_steps/preprocessing/tft/component.yaml')
tf_train_op = components.load_component_from_file('pipeline_steps/training/dnntrainer/component.yaml')
dataflow_tf_predict_op = components.load_component_from_file('pipeline_steps/training/predict/component.yaml')

confusion_matrix_op = components.load_component_from_file('pipeline_steps/metrics/confusion_matrix/component.yaml')
roc_op = components.load_component_from_file('pipeline_steps/metrics/roc/component.yaml')

kubeflow_deploy_op = components.load_component_from_file('pipeline_steps/kubeflow/deployer/component.yaml')


@dsl.pipeline(
    name='Santander Customer Transaction Prediction',
    description='Example pipeline that does classification with model analysis based on Santander customer transaction dataset.'
)
def santander_transaction_classification(
        output,
        project,
        train='gs://kubeflow-pipelines-demo/dataset/train.csv',
        evaluation='gs://kubeflow-pipelines-demo/dataset/test.csv',
        mode='local',
        preprocess_module='gs://kubeflow-pipelines-demo/dataset/preprocessing.py',
        learning_rate=0.1,
        hidden_layer_size='1500',
        steps=3000
):
    output_template = str(output) + '/{{workflow.uid}}/{{pod.name}}/data'
    target_class_lambda = """lambda x: x['target']"""

    tf_server_name = 'kfdemo-service'

    if platform != 'GCP':
        vop = dsl.VolumeOp(
            name="create_pvc",
            resource_name="pipeline-pvc",
            modes=dsl.VOLUME_MODE_RWM,
            size="1Gi"
        )

        checkout = dsl.ContainerOp(
            name="checkout",
            image="alpine/git:latest",
            command=["git", "clone", "https://github.com/kubeflow/pipelines.git", str(output) + "/pipelines"],
        ).apply(onprem.mount_pvc(vop.outputs["name"], 'local-storage', output))
        checkout.after(vop)

    preprocess = dataflow_tf_transform_op(
        training_data_file_pattern=train,
        evaluation_data_file_pattern=evaluation,
        schema="not.txt",
        gcp_project=project,
        run_mode=mode,
        preprocessing_module=preprocess_module,
        transformed_data_dir=output_template
    )

    training = tf_train_op(
        transformed_data_dir=preprocess.output,
        schema='not.txt',
        learning_rate=learning_rate,
        hidden_layer_size=hidden_layer_size,
        steps=steps,
        target='tips',
        preprocessing_module=preprocess_module,
        training_output_dir=output_template
    )

    prediction = dataflow_tf_predict_op(
        data_file_pattern=evaluation,
        schema='not.txt',
        target_column='tips',
        model=training.outputs['training_output_dir'],
        run_mode=mode,
        gcp_project=project,
        predictions_dir=output_template
    )

    cm = confusion_matrix_op(
        predictions=prediction.outputs['predictions_dir'],
        output_dir=output_template
    )

    roc = roc_op(
        predictions_dir=prediction.outputs['predictions_dir'],
        target_lambda=target_class_lambda,
        output_dir=output_template
    )

    steps = [training, prediction, cm, roc]
    for step in steps:
        if platform == 'GCP':
            step.apply(gcp.use_gcp_secret('user-gcp-sa'))
        else:
            step.apply(onprem.mount_pvc(vop.outputs["name"], 'local-storage', output))


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(santander_transaction_classification, __file__ + '.zip')
