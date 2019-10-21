# !/usr/bin/env python3

import kfp
from kfp import components
from kfp import dsl
from kfp import gcp
from kfp import onprem

platform = 'GCP'

kubeflow_deploy_op = components.load_component_from_file('pipeline_steps/serving/deployer/component.yaml')


@dsl.pipeline(
    name='Santander Customer Transaction Prediction Release Pipeline',
    description='Example pipeline that releases the trained classification model for Santander customer transaction.'
)
def santander_transaction_classification(
        output,
        project,
):
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

    if platform == 'GCP':
        deploy = kubeflow_deploy_op(
            model_dir=str(
                'gs://kubeflow-pipelines-demo/tfx/0b22081a-ed94-11e9-81fb-42010a800160/santander-customer-transaction-prediction-95qxr-268134926/data') + '/export/export',
            server_name=tf_server_name
        )
    else:
        deploy = kubeflow_deploy_op(
            cluster_name=project,
            model_dir=str(
                'gs://kubeflow-pipelines-demo/tfx/0b22081a-ed94-11e9-81fb-42010a800160/santander-customer-transaction-prediction-95qxr-268134926/data') + '/export/export',
            pvc_name=vop.outputs["name"],
            server_name=tf_server_name
        )

    webapp = dsl.ContainerOp(
        name='webapp',
        image='us.gcr.io/kf-pipelines/ml-pipeline-webapp-launcher:v0.3',
        arguments=["--model_name", 'santanderapp']

    )
    webapp.after(deploy)

    steps = [deploy, webapp]

    for step in steps:
        if platform == 'GCP':
            step.apply(gcp.use_gcp_secret('user-gcp-sa'))
        else:
            step.apply(onprem.mount_pvc(vop.outputs["name"], 'local-storage', output))


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(santander_transaction_classification, __file__ + '.zip')
