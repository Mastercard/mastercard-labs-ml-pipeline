# End-to-end  Santander Customer Transaction Prediction Demo Using Kubeflow Pipelines
This is a public demo to demonstrate the capabilities of Kubeflow Pipelines which is one Kubeflow components that could
be used to orchestrate an end-to-end real world ML application.
 
 In this demo, we will demonstrate two ML pipelines:
 
 1. `Training Pipeline:` This one will be mainly used to acquire and preprocess the data, training Boosted Trees classifier, 
 evaluate the trained model and finally calculate some metrics based on the trained model such ROC and Confusion Matrix.
 2. `Release Pipeline:` This pipeline will serve the trained model using TFX and then deploy a web frontend that operates
 on top of this serving and do live inference agaist the model.
 
## Getting the data
This demo uses the anonymized customer transaction data from Santander. To the demo code running you'll need to go to
 their [Kaggle competition page](https://www.kaggle.com/c/santander-customer-transaction-prediction), then go to the 
 data section and finally accept the competition rule to be able to download the data.
 
 ### Uploading the dataset to Google Bucket
 You'll need to upload the downloaded dataset in the same downloaded format to Google Bucket. So you'll 
 need to have one created
 
## Deploying the pipeline to GCP
In this section, we'll use deploy our entire pipeline to Google Cloud. The following will be covered here:

1. [Creating and connecting to the cluster](#creating-and-connecting-to-the-cluster)
2. [Training Pipeline](#training-pipeline)
3. [Release Pipeline](#release-pipeline)
8. [Cleaning all](#cleaning-all)


### Creating and connecting to the cluster <a name="creating-and-connecting-to-the-cluster"></a>

Before you start you'll need to have project created in Google Cloud and we'll refer to this project by the variable 
`PROJECT_ID`.

Next up, you'll need to use kubeflow click and deploy [web UI](https://deploy.kubeflow.cloud/#/deploy) to create a kubeflow deployment on google cloud.

Fill in the following values in the resulting form:

1. Project: Enter your GCP $PROJECT_ID in the top field
2. Deployment name: Set the default value to `kfdemo-deployment`. Alternatively, set `$DEPLOYMENT_NAME` in the 
makefile to a different value.
3. Choose Login with Username Password from choose how to connect to kubeflow service list
4. Create a username and password
5. GKE Zone: Use the value you have set for $ZONE, selecting it from the pulldown.
6. Kubeflow Version: v0.6.2


After it shows you in the log console that the deployment is ready, it will open a web UI for you where you can enter 
the username and password that you used while creating the cluster. Once you see the central kubeflow dashboard you can 
use the following target to connect the local `kubectl` tool to the remote one:

```bash
make connect-to-cluster
```

This target will execute the following commands:

```
# Setting the environment variables for GKE
set-gcloud-project:
	gcloud config set project $(PROJECT_ID)


# Configuring kubectl to connect to the cluster
connect-to-cluster: set-gcloud-project
	gcloud container clusters get-credentials $(DEPLOYMENT_NAME) --zone $(ZONE) --project $(PROJECT_ID)
	kubectl config set-context $(shell kubectl config current-context) --namespace kubeflow
	kubectl get nodes
```

These command will connect the local setup of `kubectl` to the GCP cluster and you should be able to see the cluster 
nodes as a final output for this target.


### Training Pipeline <a name="training-pipeline"></a>

Now we have the `kubectl` tool configured. We can now make deployments against the cluster. To execute the training pipeline,
you'll need to navigate to `notebooks` folder and open `santander_training_pipeline.ipynd` notebook in `jupyter`. 

The notebook is self-explained. So at the end you'll submit the compiled pipeline from the notebook to Kubeflow Pipelines UI, you can then 
go to the dahsboard open `Experiments` section and you should see the `Training` experiment created and inside it you should see the 
submitted execution run that we did from the notebook.

### Release Pipeline <a name="release-pipeline"></a>

Once we have the training pipeline completed and the model exported to google bucket, we can now serve the trained model
and deploy web frontend on top of it.

To do this, you'll need to navigate to `notebooks` folder and open `santander_release_pipeline.ipynd` file in `jupter` and 
run the steps till you reach the submission step. 

Once you submit the pipeline you should be able to see it inside the `Release` experiment in Kubeflow Pipelines UI.

When the release pipeline finishes the execution, you can run the last cell which will get the external IP address for this
` LoadBalancer` service.

Once Kubernetes assign external IP address to this service. You should copy it and paste in a browser window to see the deployed
frontend.

To be able to access the web-ui from externally(outside of the kubernetes cluster), we'll need to execute the 
following target, get the external IP of the output and use it in the browser to access the frontend.


### Cleaning all <a name="cleaning-all"></a>
Most importantly is to delete all the created resources on google cloud after demonstrating the full working ML 
pipeline. To do this run:

```bash
make clean-all
```

Which will execute the following commands

```
# delete the cluster and other resouces provisioned by kubeflow
clean-all:
	rm -r $(KS_NAME)
	gcloud deployment-manager deployments delete $(DEPLOYMENT_NAME)
	gsutil rm -r gs://$(BUCKET_NAME)
	gcloud container images delete us.gcr.io/$(PROJECT_ID)/kubeflow-train
	gcloud container images delete $(FRONTEND_PATH):$(TAG)
```
