# Kubeflow Customer Transaction Prediction Demo
This is a public demo to demonstrate the capabilities of a Kubernetes-based Machine learning toolkit called Kubeflow.
 Kubeflow is very handy in building end-to-end machine learning pipelines. 
 In this demo, we will demonstrate an end-to-end ML pipeline that starts with preporcessing and training the model 
 till serving the trained model via a web frontend.
 
## Getting the data
This demo uses the anonymized customer transaction data from Santander. To get the demo code running you'll need to go to
 their [Kaggle competition page](https://www.kaggle.com/c/santander-customer-transaction-prediction), then go to the 
 data section and finally accept the competition rule to be able to download the data.
 ### Update Dockerfile.model to use your download data
 you'll need to update Dockerfile.model to point to the download data. Specially, you'll need to update 
 `input/santander-ctp/train.csv` part of this line: 
 ```
 ADD input/santander-ctp/train.csv /opt/train.csv
 ```
  
 you'll need to leave `/opt/train.csv` unchanged.
 

## Creating an account on GCP 

You'll need to have an unlocked Google Cloud account to run this tutorial. You can create your account at this address
 [GCloud](https://cloud.google.com) using the free trial.  You will get 300$ which will be more than enough to run everything.
 However, because deploying the cluster requires more CPUs that are made availabe for the free trial, you'll have to unlock your account.
 We suggest that you put an alert on the spending so that when you are running low on funds, GCP will let you know and you won't verspend

## Deploying the pipeline to GCP
In this section, we'll use deploy our entire pipeline to Google Cloud. The following will be covered here:

1. [Creating and connecting to the cluster](#creating-and-connecting-to-the-cluster)
2. [Creating the Ksonnet app](#creating-the-ksonnet-app)
3. [Creating the bucket needed for storing data/models](#creating-the-bucket-needed-for-storing-datamodels)
3. [Training the model](#training-the-model)
4. [Monitoring the training job](#monitoring-the-training-job)
5. [Training at Scale](#training-at-scale)
6. [Serving the trained model](#serving-the-trained-model)
7. [Deploying frontend](#deploying-frontend)
8. [Cleaning all](#cleaning-all)


### Creating and connecting to the cluster <a name="creating-and-connecting-to-the-cluster"></a>

Before you start you'll need to have project created in Google Cloud and we'll refer to this project by the variable 
`PROJECT_ID`. You'll also need to activate the required Services for the project(i.e. Kubernetes Engine). You don't need
to create the cluster yourself, the script that deploy kubeflow will take care of this automatically. You will be ask to
select a zone in which your project will be deployed (by default `us-central1-a`), we will refer to it by $ZONE in the future.

Next up, you'll need to use kubeflow click and deploy UI to create a kubeflow deployment on google cloud.
Open https://deploy.kubeflow.cloud/#/deploy and fill in the following values in the resulting form:

1. Project: Enter your GCP $PROJECT_ID in the top field
2. Deployment name: Set the default value to `kfdemo-deployment`. Alternatively, set `$DEPLOYMENT_NAME` in the 
makefile to a different value.
3. Choose how to connect: Setup Endpoint later.
4. GKE Zone: Use the value you have set for $ZONE, selecting it from the pulldown.
5. Kubeflow Version: v0.5. If not available, choose the most recent version

After it shows you in the log console that the deployment is ready, you can use `make connect-to-cluster` to execute 
the following targets:


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

### Creating the Ksonnet app <a name="creating-the-ksonnet-app"></a>

Next, we'll need to configure Ksonnet app and configure it to point to the newly created cluster. After initializing 
the app, we'll copy the preconfigured components of an old ksonnet and add Kubeflow to Ksonnet registry to be able to
 install Kubeflow specific component.
 
```
# create and configure ksonnet app
init-ks-app:
	ks init $(KS_NAME) && \
	cd $(KS_NAME) && \
	ks env list && \
	cp $(WORKING_DIR)/demo_ks_app/components/* $(WORKING_DIR)/$(KS_NAME)/components && \
	ks registry add kubeflow github.com/kubeflow/kubeflow/tree/$(KF_VERSION)/kubeflow && \
    ks pkg install kubeflow/tf-serving@$(KF_VERSION) && \
	ks component list
```

The output of this target should be a list of the components with tf-serving as one of them.

### Creating the bucket needed for storing data/models <a name="creating-the-bucket-needed-for-storing-datamodels"></a>

In order to have storage space where we will have our data and model, we need to create a bucket. Calling the target `create-gcs-bucket` 
will create a bucket with the name stored in the variable `BUCKET_NAME`.

```
create-gcs-bucket:
	gsutil mb gs://$(BUCKET_NAME)/
```

### Training the model <a name="training-the-model"></a>

We now have the `kubectl` tool configured to make deployments against the cluster and a bucket to store our data and our trained model. 
We can finally proceed with the first step of our ML pipeline: Preprocessing and Training on the data. 
In this section we will build the container that will hold the code to train our model. When the image is build, 
we will test it locally and proceed to push it to google cloud docker registry before finally deploying the training job to kubernetes.

The following target will build the image from the docker file in the `WORKING_DIR` and tag the image with google 
cloud docker registry. In our case `TRAIN_PATH` corresponds to `TRAIN_PATH:=us.gcr.io/$(PROJECT_ID)/kubeflow-train`
```
#build the tensorflow model into a container
build-train-image:
	docker build $(WORKING_DIR) -t $(TRAIN_PATH) -f $(WORKING_DIR)/Dockerfile.model
```

We can use the following target to test that the image is working locally and the training is working before pushing 
and deploying it to the cluster.

```
# Test training image locally
test-train-image-local:
	docker run -it $(TRAIN_PATH)
```

Next, we'll need to authorize docker to access google cloud registry and be able to push images to it:
```
#authorize docker to access GCR and push train image
push-train-image: build-train-image
	gcloud auth configure-docker --quiet
	docker push $(TRAIN_PATH)
```
<!-- COMMENT: when arrived here, user will have already run the build-train-image so we might want to remove the build-train-image from
the target -->

Now that we have the training image, we can set the parameters for the `train` tfjob by pointing to the ksonnet app 
directory and setting the required parameters. By running the target `deploy-train-job` we will execute the target `set-train-params` that
will setup the parameters and deploy the tfjob to kubernetes

```
#set the parameters for the tfjob
set-train-params:
	cd $(KS_NAME) && \
	ks param set train image $(TRAIN_PATH) && \
	ks param set train name "my-train-1" && \
	ks param set train modelDir gs://$(BUCKET_NAME)/$(MODEL_PATH) && \
	ks param set train exportDir gs://$(BUCKET_NAME)/$(MODEL_PATH)/export && \
	ks param set train secret user-gcp-sa=/var/secrets && \
	ks param set train envVariables GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/user-gcp-sa.json

# deploy the training image to kubeflow
deploy-train-job: push-train-image set-train-params
	cd $(KS_NAME) && \
	ks apply default -c train

```

<!-- COMMENT: Same as before, we might want to remove the push-train-image from the target -->

### Monitoring the training job <a name="monitoring-the-training-job"></a>

We can use `TensorBoard` to visualize the model accuracy progression over time. We just need to set `logDir` 
parameter of the `tensorboard` component.

```
# setting tensorboard params
set-tensorboard-params:
	cd $(KS_NAME) && \
	ks param set tensorboard logDir $(LOGDIR) && \
	ks param set tensorboard secret user-gcp-sa=/var/secrets && \
	ks param set tensorboard envVariables GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/user-gcp-sa.json

# deploy tensorboard
deploy-tensorboard: set-tensorboard-params
	cd $(KS_NAME) && \
	ks show default -c tensorboard && \
	ks apply default -c tensorboard
```

Executing `make deploy-tensorboard` will deploy tensorboard component but we need to access from local host. To do 
that you can execute the following target which port-forward the `tensorboard service` to local host
```
TB_HOST_PORT:=8090

# port-forwarding tensorboard to be accessed locally
access-tensorboard-local:
	kubectl port-forward service/tensorboard-tb $(TB_HOST_PORT):80
	@echo TensorBoard can now be accessed at http://127.0.0.1:$(TB_HOST_PORT)
```

<!-- COMMENT: Wasn't able to make this part work somehow, but I didn't push too hard, I'll get back on it latter -->


TensorBoard can now be accessed at `http://127.0.0.1:$(TB_HOST_PORT)`

### Training at Scale <a name="training-at-scale"></a>
TODO
### Serving the trained model <a name="serving-the-trained-model"></a>

The next step in the ML pipeline is to deploy the trained model using `TensorFlow TFX` or more specificly `TF Serving`.
Executing the following target via `make deloy-serving-job` will set the parameters required and deploy the serving job.

```
# model serving
set-model-serving-params:
	cd $(KS_NAME) && \
	ks param set kfdemo-deploy-gcp modelBasePath gs://$(BUCKET_NAME)/${MODEL_PATH}/export && \
	ks param set kfdemo-deploy-gcp modelName kfdemo

# deploy the training image to kubeflow
deploy-serving-job: set-model-serving-params
	cd $(KS_NAME) && \
	ks apply default -c kfdemo-deploy-gcp && \
	ks apply default -c kfdemo-service
```

### Deploying frontend <a name="deploying-frontend"></a>

Finally, we'll build, push and deploy the frontend image.
By executing the target `push-frontend-image`, you will build the image and push it to gcloud container registery. You can override the image `TAG` by different version. 

```
#Building the frontend container
FRONTEND_PATH :=us.gcr.io/$(PROJECT_ID)/kubeflow-frontend
TAG:=v0.3

build-frontend:
	docker build $(WORKING_DIR)/frontend -t $(FRONTEND_PATH):$(TAG)


# Authorize docker and push the frontend image
push-frontend-image: build-frontend
	gcloud auth configure-docker --quiet
	docker push $(FRONTEND_PATH):$(TAG)

```

After building and pushing the image, we'll need to set the required params for the `web-ui` component in the ksonnet
 app. By executing the command `make deploy-frontend`, it will set the frontend parameters and deploy the `web-ui` component.
 
```
#setting frontend parameters
set-frontend-params:
	cd $(KS_NAME) && \
	ks param set web-ui image $(FRONTEND_PATH):$(TAG) && \
	ks param set web-ui type LoadBalancer

# deploy the frontend
deploy-frontend: set-frontend-params push-frontend-image
	cd $(KS_NAME) && \
	ks apply default -c web-ui
```


To be able to access the web-ui from outside of the kubernetes cluster, we'll need to execute the 
following target to get the external IP of the output and use it in the browser to access the frontend.

```
# get external IP for the frontend service
frontend-external-ip:
	kubectl get serving-the-trained-modelvice web-ui
```

A success message for the connection between the frontend and the serving pod will be shown in the frontend. 
Otherwise, you won't be able to connect to the serving pod and test the trained model.

### Cleaning all <a name="cleaning-all"></a>
To delete all the resources created on google cloud after running this tutorial, you can execute the following command `make clean-all`

```
# delete the cluster and other resouces provisioned by kubeflow
clean-all:
	rm -r $(KS_NAME)
	gcloud deployment-manager deployments delete $(DEPLOYMENT_NAME)
	gsutil rm -r gs://$(BUCKET_NAME)
	gcloud container images delete us.gcr.io/$(PROJECT_ID)/kubeflow-train
	gcloud container images delete $(FRONTEND_PATH):$(TAG)
```
