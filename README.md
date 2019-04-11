# Kubeflow Customer Transaction Prediction Demo
This is a public demo to demonstrate the capabilities of a Kubernates-based Machine learning toolkit called Kubeflow.
 Kubeflow is very handy in building end-to-end machine learning pipelines. 
 
 In this demo, we will demonstrate an end-to-end ML pipeline that starts with preporcessing and training the model 
 till serving the trained model via a web frontend.
 
## Getting the data
This demo uses the anonymized customer transaction data from Santander. To the demo code running you'll need to go to
 their [Kaggle competition page](https://www.kaggle.com/c/santander-customer-transaction-prediction), then go to the 
 data section and finally accept the competition rule to be able to download the data.
 ### Update Dockerfile.model to use your download data
 you'll need to update Dockerfile.model to point to the download data. Specially, you'll need to update 
 `input/santander-ctp/train.csv` part of this line: 
 ```
 ADD input/santander-ctp/train.csv /opt/train.csv
 ```
  
  you'll need to leave `/opt/train.csv` unchanged.
 
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
`PROJECT_ID`. You'll also need to activate the required Services for the project(i.e. Kubernetes Engine).

Next up, you'll need to use kubeflow click and deploy UI to create a kubeflow deployment on google cloud.

Open https://deploy.kubeflow.cloud/#/deploy and Fill in the following values in the resulting form:

1. Project: Enter your GCP $PROJECT_ID in the top field
2. Deployment name: Set the default value to `kfdemo-deployment`. Alternatively, set `$DEPLOYMENT_NAME` in the 
makefile to a different value.
3. GKE Zone: Use the value you have set for $ZONE, selecting it from the pulldown.
4. Kubeflow Version: v0.4.1
5. Check the Skip IAP box


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

The output of this target should be a list of the components with tf-serving as one of thme.

### Creating the bucket needed for storing data/models <a name="creating-the-bucket-needed-for-storing-datamodels"></a>

We need a bucket to be able to export the trained models to it. executing the following command via `make 
create-gcs-bucket` will create a bucket with the name stored in `BUCKET_NAME` variable.
```
create-gcs-bucket:
	gsutil mb gs://$(BUCKET_NAME)/
```
### Training the model <a name="training-the-model"></a>

Now we have the `kubectl` tool configured to make deployments against the cluster and a bucket to export the trained 
model we need to to do the first step in our ML pipeline which is preprocessing and training on the data. In this 
section we are building the image the contains the trainer code, test the image locally, push the image to google 
cloud docker registry and finally deploy this training job to kubernetes.


The following target will build the image from the docker file in the `WORKING_DIR` and tag the image with google 
cloud docker registry. In this case it's `TRAIN_PATH :=us.gcr.io/$(PROJECT_ID)/kubeflow-train`
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
Nextt up we'll need to authorize docker to access google cloud registry and be able to push images to it:
```
#authorize docker to access GCR and push train image
push-train-image: build-train-image
	gcloud auth configure-docker --quiet
	docker push $(TRAIN_PATH)
```

Now we have the training image we can set the parameters for the `train` tfjob by pointing to the ksonnet app 
directory and setting the required parameters.
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
```

Finally, use the following target `make deploy-train-job` will execute the previous targets first and then deploy the
 train tfjob to kubernetes.

```
# deploy the training image to kubeflow
deploy-train-job: push-train-image set-train-params
	cd $(KS_NAME) && \
	ks apply default -c train

```

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

TensorBoard can now be accessed at `http://127.0.0.1:$(TB_HOST_PORT)`

### Training at Scale <a name="training-at-scale"></a>
TODO
### Serving the trained model <a name="serving-the-trained-model"></a>

The Next step in the ML pipeline is to deploy the trained model using `TensorFlow TFX` or more specificly `TF Serving`

Executing the following target via `make dpeloy-serving-job` will deploy the serving job

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

Finally, we'll need to build, push and deploy the frontend image.

Execute the following target to build the image and you can override the image `TAG` by different version. 
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
 app. This param include image name that we built earlier.
 
```
#setting frontend parameters
set-frontend-params:
	cd $(KS_NAME) && \
	ks param set web-ui image $(FRONTEND_PATH):$(TAG) && \
	ks param set web-ui type LoadBalancer
```

Finally you can execute `make deploy-frontend` command to deploy the `web-ui` component.
```
# deploy the frontend
deploy-frontend: set-frontend-params push-frontend-image
	cd $(KS_NAME) && \
	ks apply default -c web-ui
```

To be able to access the web-ui from externally(outside of the kubernetes cluster), we'll need to execute the 
following target, get the external IP of the output and use it in the browser to access the frontend.
```
# get external IP for the frontend service
frontend-external-ip:
	kubectl get service web-ui
```

A success message for the connection between the frontend and the serving pod will be shown in the frontend. 
Otherwise, you won't be able to connect to the serving pod and test the trained model.

### Cleaning all <a name="cleaning-all"></a>
Most importantly is to delete all the created resources on google cloud after demonstrating the full working ML 
pipeline

```
# delete the cluster and other resouces provisioned by kubeflow
clean-all:
	rm -r $(KS_NAME)
	gcloud deployment-manager deployments delete $(DEPLOYMENT_NAME)
	gsutil rm -r gs://$(BUCKET_NAME)
	gcloud container images delete us.gcr.io/$(PROJECT_ID)/kubeflow-train
	gcloud container images delete $(FRONTEND_PATH):$(TAG)
```
