# you can override variables do
# make ${TARGET} ${VAR}=${VALUE}


# Default project ID, zone and deployment name
PROJECT_ID :=kfdemo-233821
ZONE :=us-central1-a
DEPLOYMENT_NAME :=kfdemo-deployment
KS_NAME :=test_ks_app
BUCKET_NAME :=$(KS_NAME)-$(PROJECT_ID)

# logdir for tensorboard
LOGDIR=gs://${BUCKET_NAME}/${MODEL_PATH}
MODEL_PATH:=my-model

#set the path on GCR for pushing the images that will be used by kubeflow
TRAIN_PATH :=us.gcr.io/$(PROJECT_ID)/kubeflow-train


TAG := $(shell date +v%Y%m%d)
all: build


#Kubeflow version
KF_VERSION:=v0.4.1

WORKING_DIR := $(shell pwd)

# Setting the environment variables for GKE
set-gcloud-project:
	gcloud config set project $(PROJECT_ID)


# Configuring kubectl to connect to the cluster
connect-to-cluster: set-gcloud-project
	gcloud container clusters get-credentials $(DEPLOYMENT_NAME) --zone $(ZONE) --project $(PROJECT_ID)
	kubectl config set-context $(shell kubectl config current-context) --namespace kubeflow
	kubectl get nodes

# create and configure ksonnet app
init-ks-app:
	ks init $(KS_NAME) && \
	cd $(KS_NAME) && \
	ks env list && \
	cp $(WORKING_DIR)/demo_ks_app/components/* $(WORKING_DIR)/$(KS_NAME)/components && \
	ks registry add kubeflow github.com/kubeflow/kubeflow/tree/$(KF_VERSION)/kubeflow && \
    ks pkg install kubeflow/tf-serving@$(KF_VERSION) && \
	ks component list


# Creating bucket for holding the training data and exported models
create-gcs-bucket:
	gsutil mb gs://$(BUCKET_NAME)/

#build the tensorflow model into a container
build-train-image:
	docker build $(WORKING_DIR) -t $(TRAIN_PATH) -f $(WORKING_DIR)/Dockerfile.model


# Test training image locally
test-train-image-local:
	docker run -it $(TRAIN_PATH)

#authorize docker to access GCR and push train image
push-train-image: build-train-image
	gcloud auth configure-docker --quiet
	docker push $(TRAIN_PATH)


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


TB_HOST_PORT:=8090

# port-forwarding tensorboard to be accessed locally
access-tensorboard-local:
	kubectl port-forward service/tensorboard-tb $(TB_HOST_PORT):80
	@echo TensorBoard can now be accessed at http://127.0.0.1:$(TB_HOST_PORT)

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

#Building the frontend container
FRONTEND_PATH :=us.gcr.io/$(PROJECT_ID)/kubeflow-frontend
TAG:=v2.6

# Build the frontend container
build-frontend:
	docker build $(WORKING_DIR)/frontend -t $(FRONTEND_PATH):$(TAG)


# Authorize docker and push the frontend image
push-frontend-image: build-frontend
	gcloud auth configure-docker --quiet
	docker push $(FRONTEND_PATH):$(TAG)


#setting frontend parameters
set-frontend-params:
	cd $(KS_NAME) && \
	ks param set web-ui image $(FRONTEND_PATH):$(TAG) && \
	ks param set web-ui type LoadBalancer

# deploy the frontend
deploy-frontend: set-frontend-params push-frontend-image
	cd $(KS_NAME) && \
	ks apply default -c web-ui


# get external IP for the frontend service
frontend-external-ip:
	kubectl get service web-ui

# delete the cluster and other resouces provisioned by kubeflow
clean-all:
	rm -r $(KS_NAME)
	gcloud deployment-manager deployments delete $(DEPLOYMENT_NAME)
	gsutil rm -r gs://$(BUCKET_NAME)
#	gcloud container images delete us.gcr.io/$(PROJECT_ID)/kubeflow-train
#	gcloud container images delete $(FRONTEND_PATH):$(TAG)

list-model-files:
	gsutil ls -r gs://$(BUCKET_NAME)/$(MODEL_PATH)/export
