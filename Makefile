# you can override variables do
# make ${TARGET} ${VAR}=${VALUE}


# Default project ID, zone and deployment name
PROJECT_ID :=kf-pipelines
ZONE :=us-central1-a
DEPLOYMENT_NAME :=kubeflow

BUCKET_NAME :=$(PROJECT_ID)

WORKING_DIR := kf-pipelines/pipeline_steps/kubeflow/dnntrainer
WEBAPP_DIR := kf-pipelines/pipeline_steps/webapp

# Setting the environment variables for GKE
set-gcloud-project:
	gcloud config set project $(PROJECT_ID)


# Configuring kubectl to connect to the cluster
connect-to-cluster: set-gcloud-project
	gcloud container clusters get-credentials $(DEPLOYMENT_NAME) --zone $(ZONE) --project $(PROJECT_ID)
	kubectl config set-context $(shell kubectl config current-context) --namespace kubeflow
	kubectl get nodes

# Creating bucket for holding the training data and exported models
create-gcs-bucket:
	gsutil mb gs://$(BUCKET_NAME)/


TRAIN_PATH :=us.gcr.io/$(PROJECT_ID)/kubeflow-train_boosted:v0.4

WEBAPP_PATH :=us.gcr.io/$(PROJECT_ID)/ml-pipeline-webapp-launcher:v0.3

#build the tensorflow model into a container
build-train-image:
	mkdir -p $(WORKING_DIR)/build && \
	rsync -arvp $(WORKING_DIR)/src/ $(WORKING_DIR)/build/ && \
	docker build $(WORKING_DIR) -t $(TRAIN_PATH) -f $(WORKING_DIR)/Dockerfile --build-arg TF_TAG=1.13.2

#authorize docker to access GCR and push train image
push-train-image: build-train-image
	gcloud auth configure-docker --quiet
	docker push $(TRAIN_PATH)

#build the tensorflow model into a container
build-webapp-image:
	mkdir -p $(WEBAPP_DIR)/build && \
	rsync -arvp $(WEBAPP_DIR)/webapp-launcher/ $(WEBAPP_DIR)/build/ && \
	docker build $(WEBAPP_DIR) -t $(WEBAPP_PATH) -f $(WEBAPP_DIR)/Dockerfile

#authorize docker to access GCR and push train image
push-webapp-image: build-webapp-image
	gcloud auth configure-docker --quiet
	docker push $(WEBAPP_PATH)


#Building the frontend container
FRONTEND_PATH :=us.gcr.io/$(PROJECT_ID)/kubeflow-frontend-santander
TAG:=v3.4

PREDICT_PATH := kf-pipelines/tfx/pipeline_steps/kubeflow

PREDICT_IMAGE :=us.gcr.io/$(PROJECT_ID)/kubeflow-predict
TAG:=v2.5


# Build the frontend container
build-predict-image:
	docker build $(PREDICT_PATH)/predict -t $(PREDICT_IMAGE):$(TAG)

# Authorize docker and push the frontend image
push-predict-image: build-predict-image
	gcloud auth configure-docker --quiet
	docker push $(PREDICT_IMAGE):$(TAG)


# Build the frontend container
build-frontend-image:
	docker build $(WEBAPP_DIR)/frontend -t $(FRONTEND_PATH):$(TAG)

# Authorize docker and push the frontend image
push-frontend-image: build-frontend-image
	gcloud auth configure-docker --quiet
	docker push $(FRONTEND_PATH):$(TAG)


PRE_PATH := kf-pipelines/tfx/pipeline_steps/dataflow

PRE_IMAGE :=us.gcr.io/$(PROJECT_ID)/kubeflow-preprocess
TAG:=v0.3


# Build the frontend container
build-preprocess-image:
	docker build $(PRE_PATH)/tft -t $(PRE_IMAGE):$(TAG)

# Authorize docker and push the frontend image
push-preprocess-image: build-preprocess-image
	gcloud auth configure-docker --quiet
	docker push $(PRE_IMAGE):$(TAG)

# get external IP for the frontend service
frontend-external-ip:
	kubectl get service santanderapp-webappsvc

clean:
	kubectl delete svc --ignore-not-found kfdemo-service
	kubectl delete deployment --ignore-not-found kfdemo-service-v1
	kubectl delete svc --ignore-not-found santanderapp-webappsvc
	kubectl delete deployment --ignore-not-found santanderapp-webapp
	kubectl delete workflows --all

