SHELL=/bin/bash

build-environment-and-services:
	@echo "Building Python environment"
	pip install pipenv &&\
	pipenv install
	@echo "Running MLFlow Server on localhost:5000"
	rm -rf mlflow.db mlruns/ &&\
	nohup mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--host localhost:5000 &
		
	@echo "Deploying Prefect Server on localhost:4200"
	nohup prefect server start &
	@echo "The local environment is ready to be used."