# Project Overview: Inventory Monitoring at Distribution Centers

Distribution centers often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects. In this project, you will have to build a model that can count the number of objects in each bin. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items.
To build this project you will use AWS SageMaker and good machine learning engineering practices to fetch data from a database, preprocess it, and then train a machine learning model. This project will serve as a demonstration of end-to-end machine learning engineering skills that you have learned as a part of this nanodegree.


## Project Set Up and Installation

### Setup 
- If using Sagemaker Notebook Instance, recommendation is to use `conda_pytorch_p39` kernel type.
- When running notebook on premises, code has been verified with python version of `3.8.10.`

`sagemaker.ipynb` notebook is the main file to be used to reproduce all the steps of ML cycle used in this project including libraries installation and import.

## Main Files

- `sagemaker.ipynb` <- notebook describing each stage of ML capstone project
- `train_model.py` <- script to run model training via AWS Training job
- `hpo.py` <- script to run HPO utilizing AWS Tuning job
- `code` directory which includes `inference.py` script (required for model deployment to an endpoint).
- `file_list.json` <- json file which includes information about subset data to be used for showcase purposes.
- `report.pdf` <- project report which includes main considerations and outcomes of experimentation process.


## Dataset

### Overview
To complete this project we will be using a subset of the <a href="https://registry.opendata.aws/amazon-bin-imagery/" target="_blank">Amazon Bin Image Dataset</a>. The dataset contains 500,000 images of bins containing one or more objects. For each image there is a metadata file containing information about the image like the number of objects, it's dimension and the type of object. For this task, we will try to classify the number of objects in each bin.

### Access

To build this project I am using the [Amazon Bin Images Dataset](https://registry.opendata.aws/amazon-bin-imagery/)
- Download the dataset: Since this is a large dataset, we have been provided with some code to download a small subset of that data. I will be using this subset to prevent any excess SageMaker credit usage.
- Preprocess and clean the files.
- Upload them to an S3 bucket so that SageMaker can use them for training

## Model Training

To create an image classifier, I will fine-tune a pre-trained convolutional neural network (ResNet50) on dataset subset using SageMaker Training jobs. Hyperparameter tuning is performed as a part of experimentation process to find an optimum values to use for learning rate, batch size, momentum, and image size. 


## Machine Learning Pipeline

Below is a brief description of the steps:

1. Download proposed data and perform exploratory data analysis. Define data preprocessing steps.
2. Split dataset into training and test datasets and upload to an S3 bucket.
3. Write model training script.
4. Train model utilizing AWS SageMaker training jobs.
5. Run Hyperparameter tuning to define a set of parameters of the best performing model.
6. Traing and deploy trained model to an endpoint. 


