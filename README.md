# Wafer Fault Detection

## Introduction

In electronics, a wafer (also called a slice or substrate) is a thin slice of semiconductor, such as crystalline silicon (c-Si), used for the fabrication of integrated circuits and, in photovoltaics, to manufacture solar cells. The wafer serves as the substrate (foundation) for microelectronic devices built in and upon the wafer. It undergoes many microfabrication processes, such as doping, ion implantation, etching, thin-film deposition of various materials, and photolithographic patterning. Finally, the individual microcircuits are separated by wafer dicing and packaged as an integrated circuit.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Project Explanation](#project-explanation)
4. [Deployment](#deployment)
5. [Running the Project](#running-the-project)
6. [Conclusion](#conclusion)
7. [Hashtags](#hashtags)

## Project Structure

The repository contains the following folders and files:

- `config/`
  - `model.yaml` - Used to fine-tune the ML model, contains parameters for Grid Search CV.
- `deployment_documentation/`
  - `local_project_setup.md` - Documentation for setting up the project in a local system.
  - `sensor_deployment.md` - Documentation for deployment on AWS server.
- `logs/` - Contains all runtime logs.
- `notebooks/` - Contains Jupyter notebooks and datasets used to build the project.
  - `in1__exploratory_analysis.ipynb`
  - `test.csv`
  - `wafer_23012020_041211.csv`
- `prediction_test_file/` - Contains the test dataset for end-to-end project testing.
  - `test.csv`
- `src/` - Source folder containing various Python scripts.
  - `exception.py` - Used to create custom exceptions.
  - `logger.py` - Used to create logs.
  - `utils.py` - Contains functions that get called by multiple Python files.
  - `components/`
    - `data_ingestion.py` - Used to ingest the data and save it.
    - `data_transformation.py` - Used to transform the data.
    - `model_trainer.py` - Used to train the model.
  - `configuration/`
    - `mongo_db_connection.py` - Used to configure MongoDB.
  - `constant/`
    - `__init__.py` - Contains constant variables.
  - `pipelines/`
    - `prediction_pipeline.py` - Used to automate the prediction of the dataset.
    - `train_pipeline.py` - Used to automate the training of the dataset.
  - `static/`
    - `css/`
      - `style.css`
  - `templates/`
    - `upload_file.html`
- `app.py` - Flask app.
- `Dockerfile` - Used to build the Docker image.
- `README.md`
- `requirements.txt` - Contains the requirements to be installed.
- `setup.py` - Used to set up the project.
- `upload_data.py` - Used to upload the data on MongoDB.

## Prerequisites

For running this project, the following prerequisites must be met:

1. Visual Studio Code installed in the local system.
2. Anaconda installed in the local system.
3. Create a virtual environment of Python 3.8:
   ```bash
   conda create -n wafer_fault_detection python=3.8
   conda activate wafer_fault_detection
   ```
4. Run `setup.py`:
   ```bash
   python setup.py install
   ```
5. Install ipykernel for Jupyter Notebook to work:
   ```bash
   pip install ipykernel
   ```
6. Install requirements from `requirements.txt`:
   ```bash
   pip install -r requirements.txt

For further clarification, read the `local_project_setup.md` file in the `deployment_documentation` folder.

## Project Explanation

In this project, we are trying to predict if the wafer is good or bad, making it a classification problem. To understand the dataset, you can go through `in1__exploratory_analysis.ipynb`. This file contains a detailed explanation of the dataset along with visualizations using Pandas, NumPy, Seaborn, and Matplotlib.

The dataset is trained using multiple models from the Scikit-learn library (`sklearn`). Based on this notebook, the project is built in a modular way. You can also refer to the logs for a better understanding.

## Deployment

This project uses Docker for deployment and employs Continuous Integration, Continuous Delivery, and Continuous Deployment (CI/CD) techniques. The deployment process is as follows:

1. **Docker**: The project is containerized using Docker to ensure consistency across different environments.
2. **CI/CD Pipeline**: GitHub Actions is used to automate the CI/CD pipeline.
    - **Continuous Integration**: Automated tests run on each commit to ensure the integrity of the code.
    - **Continuous Delivery**: The Docker image is built and pushed to AWS Elastic Container Registry (ECR) on successful tests.
    - **Continuous Deployment**: The project is deployed to AWS App Runner from the Docker image in ECR.
3. **AWS App Runner**: The application is hosted on AWS App Runner, which automatically scales based on demand.

For detailed steps, refer to the `sensor_deployment.md` file in the `deployment_documentation` folder.

## Running the Project

To run this project, complete the steps mentioned in the Prerequisites section. Then, run the `app.py` file to start the Flask application. You can access the project at `http://localhost:5000/`.

 - `http://localhost:5000/train` - Trains the dataset.
 - `http://localhost:5000/predict` - Predicts your .csv data file that you upload and downloads the predicted
     data file to your local machine.

## Conclusion

This project demonstrates a modular approach to building a wafer fault detection system using Python, Docker, and AWS. It includes all necessary steps from data ingestion and transformation to model training and deployment.

## Hashtags

#WaferFaultDetection #Python #Docker #CICD #AWS #GitHubActions #MachineLearning #BulkPrediction