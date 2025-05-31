
Purpose:
Prediction of "Sachkonto" (General ledger account) - from a few columns based on trainingsdata from SAP table BSAK

Algorithm: 
XGBoost (Catboost is not working yet)

Application: 
a dummy web-application / web-page running an ONNX-model and providing some input fields for inference, to show that this can be used within a web-application

Project Folders:
+ data_preprocessed: preprocessed (i.e. cleaned) data ready to be used by the model. Additonal data like encoder dictionaries etc.
+ data_raw: training data
+ demo_web_page: the web-page that is running the ONNX-model and providing some input fields for inference
+ models: trained models and ONNX models
+ notebooks: containing the notebooks to preprocess the data, to train an XGBoost-model, to train a Catboost-model (not ready yet) and to test the ONNX-model made from the XGBoost-model.

# Conda-Environments:

## XGBoost Environment
Make a Conda Environment from the following yml file (CondaSapBSAKEnv.yml):
### CondaSapBSAKEnv
name: CondaSapBSAKEnv
channels:
  - defaults
  - conda-forge
dependencies:
  - python
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - seaborn
  - matplotlib
  - xgboost=1.4.2
  - pip
  
Thereafter activate the environment and install the boruta and onnxmltool with: pip install boruta onnxmltools==1.7.0 <-- this worked...
BUT there seems to be a little back and forth between the required packages and versions - so expect some work with updates, Chatgpt is your friend...
The requirements.txt contains all the packages and versions that HAVE worked. It has been created by: "pip freeze > requirements.txt" in a command shell with activated CondaSapBSAKEnv environment.


## Catboost Environment:
Make a catboostEnv from the spec-file and in the next step install catboost via pip (Conda is said to be unreliable for this (?))
### catboostSpec.txt
python=3.10
numpy
pandas
scikit-learn
joblib
seaborn

### Installation of catboost AFTER the creation of the conda-environment:
conda create -n catboostEnv --file catboostSpec.txt -y
conda activate catboostEnv
pip install catboost ipykernel
python -m ipykernel install --user --name catboostEnv --display-name "Python (catboostEnv)"
