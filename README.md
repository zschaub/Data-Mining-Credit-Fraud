# Data-Mining-Credit-Fraud
### For Data Mining-COMP-7800-01
### By: Brian Zschau

This is a ipython notebook that explores different models and there effectiveness at detecting credit card fraud.

We will be using the dataset found [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This dataset has 284,807 tranactions with 492 fradulent transactions making the dataset unbalanced. Due to confidentiality issues the data only containes time and amount columns and the rest of the columns have been transformed using PCA.

I will be doing two experiments, first I will be comparing four different algorithms to see how they do at credit card fraud detection. These algorithms are Decision Trees, Logistic Regression, Random Forest, and Nerual Networks.Next I will be undersampling the data to make it more balanced and seeing if the model preforms better

## Instructions

To install the necessary packages run
```
pip install -r requirements.txt
```

In the first cell please replace 'your_username' and 'your_key'
```
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_key'
```
with your kaggle username and your kaggle key. This allows you to download the dataset from kaggle. 

This notebook has not been tested on google colab

# Part One:
First we are going to use the raw data. We are going to run the data through 4 different models, 3 classical machine learning algorithms and 1 deep learning algorithm. We will be using Decision Trees, Logistic Regression, Random Forest, and Nerual Networks.

## Results

The models were tested on a dataset with 42722 transactions with predition time being the time to for the model to predict all 42722 transactions.

### Decision Tree

Overall Accuracy: 99.92%  
Prediction Speed: 0.0060 seconds  
False Negatives: 10  

![Decision Tree Confusion Matrix](figs/part1/decisiontree_confusionmatrix.png)

### Logistic Regression
Logistic Regression Overall Accuracy: 99.95%  
Logistic Regression Prediction Speed: 0.0030 seconds  
Logistic Regression False Negatives: 17  

![Logistic Regression Confusion Matrix](figs/part1/logisticregression_confusionmatrix.png)

### Random Forest
Random Forest Overall Accuracy: 99.97%  
Random Forest Prediction Speed: 0.2255 seconds  
Random Forest False Negatives: 9  

![Random Forest Confusion Matrix](figs/part1/randomforest_confusionmatrix.png)

### Neural Networks
Neural Network Overall Accuracy: 99.96%  
Neural Network Prediction Speed: 1.1850 seconds  
Neural Network False Negatives: 8  

![Neural Networks Confusion Matrix](figs/part1/neuralnetwork_confusionmatrix.png)

### Comparing Results

| Model               |  Accuracy  | Prediction Time (seconds)  | False Negatives  |
|---------------------|------------|----------------------------|------------------|
| Decision Tree       | 0.9992     | 0.006                      | 10               |
| Logistic Regression | 0.9994     | 0.002                      | 17               |
| Random Forest       | 0.9997     | 0.225                      | 9                |
| Neural Network      | 0.9996     | 1.185                      | 8                |

![Prediction Time](figs/part1/predictiontime.png)

![False Negatives](figs/part1/falsenegatives.png)