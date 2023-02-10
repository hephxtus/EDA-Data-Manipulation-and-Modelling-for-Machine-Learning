# EDA, Data Manipulation, and Modelling for Machine Learning

The goal of this project was to gain experience with the data mining process and exploratory data analysis. The project was completed in a group of 3. 
The project was completed in 3 parts. 

## Part 1: Business and Data Understanding
The first goal was simply to explore the data using CRISP-DM and EDA. I also performed some preliminary data manipulation to
define the machine learning problem and prepare the data for modelling. Output from this consists of:
1. summary statistics about the data including number of instances, number of features and how many were categorical and numerical, respectively
2. a pearson correlation matrix for the numerical features with the target variable 
3. a histogram showing the distribution of the top 5 most correlated features with the target variable 
4. An EDA using clustering to identify how the target variable is distributed across the features

## Part 2: Data Preparation and Machine Learning
The second goal was to prepare the data for modelling and to build a baseline model. I performed the following steps:
1. strip out the metadata
2. remove missing values
3. normalisation, scaling, and standardisation
4. Duplicate removal
5. Train, test split
6. Dimensionality reduction using PCA and t-SNE
7. Building a baseline model using a random forest classifier

## Part 3: Model Evaluation and Improvement
The third goal was to evaluate the baseline model against other, more advanced models. The models used were
* Logistic regressor
* Gaussian Naive Bayes Regressor

## Running the code
To run the code, simply run the main.py file. The code will run the EDA, data preparation, and modelling for the baseline model and the improved model.
The code will also output the results of the cross-validation for the baseline and improved models.

```bash
pip install -r requirements.txt
python main.py
```

## Results
Results can be found in the file `EDA_Data_Manipulation_and_Modelling_for_Machine_Learning_Results.pdf` which is included in the results directory of repository.

