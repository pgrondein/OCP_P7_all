# Scoring Tool for Financial Company

![banniere](https://github.com/pgrondein/scoring_model_financial_company/assets/113172845/1973f8de-2b96-4b86-9823-833ce17a70e3)

## Problem Definition

A financial company offers consumer credit to people with little or no loan history. She wants to develop a **Scoring Credit** **Tool** to obtain the customer's probability of repayment. She therefore wishes to develop a **classification algorithm** based on various data sources (behavioral data, data from other financial institutions, etc.).

In addition, it is important for the company to provide **transparency** with regard to credit granting decisions. Consequently, an interactive dashboard is developed so that customer relationship managers can both explain credit granting decisions as transparently as possible, but also allow their customers to have access to their personal information.

For this project, the different stages were :

- Development of a scoring model that will automatically predict the probability of a customer's bankruptcy.
- Construction of an interactive dashboard for customer relationship managers to interpret the predictions made by the model, and to improve the customer knowledge of customer relationship managers.
- Production of the prediction scoring model using an API, as well as the interactive dashboard which calls the API for predictions.

## Data Collection

The data is divided into 10 datasets, with information on credit requests (id, history, etc.) as well as on the applicants (age, profession, etc.), and between training sets with the TARGET variable, which gives the final decision (0: reimbursement, 1: non-reimbursement), and test set, without the TARGET variable.

The data is available [here](https://www.kaggle.com/c/home-credit-default-risk/data).

## Exploratory Data Analysis

### Single feature analysis
#### TARGET

<img src="https://github.com/pgrondein/scoring_model_financial_company/assets/113172845/7876f52b-9389-4813-8ffa-66b88ff98bb6" height="400">

By observing the data from the training game, we see that it is unbalanced: we observe a large majority of class 0 (92%) compared to class 1 (8%). We are therefore faced with a Class Imbalance problem.

There are several methods to manage a Class Imbalance problem in training data (SMOTE, Under-sampling, Over-sampling, etc.). The Class Weight method is selected, that is to say we adjust the cost function of the model so that the misclassification of a minority class observation is more heavily penalized than the misclassification of an observation of the class majority.

## Data Pre-processing

### Split Train/Validation

The dataset with TARGET is divided into train set/validation set, with a proportion 80%/20%.

### Data Cleaning

After separating the dataset into train data and validation data for the model, they are separately cleaned (treatment of missing values, outliers, etc.). Then the quantitative variables are normalized, the categorical variables binarized.

## Scoring model

### Tested algos

L’entraînement des algorithmes se fait sur le jeu train. La validation des performances se fait sur l’échantillon de validation. 

The models tested are:

- **Dummy Classifier**: to have a reference
- **Logistic Regression**
- **Random Forest Classifier**

Search for optimal hyperparameters for the Logistic Regression and Random Forest algorithms is done using GridSearchCV.

Training of the algorithms is done on the train game. Performance validation is done on the validation sample.

### Performance comparison

In order to compare the algorithms, the AUC is calculated. This is the indicator of the discriminatory capacity of the model, between class 1 and class 0. We add the calculation time, always an interesting element.

| Algorithm | AUC | Computation time |
| :---: | :---: | :---: |
| `Dummy Classifier`  | 0.50 | 8 s |
| `Logistic Regression`  | 0.76 | 17 s |
| `Random Forest Classifier`  | 0.75 | 3 min 10 s |

The Logistic Regression algorithm is selected.

<img src="https://github.com/pgrondein/scoring_model_financial_company/assets/113172845/5abd7bcc-254e-4623-807f-010510ecea8b" height="400">

### Cost Function

Financial losses due to bad prediction are undesirable. However, it is impossible to completely remove them. We can still optimize the model as well as the Business Cost function in order to limit them.

Predictions can be categorized as follows:

- **False Positives**: cases where the prediction is positive but the actual value is negative. Here we have a loss of opportunity if the credit is wrongly refused to the customer even though he would have been able to repay the credit.
- **False Negatives**: cases where the prediction is negative, but the actual value is positive. Here we have a real loss if the credit is accepted since this can turn into a payment default.
- **True Positives**: case of acceptance where the credit will be reimbursed
- **True Negatives**: cases of refusal where the credit would not have been repaid

In our case, we understand that it is necessary to minimize False Negatives, and False Positives, particularly FNs. We then consider two criteria: Recall and Precision.

<img src="https://github.com/pgrondein/scoring_model_all/assets/113172845/63b6bf33-dbf5-474f-8125-e5e6cf928603" height="100">

Both values must be maximized.

To do this, we use the Fbeta Score function, with beta = 1 so Recall and Precision have the same weight :

<img src="https://github.com/pgrondein/scoring_model_all/assets/113172845/8eddc23e-be0b-41d9-a50f-45bf55e1977c" height="100">


This function is selected as a business cost function, and must be maximized.

In order to determine the threshold for the logistic regression algorithm, we plot the F1 score as a function of the decision threshold of the classification algorithm.

<img src="https://github.com/pgrondein/scoring_model_financial_company/assets/113172845/03b98d28-439f-4ad9-9343-bb67a327f69a" height="400">

We note that the score is maximized for a threshold of 0.67, therefore 67% probability of non-reimbursement.

We define the decision thresholds for classes 1 (non-reimbursement) and 0 (reimbursement):

- 1: 0.69
- 0: 0.31

### Interpretability

For the sake of transparency of the final decision to accept or reject a loan application, it is necessary to be able to explain this decision. To do this, we detail the score obtained and the impact of the different variables. We interpret the results.

Interpretability therefore refers to the global or local evaluation of the decision-making process. It aims to represent the relative importance of each variable in the decision-making process.

We therefore plot in descending order the coefficients associated with each variable, knowing that they can be positive or negative.

#### Global

<img src="https://github.com/pgrondein/scoring_model_financial_company/assets/113172845/0fdd8f33-caa5-4d53-9b19-2c0ff1f760df" height="400">

#### Local

<img src="https://github.com/pgrondein/scoring_model_financial_company/assets/113172845/321da8d9-6eb4-416c-a087-c90e18529277" height="400">

### API

<img src="https://github.com/pgrondein/scoring_model_all/assets/113172845/160d4c85-60ff-4f05-84db-273a2368f06a" height="400">

To set up the API, we start by loading the pre-trained model on the training data.

We define a home page, as well as a prediction page. It receives the chosen test data as input, and returns the probability of non-reimbursement, as well as the status of the decision.

### Dashbord

<img src="https://github.com/pgrondein/scoring_model_all/assets/113172845/88d30b7c-d76e-402f-8834-868369e162a3" height="400">

The following features are implemented:

- Visualization of the score and interpretation of this score for each client in a way that is intelligible for a person who is not an expert in data science.
- Visualization of descriptive information relating to a customer (via a filter system).
- Possibility of comparing descriptive information relating to a customer to all customers or to a group of similar customers.

<img src="https://github.com/pgrondein/scoring_model_all/assets/113172845/99cd26df-9fd8-40ff-8736-e3d128103218" height="200">

The dashboard calls the API:

- asks the user for the file number
- sending a request to the API url with the test data selected from the file number
- reception of the probability value for class 1
- display of the result


![streamlit](https://github.com/pgrondein/scoring_model_all/assets/113172845/f4133ac0-d2f6-45df-8f56-2e1e678ebcc3)





