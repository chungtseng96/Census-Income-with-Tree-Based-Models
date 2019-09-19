# Census Income with Tree Based Models
## Context 
<font size="+2">
 This project will explore the Census Income Data Set from the UCI Machine Learning Repository. The objective of this project is to classify the income levels of around 50,000 individuals using their demographic information such as gender, working class, education, race, and occupation. We will be using tree based algorithms from Scikit-Learn such as Decision Tree, Random Forest, and Extra Trees to build our model. The purpose of this project is to gain an indepth understanding of Tree Based classification algorithms.  
</font>

## Content 
<font size="+2">
 The data set was extracted by Barry Becker from the 1994 Census database. <br>
 Link to the UCI repository page:https://archive.ics.uci.edu/ml/datasets/census+income <br>
 Classification objective: To determine if a person makes over 50 thousand dollars a year. <br>
 This multivariate data set contains 48,842 instances and 14 attributes. <br>
 
 
 **Attributes** <br>
 Age: Continuous. <br>
 Workclass: Categorical <br>
 Education: Categorical <br>
 Education-num: Continuous <br>
 Marital-status: Categorical <br>
 Relationship: Categorical <br>
 Capital-gain: Continuous <br>
 Capital-loss: Continuous <br>
 Hours-per-week: Continuous <br>
 Native-country: Categorical <br>
 
</font>

## Preliminary Data Analysis 
<font size="+2">
 Before we jump into any type of analysis or modeling we need to understand the quality, structure, and the range of our data. The purposes of preliminary data analysis are to modify the data to prepare it for further analysis. To perform our preliminary data analysis, we need to first install relavant dependencies such as numpy, pandas, and matplotlib. After we read our data set into a pandas dataframe, we can take a quick high-level overview of our data using the info and describe method. Some common data quality problems we can extract from the info method output are: missing values, data types, attributes, and memory usage. From the describe method we can extract basic statistical information from our data set. See below for info and describe method outputs. 
<img src="/images/info.PNG" width="500" height="530">
<br> From a quick glance at the outputs, we can see that there are no missing values. However, if we take a look closer look we will see that missing values are expressed a question mark ('?')
I used the code below to count the number of missing values in each column.
 
```python
for n in adult.columns:
    count = adult[n][adult[n] == '?'].count()
    print(str(n)+ ' :' + str(count))
```
**Output** <br>
Workclass: 2799 <br>
Occupation: 2809 <br>
Native-Country: 857 <br>

Summary: Multiple columns have missing values, drop fnlwgt column, and binarize gender. <br> 

</font>
 
## Treating Missing Values  
<font size="+2">
Now that we know there are missing values, we need to treat them appropriately. There are several ways to treat missing values which include: deletion, impute with mean value, label encode as another level of categorical variable, or impute with predictive model. The best method to maintain the integrity of the data set is to impute with a predictive model. To do so we need to create two functions to help us execute this process in an efficient manner.  <br>
<br>

**Function 1**: One Hot Encoder <br>
Since marchine learning algorithms can't work with categorical data directly, we have to convert them into integers. 
The function below takes two parameters: a dataframe and column(s) of that dataframe. 
The function converts the selected columns into dummy/indicator variables using the pandas.get_dummies function and then merge it with the rest of the columns. 

```python
def onehotencoder(df, df_cols):
    
    df_1 = adult_data = df.drop(columns = df_cols, axis = 1)
    df_2 = pd.get_dummies(df[df_cols])
    
    return (pd.concat([df_1, df_2], axis=1, join='inner'))
```

**Function 2**: Logistic Regression imputation. <br>
Objective: To use logistic regression to impute missing data in the workclass, occupation, and native-country columns. <br>
The function will essentially use instances with no missing values as training data with workclass, occupation, and natuve-country as dependent variables and all other columns as independent variables. The testing data will be instances with missing values. Then the function will fit a logistic regression with the trianing data and predict with the testing data. The result is a dataset with missing values filled with logistic regression predicted values.

```python
def logimpute(col):
    test_data = adult[(adult[col].values == '?')].copy()
    test_label = test_data[col]

    train_data = adult[(adult[col].values != '?')].copy()
    train_label = train_data[col]

    test_data.drop(columns = [str(col)], inplace = True)
    train_data.drop(columns = [str(col)], inplace = True)

    train_data = onehotencoder(train_data, train_data.select_dtypes('object').columns)
    test_data = onehotencoder(test_data, test_data.select_dtypes('object').columns)

    missing_cols = set(train_data.columns) - set(test_data.columns)
    for c in missing_cols:
        test_data[c] = 0
    test_data = test_data[train_data.columns]

    log_reg = LogisticRegression()
    log_reg.fit(train_data, train_label)
    log_reg_pred = log_reg.predict(test_data)

    adult.loc[(adult[col].values == '?'),str(col)] = log_reg_pred
```

Once we run both functions on our data set we can see that we don't have anymore missing values. 
</font>
 
## Exploratory Data Analysis 
<font size="+2">
In this section we will use graphical and non-graphical exploratory data analysis to summarize the data set's main characteristics such as distribution, correlation, range, and behavior. <br>

**Non-Graphical Univariant EDA** <br>
Tabulate Frequency of Occupation 

```python 
freq_occ = pd.DataFrame(adult.occupation.value_counts())
freq_occ = freq_occ.rename(columns = {'occupation':'Count'})
freq_occ['Proportion'] = freq_occ['Count']/freq_occ.Count.sum()
freq_occ['Percent'] = freq_occ.Proportion*100
freq_occ
```

<img src="/images/occu.PNG" width="240" height="300">
<br>
Tabulate Frequency of Race
<img src="/images/race count.PNG" width="250" height="150">
<br>
Tabulate Frequency of Workclass
<img src="/images/workclass.PNG" width="250" height="200">
<br>
Tabulate Frequency of Education
<img src="/images/edu count.PNG" width="200" height="300">
<br>

**Graphical Multivariant EDA** <br>
Correlation Matrix <br>
<img src="/images/corr.png" width="400" height="400">
<br>
Income Level Pie Chart <br>
<img src="/images/income.png" width="300" height="300">
<br>
Class vs. Occupation Count Plot <br>
<img src="/images/class vs occ.png" width="700" height="400">
<br>
Education Count Plot <br>
<img src="/images/edu.png" width="700" height="400">
<br>
Class vs. Race Count Plot <br>
<img src="/images/race.png" width="700" height="400">
<br>
Class vs. Gender Count Plot <br>
<img src="/images/sex.png" width="500" height="400">


</font>

## Data Preprocessing
<font size="+2">
Before we move onto building the actual model, we need to preprocess our dataset into a format in which the Scikit-Learn algorithms can process. This includes binarizing the target variable, converting categorical features into dummy variables, and splitting the data set into training and testing data. 
We will use LabelBinarizer, train_test_split from Scikit-Learn preprocessing and pandas get_dummy function to perform these tasks.
 
```python
#Seperating target variable from the rest of the features
adult_data = adult.drop(columns = ['class'])
adult_label = adult['class']

#Binarize target label
from sklearn.preprocessing import LabelBinarizer 
Binarizer = LabelBinarizer()
adult_label = Binarizer.fit_transform(adult_label)

#Converting all categorical features into dummy variables
adult_cat = pd.get_dummies(adult_data.select_dtypes('object'))
adult_noncat = adult_data.select_dtypes(exclude = 'object')
adult_data = pd.concat([adult_noncat, adult_cat], axis = 1, join = 'inner')

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(adult_data, adult_label, test_size = 0.3, random_state = 21)
```
</font>

## Building the Model
Now that we have everything set up, we can start building the model. 
The first step to building our model is creating model evaluation function so we can measure and track the performance of our model.
The function below will take 2 parameters as inputs: actual testing values and predicted values.
Our model_eval function will create a confusion matrix and store the results in 4 variables: true positive (tp), true negative (tn), false positive (fp), and false negative (fn).
The function will then use the variables to calculate accuracy, precision, recall, f_measure, sensitivity, specificity, and error_rate. 

```python
#Model Evaluation Function 
from sklearn.metrics import confusion_matrix
def model_eval(true, pred):
    #Confusion Matrix 
    true = true.reshape(len(true),)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    #Output 
    output = {}
    output['accuracy'] = ((tp+tn))/(tp+fn+fp+tn)
    output['precision'] = (tp)/(tp+fp)
    output['recall'] = (tp)/(tp+fn)
    output['f_measure'] = (2*output['recall']*output['precision'])/(output['recall']+output['precision'])
    output['sensitivity'] = tp / (tp + fn)
    output['specificity'] = tn / (tn + fp)
    output['error_rate'] = 1 - output['accuracy']
    return output
```

**Building Decision Trees** <br>
To build our decision trees we will use the Decision Tree Classifier from Sci-kit Learn. 
We will start by feeding the data into the model with default hyperparamters then we will tune them accordingly. 
<br>
Initial Tree

```python 
from sklearn.tree import DecisionTreeClassifier
tree_clf_1 = DecisionTreeClassifier(max_depth = 2)
tree_clf_1.fit(X_train, y_train)
tree_clf_1_pred = tree_clf_1.predict(X_test)
tree1_res = model_eval(y_test, tree_clf_1_pred)
```

Model Performance: <br>
accuracy:	0.826179 <br>
precision:	0.732828 <br>
recall:	0.438244 <br>
f_measure:	0.548484 <br>
sensitivity:	0.438244 <br>
specificity:	0.949294 <br>
error_rate:	0.173821 <br>
 <img src="/images/dtree_pipe.png" width="500" height="400">
 






## Visualizing Model Output
## Conclusion 



















