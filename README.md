# Census Income with Tree Based Models
## Context 
<font size="+2">
 This project will explore the Census Income Data Set from the UCI Machine Learning Repository. The objective of this project is to classify the income levels of around 50,000 individuals using their demographic information such as gender, working class, education, race, and occupation. We will be using tree based algorithms from Scikit-Learn such as Decision Tree, Random Forest, and Extra Trees to build our model. The purpose of this project is to gain an indepth understanding of Tree Based classification algorithms.  
</font>

## Content 
<font size="+2">
 The data set was extracted by Barry Becker from the 1994 Census database. <br>
 Link to the UCI repository page:https://archive.ics.uci.edu/ml/datasets/census+income
 Classification objective: To determine if a person makes over 50 thousand dollars a year. <br>
 This multivariate data set contains 48,842 instances and 14 attributes. <br>
 
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
 **Image**
<br> From a quick glance at the outputs, we can see that there are no missing values. However, if we take a look closer look we will see that missing values are expressed a question mark ('?')
I used the code below to count the number of missing values in each column.
 
```python
for n in adult.columns:
    count = adult[n][adult[n] == '?'].count()
    print(str(n)+ ' :' + str(count))
```


<br> Summary: 
</font>
 
## Treating Missing Values  
## Exploratory Data Analysis 
## Data Preprocessing
## Building the Model
## Visualizing Model Output
## Conclusion 








