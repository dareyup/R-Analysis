---
title: "Loan Default Predicitions"
author: "Dareyus Person"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r}
#install packages
#install.packages ("tidyverse")
#install.packages("caret")
#install.packages("ROCR")
#install.packages("ROSE")


#load libraries
library(tidyverse)
library(caret)
library(ROCR)
library(ROSE)
```


```{r}
#set working directory (adjust this for your own computer)
setwd("~/Documents/Datasets/Stats Datasets/Stats Projects/Loan Default")
```


The purpose of this project is build a logistic model to:

1. Classify if a new customer will default on a loan based a number of factors
2. Go through the steps of predicting and evaluating the model and adjusting parameters to increase evaulation measures


```{r}
# Read dataset into R
optivadf <- read.csv("loandefault.csv")
#View(optivadf)
```

```{r}
#Convert categorical variables to factors with levels and labels

optivadf$LoanDefault<-factor(optivadf$LoanDefault,levels = c(0,1), labels = c("No","Yes"))
optivadf$Entrepreneur<-factor(optivadf$Entrepreneur,levels = c(0,1), labels = c("No","Yes"))
optivadf$Unemployed<-factor(optivadf$Unemployed,levels = c(0,1), labels = c("No","Yes"))
optivadf$Married<-factor(optivadf$Married,levels = c(0,1), labels = c("No","Yes"))
optivadf$Divorced<-factor(optivadf$Divorced,levels = c(0,1), labels = c("No","Yes"))
optivadf$HighSchool<-factor(optivadf$HighSchool,levels = c(0,1), labels = c("No","Yes"))
optivadf$College<-factor(optivadf$College,levels = c(0,1), labels = c("No","Yes"))
```


```{r}
#check for missing data
sum(is.na(optivadf))

#generate summary statistics for all variables in dataframe
summary(optivadf)

## As observed, the target variable "Loan Default" is highly imbalanced and I'll be dealing with that shortly
```

Before addressing the class imbalance, is imperative to split the dataset into train, validate, and test to ensure the "test" dataset results are unbiased (avoid overfitting)
```{r}
#set seed so the random sample is reproducible
set.seed(42)

#Partition the dataset into a training, validation and test set
Samples <- sample(seq(1,3), size = nrow(optivadf), replace = TRUE, prob = c(0.6,0.2,0.2)) 
train <- optivadf[Samples==1,]
validate <- optivadf[Samples==2,]
test <- optivadf[Samples==3,]

#View descriptive statistics for each dataframe
summary(train)
summary(validate)
summary(test)
```


Next I will adresss the class imbalance by creating 2 datasets: 1 undersampled and 1 oversampled

Undersampling means taking all of the minority class and a matching number of observation in the majority so classes will be balanced or equal

Oversampling means either using all observations from mjaority class and resampling minority class observations to achieve balanced classes


```{r}

# Create a data frame with only the predictor variables by removing column 2 (Loan Default)

pred.vars <- train[-2]

#Create an undersampled training subset
set.seed(42)
undersample <- downSample(x = pred.vars, y = train$LoanDefault, yname = "LoanDefault")

table(undersample$LoanDefault)

# Create an oversampled training subset
set.seed(42)
oversample <- upSample(x = pred.vars, y = train$LoanDefault, yname = "LoanDefault")

table(oversample$LoanDefault)

# Create a training subset with ROSE
## The ROSE function deals with class imbalances by generating synthetically balanced samples -- (Boostrapping technique essentially)

rose <- ROSE(LoanDefault ~ ., data = train)$data                         

table(rose$LoanDefault)
```


```{r}
# Fit logistic regression model on the LoanDefault outcome variable using specified input variables with the undersample dataframe

#Logistic regression is part of the general linear model family, so the R 
#function is glm.
options(scipen=999)
lrUnder <- glm(LoanDefault ~ . - CustomerID, data = undersample, 
               family = binomial(link = "logit"))

# model summary
summary(lrUnder)

# Fit logistic regression model on the LoanDefault outcome variable using specified input variables with the oversample dataframe
lrOver <- glm(LoanDefault ~ . - CustomerID, data = oversample, 
              family = binomial(link = "logit"))

# model summary
summary(lrOver)

# Fit logistic regression model on the LoanDefault outcome variable using specified input variables with the rose dataframe

lrrose <- glm(LoanDefault ~ . - CustomerID, data = rose, 
              family = binomial(link = "logit"))

# model summary
summary(lrrose)
```


Let's take a moment to interpret model coefficients: looking at the oversampled model

-- The coefficients for logistic regressions are interpreted as the average change in log odds in the outcome for a one unit increase in predictor variables which isn't super helpful

-- Key element: Look at the sign (+/-); it indicates which variables increase/decrease the likelihood of the outcome

-- For example (oversample): Being married and having a college degree decrease the likelihood of deafulting on a loan (**hint hint**) :)

```{r}

# Exponentiate the regression coefficients to better understand the effect they have -- this turns them from log odds to "odds ratio"

## Log odds is the natural log of the odds ratio so the above step is necessary

## Remember odds ration compares the odds of two events (odds in favor / odds against)

exp(coef(lrOver))
```

# Interpretation: 
 - Over 1 = Positive effect
 - Under 1 = Negative effect
 - close to 1 = Little to no effect

Example: 
-- If a customer is an Entrepreneur, holding all other variables constant, the odds of them defaulting on a loan increases by 1.66 times on average

In contrasts.....

-- If a customer has a college degree, holding all other variables constant, the odds of them defaulting on a loan decreases by 0.84 times on average

Next, I'll create a confusion matrix to evaluate model accuracy

First I will calculate the probability of each record defaulting on a loan

** Remember in Logistic Regression, the "classification" (0 and 1) part is based on probabilities calculated and then compared to a specified threshold...hence Regression**

```{r}
# First using the model built on the oversampled training subset obtain probability of defaulting for each observation in validation set
lrprobs.os <- predict(lrOver, newdata = validate, type = "response")

# Attach probability scores to validate dataframe
validate <- cbind(validate, Probabilities = lrprobs.os)

# Obtain predicted class for each observation in validation set using threshold of 0.5
lrclass.os <- as.factor(ifelse(lrprobs.os > 0.5, "Yes","No"))

#Attach predicted class to validate dataframe
validate <- cbind(validate, PredClass = lrclass.os)

#Create a confusion matrix using "Yes" as the positive class 
confusionMatrix(lrclass.os, validate$LoanDefault, positive = "Yes" )
```


Do the same for the undersample and ROSE training subsets

```{r}
# Using model built on undersampled training subset:

# Obtain probability for each observation in validation set
lrprobs.us <- predict(lrUnder, newdata = validate, type = "response")

# Obtain predicted class for each observation in validation set using threshold of 0.5
lrclass.us <- as.factor(ifelse(lrprobs.us > 0.5, "Yes","No"))

# output performance metrics using "Yes" as the positive class 
confusionMatrix(lrclass.us, validate$LoanDefault, positive = "Yes" )
```
```{r}
# Using logistic regression model built on ROSE training subset:

# Obtain probability for each observation in validation set
lrprobs.rose <- predict(lrrose, newdata = validate, type = "response")

# Obtain predicted class for each observation in validation set using threshold of 0.5
lrclass.rose <- as.factor(ifelse(lrprobs.rose > 0.5, "Yes","No"))

# output performance metrics using "Yes" as the positive class 
confusionMatrix(lrclass.rose, validate$LoanDefault, positive = "Yes" )
```


```{r}
# Lets run the rose method predictions with a higher threshold and see if anything changes

# Obtain probability for each observation in validation set
lrprobs.rose <- predict(lrrose, newdata = validate, type = "response")

# Obtain predicted class for each observation in validation set using threshold of 0.6
lrclass.rose <- as.factor(ifelse(lrprobs.rose > 0.6, "Yes","No"))

# output performance metrics using "Yes" as the positive class 
confusionMatrix(lrclass.rose, validate$LoanDefault, positive = "Yes" )


## Sensitivity drops significantly as the number of correctly predicted positives go from 114 to 41
```

Confusion Matrices are just one way to evaluate models, let's look at ROC curves below and see how they can be used

ROC curve (Receiver Operator Characteristic) curve is a graph displaying the true positive vs false positive rate -- measuring the accurate ability of the model

```{r}
# Create a prediction object to use for the ROC Curve
predROC <- prediction(lrprobs.os, validate$LoanDefault)

#create a performance object to use for the ROC Curve
perfROC <- performance(predROC,"tpr", "fpr")

#plot the ROC Curve
plot(perfROC)
abline(a=0, b= 1) # add 45 degree line from origin of graph to see how far our ROC curve is from diagonal

# compute AUC -- value for area under the curve 
performance(predROC, measure="auc")@y.values[[1]]
```
AUC (Area under the Curve) = Used as a summary of the ROC curve; the higher the AUC the better. For this model it's 0.7 which is okay, but could be better

```{r}
# Let's evaluate the accuracy of the model built using the oversampled training set and apply it to the test set

# Obtain probability of defaulting for each observation in test set
lrprobstest <- predict(lrOver, newdata = test, type = "response")

# Obtain predicted class for each observation in test set using threshold of 0.5
lrclasstest <- as.factor(ifelse(lrprobstest > 0.5, "Yes","No"))

#Create a confusion matrix using "Yes" as the positive class 
confusionMatrix(lrclasstest, test$LoanDefault, positive = "Yes" )


#Plot ROC Curve for model from oversampled training set using Test set

#create a prediction object to use for the ROC Curve
predROCtest <- prediction(lrprobstest, test$LoanDefault)

#create a performance object to use for the ROC Curve
perfROCtest <- performance(predROCtest,"tpr", "fpr")

#plot the ROC Curve
plot(perfROCtest)
abline(a=0, b= 1)

# compute AUC 
performance(predROCtest, measure="auc")@y.values[[1]]
```

Now I'll predict the probability of default for new customers

```{r}

# Read new dataset into R
new_customers <- read.csv("loandefault_newcust.csv")
#View(new_customers)

#Convert categorical variables to factors with levels and labels
new_customers$Entrepreneur <- factor(new_customers$Entrepreneur, levels = c(0,1), labels = c("No","Yes"))
new_customers$Unemployed <- factor(new_customers$Unemployed, levels = c(0,1), labels = c("No","Yes"))
new_customers$Married <- factor(new_customers$Married, levels = c(0,1), labels = c("No","Yes"))
new_customers$Divorced <- factor(new_customers$Divorced, levels = c(0,1),labels = c("No","Yes"))
new_customers$HighSchool <- factor(new_customers$HighSchool, levels = c(0,1), labels = c("No","Yes"))
new_customers$College <- factor(new_customers$College, levels = c(0,1), labels = c("No","Yes"))

# Make predictions for new data (for which loan default is unknown)
lrprobsnew <- predict(lrOver, newdata = new_customers , type = "response")

# Attach probability scores to new_customers dataframe 
new_customers <- cbind(new_customers, Probabilities = lrprobsnew)
#View(new_customers)
```

After the above steps, we would set our threshold to determine which customers would default and proceed to make our decisions

