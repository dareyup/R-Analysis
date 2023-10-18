---
title: "Loan Default Predictions"
author: "Dareyus Person"
output: github_document
---


```r
#install packages
#install.packages ("tidyverse")
#install.packages("caret")
#install.packages("ROCR")
#install.packages("ROSE")


#load libraries
library(tidyverse)
```

```
## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2 ──
## ✔ ggplot2 3.3.6      ✔ purrr   0.3.5 
## ✔ tibble  3.1.8      ✔ dplyr   1.0.10
## ✔ tidyr   1.2.1      ✔ stringr 1.4.1 
## ✔ readr   2.1.3      ✔ forcats 0.5.2 
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ dplyr::filter() masks stats::filter()
## ✖ dplyr::lag()    masks stats::lag()
```

```r
library(caret)
```

```
## Loading required package: lattice
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:purrr':
## 
##     lift
```

```r
library(ROCR)
library(ROSE)
```

```
## Loaded ROSE 0.0-4
```



```r
#set working directory (adjust this for your own computer)
setwd("~/Documents/Datasets/Stats Datasets/Stats Projects/Loan Default")
```


The purpose of this project is build a logistic model to:

1. Classify if a new customer will default on a loan based a number of factors
2. Go through the steps of predicting and evaluating the model and adjusting parameters to increase evaulation measures



```r
# Read dataset into R
optivadf <- read.csv("loandefault.csv")
#View(optivadf)
```


```r
#Convert categorical variables to factors with levels and labels

optivadf$LoanDefault<-factor(optivadf$LoanDefault,levels = c(0,1), labels = c("No","Yes"))
optivadf$Entrepreneur<-factor(optivadf$Entrepreneur,levels = c(0,1), labels = c("No","Yes"))
optivadf$Unemployed<-factor(optivadf$Unemployed,levels = c(0,1), labels = c("No","Yes"))
optivadf$Married<-factor(optivadf$Married,levels = c(0,1), labels = c("No","Yes"))
optivadf$Divorced<-factor(optivadf$Divorced,levels = c(0,1), labels = c("No","Yes"))
optivadf$HighSchool<-factor(optivadf$HighSchool,levels = c(0,1), labels = c("No","Yes"))
optivadf$College<-factor(optivadf$College,levels = c(0,1), labels = c("No","Yes"))
```



```r
#check for missing data
sum(is.na(optivadf))
```

```
## [1] 0
```

```r
#generate summary statistics for all variables in dataframe
summary(optivadf)
```

```
##    CustomerID    LoanDefault AverageBalance          Age        Entrepreneur
##  Min.   :    1   No :42411   Min.   :     1.3   Min.   :18.00   No :40242   
##  1st Qu.:10799   Yes:  782   1st Qu.:   176.8   1st Qu.:33.00   Yes: 2951   
##  Median :21597               Median :   625.3   Median :39.00               
##  Mean   :21597               Mean   :  1836.7   Mean   :40.76               
##  3rd Qu.:32395               3rd Qu.:  1849.9   3rd Qu.:48.00               
##  Max.   :43193               Max.   :132765.1   Max.   :95.00               
##  Unemployed  Married     Divorced    HighSchool  College    
##  No :41919   No :17247   No :38165   No :20062   No :29931  
##  Yes: 1274   Yes:25946   Yes: 5028   Yes:23131   Yes:13262  
##                                                             
##                                                             
##                                                             
## 
```

```r
## As observed, the target variable "Loan Default" is highly imbalanced and I'll be dealing with that shortly
```

Before addressing the class imbalance, is imperative to split the dataset into train, validate, and test to ensure the "test" dataset results are unbiased (avoid overfitting)

```r
#set seed so the random sample is reproducible
set.seed(42)

#Partition the dataset into a training, validation and test set
Samples <- sample(seq(1,3), size = nrow(optivadf), replace = TRUE, prob = c(0.6,0.2,0.2)) 
train <- optivadf[Samples==1,]
validate <- optivadf[Samples==2,]
test <- optivadf[Samples==3,]

#View descriptive statistics for each dataframe
summary(train)
```

```
##    CustomerID    LoanDefault AverageBalance          Age        Entrepreneur
##  Min.   :    3   No :25450   Min.   :     1.3   Min.   :18.00   No :24080   
##  1st Qu.:10866   Yes:  491   1st Qu.:   176.8   1st Qu.:33.00   Yes: 1861   
##  Median :21626               Median :   625.3   Median :39.00               
##  Mean   :21615               Mean   :  1856.2   Mean   :40.78               
##  3rd Qu.:32413               3rd Qu.:  1882.4   3rd Qu.:48.00               
##  Max.   :43193               Max.   :132765.1   Max.   :94.00               
##  Unemployed  Married     Divorced    HighSchool  College    
##  No :25173   No :10285   No :22953   No :12119   No :17919  
##  Yes:  768   Yes:15656   Yes: 2988   Yes:13822   Yes: 8022  
##                                                             
##                                                             
##                                                             
## 
```

```r
summary(validate)
```

```
##    CustomerID    LoanDefault AverageBalance         Age        Entrepreneur
##  Min.   :    1   No :8581    Min.   :    1.3   Min.   :18.00   No :8164    
##  1st Qu.:10878   Yes: 137    1st Qu.:  171.6   1st Qu.:32.00   Yes: 554    
##  Median :21919               Median :  621.4   Median :39.00               
##  Mean   :21744               Mean   : 1839.9   Mean   :40.69               
##  3rd Qu.:32568               3rd Qu.: 1810.9   3rd Qu.:48.00               
##  Max.   :43190               Max.   :92544.4   Max.   :95.00               
##  Unemployed Married    Divorced   HighSchool College   
##  No :8454   No :3490   No :7715   No :3975   No :6069  
##  Yes: 264   Yes:5228   Yes:1003   Yes:4743   Yes:2649  
##                                                        
##                                                        
##                                                        
## 
```

```r
summary(test)
```

```
##    CustomerID    LoanDefault AverageBalance          Age        Entrepreneur
##  Min.   :    5   No :8380    Min.   :     1.3   Min.   :19.00   No :7998    
##  1st Qu.:10484   Yes: 154    1st Qu.:   182.0   1st Qu.:33.00   Yes: 536    
##  Median :21106               Median :   625.3   Median :39.00               
##  Mean   :21392               Mean   :  1773.9   Mean   :40.79               
##  3rd Qu.:32082               3rd Qu.:  1810.6   3rd Qu.:48.00               
##  Max.   :43191               Max.   :105565.2   Max.   :95.00               
##  Unemployed Married    Divorced   HighSchool College   
##  No :8292   No :3472   No :7497   No :3968   No :5943  
##  Yes: 242   Yes:5062   Yes:1037   Yes:4566   Yes:2591  
##                                                        
##                                                        
##                                                        
## 
```


Next I will adresss the class imbalance by creating 2 datasets: 1 undersampled and 1 oversampled

Undersampling means taking all of the minority class and a matching number of observation in the majority so classes will be balanced or equal

Oversampling means either using all observations from mjaority class and resampling minority class observations to achieve balanced classes



```r
# Create a data frame with only the predictor variables by removing column 2 (Loan Default)

pred.vars <- train[-2]

#Create an undersampled training subset
set.seed(42)
undersample <- downSample(x = pred.vars, y = train$LoanDefault, yname = "LoanDefault")

table(undersample$LoanDefault)
```

```
## 
##  No Yes 
## 491 491
```

```r
# Create an oversampled training subset
set.seed(42)
oversample <- upSample(x = pred.vars, y = train$LoanDefault, yname = "LoanDefault")

table(oversample$LoanDefault)
```

```
## 
##    No   Yes 
## 25450 25450
```

```r
# Create a training subset with ROSE
## The ROSE function deals with class imbalances by generating synthetically balanced samples -- (Boostrapping technique essentially)

rose <- ROSE(LoanDefault ~ ., data = train)$data                         

table(rose$LoanDefault)
```

```
## 
##    No   Yes 
## 12889 13052
```



```r
# Fit logistic regression model on the LoanDefault outcome variable using specified input variables with the undersample dataframe

#Logistic regression is part of the general linear model family, so the R 
#function is glm.
options(scipen=999)
lrUnder <- glm(LoanDefault ~ . - CustomerID, data = undersample, 
               family = binomial(link = "logit"))
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```r
# model summary
summary(lrUnder)
```

```
## 
## Call:
## glm(formula = LoanDefault ~ . - CustomerID, family = binomial(link = "logit"), 
##     data = undersample)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.8321  -1.1988   0.2419   1.0461   3.4433  
## 
## Coefficients:
##                    Estimate  Std. Error z value           Pr(>|z|)    
## (Intercept)      0.43348449  0.35277374   1.229            0.21915    
## AverageBalance  -0.00060824  0.00007972  -7.630 0.0000000000000235 ***
## Age             -0.00423079  0.00778009  -0.544            0.58658    
## EntrepreneurYes  0.38178080  0.25002694   1.527            0.12677    
## UnemployedYes    1.12352119  0.46402502   2.421            0.01547 *  
## MarriedYes       0.10848735  0.16883864   0.643            0.52052    
## DivorcedYes      0.64946422  0.25127894   2.585            0.00975 ** 
## HighSchoolYes    0.09175150  0.19872562   0.462            0.64430    
## CollegeYes      -0.02048428  0.22415136  -0.091            0.92719    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1361.3  on 981  degrees of freedom
## Residual deviance: 1202.8  on 973  degrees of freedom
## AIC: 1220.8
## 
## Number of Fisher Scoring iterations: 7
```

```r
# Fit logistic regression model on the LoanDefault outcome variable using specified input variables with the oversample dataframe
lrOver <- glm(LoanDefault ~ . - CustomerID, data = oversample, 
              family = binomial(link = "logit"))
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```r
# model summary
summary(lrOver)
```

```
## 
## Call:
## glm(formula = LoanDefault ~ . - CustomerID, family = binomial(link = "logit"), 
##     data = oversample)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.7828  -1.1983   0.3346   1.0271   3.5321  
## 
## Coefficients:
##                    Estimate  Std. Error z value             Pr(>|z|)    
## (Intercept)      0.62984034  0.04915732  12.813 < 0.0000000000000002 ***
## AverageBalance  -0.00065505  0.00001166 -56.173 < 0.0000000000000002 ***
## Age             -0.00446456  0.00107383  -4.158        0.00003215941 ***
## EntrepreneurYes  0.50656700  0.03498606  14.479 < 0.0000000000000002 ***
## UnemployedYes    0.44324024  0.05194536   8.533 < 0.0000000000000002 ***
## MarriedYes      -0.04199591  0.02384901  -1.761              0.07825 .  
## DivorcedYes      0.33842467  0.03349792  10.103 < 0.0000000000000002 ***
## HighSchoolYes    0.07915422  0.02740507   2.888              0.00387 ** 
## CollegeYes      -0.17883934  0.03077079  -5.812        0.00000000617 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 70562  on 50899  degrees of freedom
## Residual deviance: 63122  on 50891  degrees of freedom
## AIC: 63140
## 
## Number of Fisher Scoring iterations: 6
```

```r
# Fit logistic regression model on the LoanDefault outcome variable using specified input variables with the rose dataframe

lrrose <- glm(LoanDefault ~ . - CustomerID, data = rose, 
              family = binomial(link = "logit"))
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```r
# model summary
summary(lrrose)
```

```
## 
## Call:
## glm(formula = LoanDefault ~ . - CustomerID, family = binomial(link = "logit"), 
##     data = rose)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.0342  -1.1544   0.7463   1.0686   2.9554  
## 
## Coefficients:
##                    Estimate  Std. Error z value             Pr(>|z|)    
## (Intercept)      0.55661926  0.06587627   8.449 < 0.0000000000000002 ***
## AverageBalance  -0.00043626  0.00001119 -38.987 < 0.0000000000000002 ***
## Age             -0.00689990  0.00140765  -4.902  0.00000094993966979 ***
## EntrepreneurYes  0.49755884  0.04818880  10.325 < 0.0000000000000002 ***
## UnemployedYes    0.45612866  0.07234656   6.305  0.00000000028861675 ***
## MarriedYes      -0.02991339  0.03273346  -0.914              0.36080    
## DivorcedYes      0.36597198  0.04586661   7.979  0.00000000000000147 ***
## HighSchoolYes    0.10761830  0.03818292   2.818              0.00482 ** 
## CollegeYes      -0.12658665  0.04269950  -2.965              0.00303 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 35961  on 25940  degrees of freedom
## Residual deviance: 32841  on 25932  degrees of freedom
## AIC: 32859
## 
## Number of Fisher Scoring iterations: 5
```


Let's take a moment to interpret model coefficients: looking at the oversampled model

-- The coefficients for logistic regressions are interpreted as the average change in log odds in the outcome for a one unit increase in predictor variables which isn't super helpful

-- Key element: Look at the sign (+/-); it indicates which variables increase/decrease the likelihood of the outcome

-- For example (oversample): Being married and having a college degree decrease the likelihood of deafulting on a loan (**hint hint**) :)


```r
# Exponentiate the regression coefficients to better understand the effect they have -- this turns them from log odds to "odds ratio"

## Log odds is the natural log of the odds ratio so the above step is necessary

## Remember odds ration compares the odds of two events (odds in favor / odds against)

exp(coef(lrOver))
```

```
##     (Intercept)  AverageBalance             Age EntrepreneurYes   UnemployedYes 
##       1.8773108       0.9993452       0.9955454       1.6595841       1.5577465 
##      MarriedYes     DivorcedYes   HighSchoolYes      CollegeYes 
##       0.9588737       1.4027361       1.0823712       0.8362402
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


```r
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

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   No  Yes
##        No  4045   21
##        Yes 4536  116
##                                              
##                Accuracy : 0.4773             
##                  95% CI : (0.4668, 0.4878)   
##     No Information Rate : 0.9843             
##     P-Value [Acc > NIR] : 1                  
##                                              
##                   Kappa : 0.0185             
##                                              
##  Mcnemar's Test P-Value : <0.0000000000000002
##                                              
##             Sensitivity : 0.84672            
##             Specificity : 0.47139            
##          Pos Pred Value : 0.02494            
##          Neg Pred Value : 0.99484            
##              Prevalence : 0.01571            
##          Detection Rate : 0.01331            
##    Detection Prevalence : 0.53361            
##       Balanced Accuracy : 0.65905            
##                                              
##        'Positive' Class : Yes                
## 
```


Do the same for the undersample and ROSE training subsets


```r
# Using model built on undersampled training subset:

# Obtain probability for each observation in validation set
lrprobs.us <- predict(lrUnder, newdata = validate, type = "response")

# Obtain predicted class for each observation in validation set using threshold of 0.5
lrclass.us <- as.factor(ifelse(lrprobs.us > 0.5, "Yes","No"))

# output performance metrics using "Yes" as the positive class 
confusionMatrix(lrclass.us, validate$LoanDefault, positive = "Yes" )
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   No  Yes
##        No  3967   19
##        Yes 4614  118
##                                              
##                Accuracy : 0.4686             
##                  95% CI : (0.4581, 0.4791)   
##     No Information Rate : 0.9843             
##     P-Value [Acc > NIR] : 1                  
##                                              
##                   Kappa : 0.0185             
##                                              
##  Mcnemar's Test P-Value : <0.0000000000000002
##                                              
##             Sensitivity : 0.86131            
##             Specificity : 0.46230            
##          Pos Pred Value : 0.02494            
##          Neg Pred Value : 0.99523            
##              Prevalence : 0.01571            
##          Detection Rate : 0.01354            
##    Detection Prevalence : 0.54279            
##       Balanced Accuracy : 0.66181            
##                                              
##        'Positive' Class : Yes                
## 
```

```r
# Using logistic regression model built on ROSE training subset:

# Obtain probability for each observation in validation set
lrprobs.rose <- predict(lrrose, newdata = validate, type = "response")

# Obtain predicted class for each observation in validation set using threshold of 0.5
lrclass.rose <- as.factor(ifelse(lrprobs.rose > 0.5, "Yes","No"))

# output performance metrics using "Yes" as the positive class 
confusionMatrix(lrclass.rose, validate$LoanDefault, positive = "Yes" )
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   No  Yes
##        No  4006   23
##        Yes 4575  114
##                                              
##                Accuracy : 0.4726             
##                  95% CI : (0.4621, 0.4831)   
##     No Information Rate : 0.9843             
##     P-Value [Acc > NIR] : 1                  
##                                              
##                   Kappa : 0.0172             
##                                              
##  Mcnemar's Test P-Value : <0.0000000000000002
##                                              
##             Sensitivity : 0.83212            
##             Specificity : 0.46685            
##          Pos Pred Value : 0.02431            
##          Neg Pred Value : 0.99429            
##              Prevalence : 0.01571            
##          Detection Rate : 0.01308            
##    Detection Prevalence : 0.53785            
##       Balanced Accuracy : 0.64948            
##                                              
##        'Positive' Class : Yes                
## 
```



```r
# Lets run the rose method predictions with a higher threshold and see if anything changes

# Obtain probability for each observation in validation set
lrprobs.rose <- predict(lrrose, newdata = validate, type = "response")

# Obtain predicted class for each observation in validation set using threshold of 0.6
lrclass.rose <- as.factor(ifelse(lrprobs.rose > 0.6, "Yes","No"))

# output performance metrics using "Yes" as the positive class 
confusionMatrix(lrclass.rose, validate$LoanDefault, positive = "Yes" )
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   No  Yes
##        No  7437   96
##        Yes 1144   41
##                                              
##                Accuracy : 0.8578             
##                  95% CI : (0.8503, 0.865)    
##     No Information Rate : 0.9843             
##     P-Value [Acc > NIR] : 1                  
##                                              
##                   Kappa : 0.0348             
##                                              
##  Mcnemar's Test P-Value : <0.0000000000000002
##                                              
##             Sensitivity : 0.299270           
##             Specificity : 0.866682           
##          Pos Pred Value : 0.034599           
##          Neg Pred Value : 0.987256           
##              Prevalence : 0.015715           
##          Detection Rate : 0.004703           
##    Detection Prevalence : 0.135926           
##       Balanced Accuracy : 0.582976           
##                                              
##        'Positive' Class : Yes                
## 
```

```r
## Sensitivity drops significantly as the number of correctly predicted positives go from 114 to 41
```

Confusion Matrices are just one way to evaluate models, let's look at ROC curves below and see how they can be used

ROC curve (Receiver Operator Characteristic) curve is a graph displaying the true positive vs false positive rate -- measuring the accurate ability of the model


```r
# Create a prediction object to use for the ROC Curve
predROC <- prediction(lrprobs.os, validate$LoanDefault)

#create a performance object to use for the ROC Curve
perfROC <- performance(predROC,"tpr", "fpr")

#plot the ROC Curve
plot(perfROC)
abline(a=0, b= 1) # add 45 degree line from origin of graph to see how far our ROC curve is from diagonal
```

![](loandefault_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

```r
# compute AUC -- value for area under the curve 
performance(predROC, measure="auc")@y.values[[1]]
```

```
## [1] 0.7093664
```
AUC (Area under the Curve) = Used as a summary of the ROC curve; the higher the AUC the better. For this model it's 0.7 which is okay, but could be better


```r
# Let's evaluate the accuracy of the model built using the oversampled training set and apply it to the test set

# Obtain probability of defaulting for each observation in test set
lrprobstest <- predict(lrOver, newdata = test, type = "response")

# Obtain predicted class for each observation in test set using threshold of 0.5
lrclasstest <- as.factor(ifelse(lrprobstest > 0.5, "Yes","No"))

#Create a confusion matrix using "Yes" as the positive class 
confusionMatrix(lrclasstest, test$LoanDefault, positive = "Yes" )
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   No  Yes
##        No  3954   32
##        Yes 4426  122
##                                              
##                Accuracy : 0.4776             
##                  95% CI : (0.467, 0.4883)    
##     No Information Rate : 0.982              
##     P-Value [Acc > NIR] : 1                  
##                                              
##                   Kappa : 0.0176             
##                                              
##  Mcnemar's Test P-Value : <0.0000000000000002
##                                              
##             Sensitivity : 0.79221            
##             Specificity : 0.47184            
##          Pos Pred Value : 0.02682            
##          Neg Pred Value : 0.99197            
##              Prevalence : 0.01805            
##          Detection Rate : 0.01430            
##    Detection Prevalence : 0.53293            
##       Balanced Accuracy : 0.63202            
##                                              
##        'Positive' Class : Yes                
## 
```

```r
#Plot ROC Curve for model from oversampled training set using Test set

#create a prediction object to use for the ROC Curve
predROCtest <- prediction(lrprobstest, test$LoanDefault)

#create a performance object to use for the ROC Curve
perfROCtest <- performance(predROCtest,"tpr", "fpr")

#plot the ROC Curve
plot(perfROCtest)
abline(a=0, b= 1)
```

![](loandefault_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

```r
# compute AUC 
performance(predROCtest, measure="auc")@y.values[[1]]
```

```
## [1] 0.6888115
```

Now I'll predict the probability of default for new customers


```r
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

After the above steps, we would set our threshold to determine which customers would default and proceed to make our business decisions

