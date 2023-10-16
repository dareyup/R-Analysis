---
title: "Credit Balance Analysis"
author: "Dareyus Person"
output: "github_document"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```



```{r}
# The purpose of this project is to build a regression model so I can predict credit card balances for new customers (based on data of customers with a current balance)

# Guiding Business Questions:

#1: “What variables effectively contribute to predicting active cardholders’ credit card balances?” 
#2: “What credit card balance might a new active cardholder hold depending on certain variables?” 

```



```{r}
# install packages
#install.packages("tidyverse")
#install.packages("lm.beta")
#install.packages("car")

#load libraries 
library(tidyverse)
library(lm.beta)
library(car)
```

```{r warning=FALSE}
# Set working directory
setwd("~/Documents/Datasets/Stats Datasets")

# Read dataset into RStudio =)
credit <- read.csv("credit.csv")
```


```{r}
# I noticed that the "Income" feature has a '.' to denote a comma for larger number... I'm assuming so lets remove the comma so the number can be represented accurately

credit$Income <- as.numeric(gsub("\\.", "", credit$Income))


# I also noticed a random column "X". I'll remove because it's irrelevant and there's already an index and I'll remove the number of cards

credit <- credit[, -c(1, 5)]


# I'll also remove customers with a 0 balance
credit <- filter(credit, Balance != 0)
```


```{r}
# Convert categorical variables to factors with levels and labels -- easier for the model =)
credit$Student <- factor(credit$Student)
credit$Gender <- factor(credit$Gender)
credit$Married <- factor(credit$Married)
credit$Ethnicity <- factor(credit$Ethnicity)
```


```{r}
# Summary Statistics
summary(credit)
```

```{r}
# Partition the dataset into a training set and a validation set (60-40)
set.seed(28)
sample <- sample(c(TRUE, FALSE), nrow(credit), replace=TRUE, prob=c(0.6,0.4))
traincredit  <- credit[sample, ]
validatecredit <- credit[!sample, ]
```

```{r}
# I want to see correlation between all of the variables, so I'll use a correlation matrix
cor(traincredit[c(1,2,3,4,5,10)])

# It seems 'Limit' and 'Rating' are highly correlated at 99%
```

```{r}
# Turn off scientific notation for all variables
options(scipen=999)

# Create a multiple regression analysis model using the training dataframe with Balance as the outcome variable and all the other variables in the dataset as predictor variables. 

credit_train_MR <- lm(Balance ~ Income + Limit + Rating + Age + Education + Gender + Student + Married + Ethnicity, data = traincredit)

# View train credit multiple regression out
summary(credit_train_MR)

```

```{r}
# Calculate the Variance Inflation Factor (VIF) for all predictor variables
vif(credit_train_MR)

# Limit and Rating have very high VIF, meaning these two are highly correlated with each other, let's remove Limit and re-run the regression and revisit the results
```
```{r}
# Create another multiple regression analysis model without the "Limit" feature

credit_train_MR_nolimit <- lm(Balance ~ Income + Rating + Age + Education + Gender + Student + Married + Ethnicity, data = traincredit)

# View train credit multiple regression out
summary(credit_train_MR_nolimit)

## Observed, the Rating coefficient increased and has more of an influence on Balance
```
```{r}
# Create residual plot and QQ plot of regression analysis "credit_train_MR_nolimit


## Vector of predicted values from credit_MR_nolimit
credit_pred <- predict(credit_train_MR_nolimit)

## Vector of residuals from credit_MR_nolimit
credit_res <- resid(credit_train_MR_nolimit)

pred_res_df <- data.frame(credit_pred, credit_res)

## Create a scatterplot of the residuals versus the predicted values
ggplot(data = pred_res_df, mapping = aes(x = credit_pred, y = credit_res)) +
  geom_point() +
  labs(title = "Plot of residuals vs. predicted values", x = "Predicted values",
       y = "Residuals")

# QQ Plot (Normal Probability Plot)
credit_std_res <- rstandard(credit_train_MR_nolimit)

# QQ plot
qqnorm(credit_std_res, ylab = "Standardized residuals", xlab = "Normal scores")
```
```{r}
# From the model a few steps above, 4 variables show statistical significance with the Balance outcome -- Income, Rating, Age, and Student

# Let's create a new model these variables

credit_sigvar_MR <- lm(Balance ~ Income + Age + Rating + Student, data = traincredit)
summary(credit_sigvar_MR)

## Let's interpret the model coefficients and key summary results

## For example: 
## Student: If you are a student, your balance on average is 402.15 higher than if you weren't.
## Age: For every 1 unit (1 year increase in age), on average, your balance decreases by 1.78
## Rating: For every 1 unit increase, on average, your balance increases by 3.45

## Adjusted R-squared: All the predictor variables account for 81% of the variation in balance
```
```{r}
# Remember that these predictor variables have different scales. We can look at 'standardized regressor coefficients' to compare and see which coefficient makes the strongest contribution to Balance

lm.beta(credit_sigvar_MR)

## Standardized... Rating influences Balance the most
```
```{r}
# Conduct a final multiple regression analysis using the validation dataframe

credit_val_MR <- lm(Balance ~ Income + Age + Rating + Student, data = validatecredit)
summary(credit_val_MR)
```
```{r}
# Using the validation model, let's predict the Balances of a few new card holders with 95% prediction interval



# Let's create 4 new cardholders
cc_pred <- data.frame(Income = c(100000, 43200, 55000, 68000), 
                      Age = c(55, 36, 29, 28), 
                      Rating = c(683, 598, 720, 812), 
                      Student = c("No", "No", "Yes", "Yes")
                      )

# Convert categorical variable to factor with levels and labels
cc_pred$Student <- factor(cc_pred$Student)

# Estimate predicted y values and prediction intervals for 3 new cardholders 
predict(credit_val_MR, cc_pred, interval = "prediction", level = 0.95)
```
```{r}
# Interpreting the results of the new cardholders:

## For new cardholder #1, their predicted balance is 1386.71 and according to the model, we are 95% sure it will be somewhere between 1139.96 and 1633.49
```
