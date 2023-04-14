# title: "Bank Marketing: Predicting results of telemarketing campaigns"
# author: "Ngô Thành Công & Phan Trọng Hiếu"
# date: "3/28/2023"


##########################################################
# Create training set, test set (from bank set)
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColor.Brewer", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(RColorBrewer)
library(randomForest)
library(e1071)
library(tinytex)
# install tex packages
if (tinytex::is_tinytex() == FALSE){
  tinytex::install_tinytex()
}


# Bank Marketing dataset:
# https://archive.ics.uci.edu/ml/datasets/bank+marketing
# https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip", dl)

bank <- read_csv2(unzip(dl, "bank-full.csv"))
bank.small <- read_csv2(unzip(dl, "bank.csv")) # small dataset: 10% of "bank-full.csv" data
# bank <- read_csv2("bank-full.csv")
# bank.small <- read_csv2("bank.csv")
# convert to data frame
bank <- as.data.frame(bank)
bank.small <- as.data.frame(bank.small)

# convert to data frame
bank <- as.data.frame(bank)
# convert to data frame
bank.small <- as.data.frame(bank.small)
# inspect the data
head(bank)
str(bank)
summary(bank)
# there are several character columns which are actually factors (categorical variables)

# convert character columns to factors
colnames(bank)
cols_character <- c("job", "marital", "education", "default", "housing", "loan", "contact", 
                    "month", "poutcome",  "y")
bank[cols_character] <- lapply(bank[cols_character], as.factor)
summary(bank)
#bank.small[cols_character] <- lapply(bank.small[cols_character], as.factor)
#summary(bank.small)

# train-test split
# Validation set will be 10% of the bank dataset, because the dataset is large enough for proper training and we want to use this
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = bank$y, times = 1, p = 0.1, list = FALSE)
train.bank <- bank[-test_index,]
test.bank <- bank[test_index,]

# clean up - remove unused objects
rm(dl, test_index, bank)
# save training and test set 
saveRDS(train.bank, "train.bank.Rda")
saveRDS(test.bank, "test.bank.Rda")

# clean up - not mandatory, because the data is not very large
rm(train.bank, test.bank)
gc()

# load saved data, if not in workspace already
train.bank <- readRDS("train.bank.Rda")
test.bank <- readRDS("test.bank.Rda")


##########################################################
# Explore the data (EDA)
##########################################################
# Numerical data
##########################################################
# age
train.bank %>% ggplot(aes(x = age)) + geom_histogram(bins = 20)
#train.bank %>% ggplot(aes(y = age)) + geom_boxplot()
train.bank %>% ggplot(aes(x = age)) + geom_boxplot()
mean(train.bank$age)
# mean and median age are close, value is around 40 years
# right skewed

# balance
train.bank %>% ggplot(aes(x = balance)) + geom_histogram(bins = 30)
train.bank %>% ggplot(aes(y = balance)) + geom_boxplot()
mean(train.bank$balance) # 1365.923
# strong right skew, with many outliers
sum(train.bank$balance <= 0) # 6566

# duration
train.bank %>% ggplot(aes(x = duration)) + geom_histogram(bins = 50)
train.bank %>% ggplot(aes(y = duration)) + geom_boxplot()
mean(train.bank$duration)

# right skew, many outliers

# campaign
train.bank %>% ggplot(aes(x = campaign)) + geom_histogram(bins = 30)
train.bank %>% ggplot(aes(y = campaign)) + geom_boxplot()
# right skew, many outliers

# pdays
train.bank %>% ggplot(aes(x = pdays)) + geom_histogram(bins = 30)
train.bank %>% ggplot(aes(y = pdays)) + geom_boxplot()
sum(train.bank$pdays == -1) # 33252
# mostly -1 values, right skew with outliers
# check without -1 entries
train.bank %>% filter(pdays > -1) %>% ggplot(aes(x = pdays)) + geom_histogram(bins = 30)
train.bank %>% filter(pdays > -1) %>% ggplot(aes(y = pdays) ) + geom_boxplot()

# previous
train.bank %>% ggplot(aes(x = previous)) + geom_histogram(bins = 50)
train.bank %>% ggplot(aes(y = previous)) + geom_boxplot()
sum(train.bank$previous == 0) # 33252 -> same number as for pdays == -1 -> new potential customers?
# mostly 0 values, right skew with outliers
# check without 0 entries
train.bank %>% filter(previous > 0) %>% ggplot(aes(x = previous)) + geom_histogram(bins = 30)
train.bank %>% filter(previous > 0) %>% ggplot(aes(y = previous) ) + geom_boxplot()


##########################################################
# Categorical data
##########################################################
# job
train.bank %>% ggplot(aes(x = job)) + geom_bar(aes(fill = job)) + 
  theme(axis.text.x = element_text(angle=80, hjust=1))
# most persons work in blue-collar jobs, management, technician, admin and services

# marital
train.bank %>% ggplot(aes(x = marital)) + geom_bar(aes(fill = marital))
sum(train.bank$marital== "married")
# most persons are married

# education
train.bank %>% ggplot(aes(x = education)) + geom_bar(aes(fill = education))
# most persons completed secondary education

# default
train.bank %>% ggplot(aes(x = default)) + geom_bar(aes(fill = default))
# most persons did not default

# housing
train.bank %>% ggplot(aes(x = housing)) + geom_bar(aes(fill = housing))
# most persons have housing, but the proportions are more similar than in other factors (55% yes vs 45% no)
table(train.bank$housing)
mean(train.bank$housing == "yes")

# loan
train.bank %>% ggplot(aes(x = loan)) + geom_bar(aes(fill = loan))
# most persons do not have a loan

# contact
train.bank %>% ggplot(aes(x = contact)) + geom_bar(aes(fill = contact))
# most persons were contacted using cellular (mobile phones), but the number of unknown values is relatively large

# month
train.bank %>% ggplot(aes(x = month)) + geom_bar(aes(fill = month))
# most persons were contacted during may, july, june and august, the fewest during december and march

# poutcome
train.bank %>% ggplot(aes(x = poutcome)) + geom_bar(aes(fill = poutcome))
# mostly failure for known data, but most entries are unknown

# day as factor
train.bank %>% ggplot(aes(x = as.factor(day))) + geom_bar(aes(fill = as.factor(day)))
# most persons were contacted during 18, 19 and 20, the fewest during 1, 10 and 24


# y (dependent variable)
train.bank %>% ggplot(aes(x = y)) + geom_bar(aes(fill = y))
# more "no" than "yes" outcomes
# -> data not balanced - might use under-/upsampling or weights?


##########################################################
# Bivariate plots
##########################################################
# y and numeric variables
train.bank %>% group_by(y)  %>% ggplot(aes(y = age, x = y))+ geom_boxplot(aes(fill = y))
# median age for yes is slightly lower, but data with wider spread
train.bank %>% group_by(y)  %>% ggplot(aes(y = balance, x = y))+ geom_boxplot(aes(fill = y))
train.bank %>% group_by(y)  %>% ggplot(aes(y = duration, x = y))+ geom_boxplot(aes(fill = y)) # important feature
# median duration is higher for yes than for no, even when considering wider spread of the data
train.bank %>% group_by(y)  %>% ggplot(aes(y = campaign, x = y))+ geom_boxplot(aes(fill = y))
train.bank %>% group_by(y)  %>% ggplot(aes(y = pdays, x = y))+ geom_boxplot(aes(fill = y)) # why almost no spread for "no"?
# wider spread for yes values
train.bank %>% group_by(y)  %>% ggplot(aes(y = previous, x = y))+ geom_boxplot(aes(fill = y))

# y and categorical variables
# marital status
# grouped bar chart
train.bank %>% ggplot(aes(x = y, fill = marital))+ geom_bar(position = "dodge")
# segmented bar chart (adds up to 100%)
train.bank %>% ggplot(aes(x = y, fill = marital))+ geom_bar(position = "fill")
# married people are the majority in both groups
# single persons said yes more often

# education
# grouped bar chart
train.bank %>% ggplot(aes(x = y, fill = education))+ geom_bar(position = "dodge")
# segmented bar chart (adds up to 100%)
train.bank %>% ggplot(aes(x = y, fill = education))+ geom_bar(position = "fill")
# secondary educated are the majority in both groups
# tertiary educated persons said yes more often, primary educated said no more often

# default status
# grouped bar chart
train.bank %>% ggplot(aes(x = y, fill = default))+ geom_bar(position = "dodge")
# segmented bar chart (adds up to 100%)
train.bank %>% ggplot(aes(x = y, fill = default))+ geom_bar(position = "fill")
# majority of yes come from persons with no default, but situation is similar for no
table(train.bank$y, train.bank$default)

# housing
# grouped bar chart
train.bank %>% ggplot(aes(x = y, fill = housing))+ geom_bar(position = "dodge")
# segmented bar chart (adds up to 100%)
train.bank %>% ggplot(aes(x = y, fill = housing))+ geom_bar(position = "fill")
# most yes answers come from persons with no housing, most no asnwers come from persons with housing
# (-> which aligns with the purpose of finding new credit customers)

# loan
# grouped bar chart
train.bank %>% ggplot(aes(x = y, fill = loan))+ geom_bar(position = "dodge")
# segmented bar chart (adds up to 100%)
train.bank %>% ggplot(aes(x = y, fill = loan))+ geom_bar(position = "fill")
# most yes answers come from persons who don't have a loan
# (-> which aligns with the purpose of finding new credit customers)

# contact
# grouped bar chart
train.bank %>% ggplot(aes(x = y, fill = contact))+ geom_bar(position = "dodge")
# segmented bar chart (adds up to 100%)
train.bank %>% ggplot(aes(x = y, fill = contact))+ geom_bar(position = "fill")
# most yes answers come fram persons contacted via cellular(cellphone)

# poutcome
# grouped bar chart
train.bank %>% ggplot(aes(x = y, fill = poutcome))+ geom_bar(position = "dodge")
# segmented bar chart (adds up to 100%)
train.bank %>% ggplot(aes(x = y, fill = poutcome))+ geom_bar(position = "fill")
# many yes answers come from persons who had succesful previous outcomes
# (higher trust factor?)

# month
# grouped bar chart
train.bank %>% ggplot(aes(x = y, fill = month))+ geom_bar(position = "dodge")
# segmented bar chart (adds up to 100%)
# y outcome by month (proportion)
train.bank %>% ggplot(aes(x = month, fill = y))+ geom_bar(position = "fill")
# highest proportion of yes in march, dec, oct, sep -> when there are fewer calls
# during months with the highest number of calls, the proportions is relatively constant

# day as factor
# grouped bar chart
train.bank %>% ggplot(aes(x = y, fill = as.factor(day)))+ geom_bar(position = "dodge")
# segmented bar chart (adds up to 100%)
# y outcome by day (proportion)
train.bank %>% ggplot(aes(x = as.factor(day), fill = y))+ geom_bar(position = "fill")
# highest proportion of yes on 1, 10 and 30 of the month
# 1 and 10 are days with fewer calls, 30 is a day with many calls


##########################################################
# Check correlations (for numeric variables)
##########################################################
# get numeric columns
cols_number <- colnames(train.bank) [!(colnames(train.bank) %in% cols_character)]
# cols_number
# get correlation between the variables and the outcome y
cor(train.bank[, cols_number], as.numeric(train.bank$y))
# Stronger correlation for duration, pdays, and previous


##########################################################
# Models
##########################################################
# Preprocess the data: Data imbalance and undersampling
##########################################################
# proportion of "yes" and "no" outcomes (y variable)
mean(train.bank$y == "yes") # ~12%
# y outcomes are not evenly distributed -> imbalanced data
sum(train.bank$y == "yes") # 4760 / 40689

# generate reduced, more balanced dataset
# split data by outcome
train.yes <- train.bank %>% filter(y == "yes")
head(train.yes)
train.no <- train.bank %>% filter(y == "no")
head(train.no)
# undersample the "no" data to match the amount of "yes" data
set.seed(1, sample.kind="Rounding")
size_yes <- dim(train.yes)[1] # get size of the "yes" data
size_yes
# (alternatively - not done here: round size up to the nearest thousand)
rows.no <- sample(rownames(train.no), size = size_yes, replace = FALSE) # sampled "no" data
reduced.no <- train.no[rows.no, ]
summary(reduced.no)
# merge yes and no data in a new, more balanced dataset
train.balanced <- rbind(train.yes, reduced.no)
head(train.balanced)
summary(train.balanced)

##########################################################
# Random forest (classification) model
##########################################################
# random forest (classification) - original dataset -> warning: long runtime
set.seed(14, sample.kind = "Rounding")
# fit_rf <- train(y ~ ., method = "rf", data = train.bank, tuneGrid = data.frame(mtry = seq(1, 10)), ntree = 100)
# training with reduced dataset (bank.small)
fit_rf <- train(y ~ ., method = "rf", data = bank.small, tuneGrid = data.frame(mtry = seq(1, 10)), ntree = 100)
ggplot(fit_rf, highlight = TRUE)
# fit_rf$bestTune # mtry = 8 -> optimal value for mtry parameter
y_hat_rf <- predict(fit_rf, test.bank, type = "raw")
confusionMatrix(y_hat_rf, test.bank$y)$overall["Accuracy"] # 0.9170721
cmat_rf <- confusionMatrix(y_hat_rf, test.bank$y)
# but yes is detected in too few cases (low specificity: 0.4783) - due to data imbalance
# importance of variables in the random forest model
imp <- varImp(fit_rf)
imp
# F-score
f_rf <- F_meas(data = y_hat_rf, reference = test.bank$y) 

model_results <- add_result(model_results, "Random Forest",  cmat_rf, f_rf)
# model_results %>% knitr::kable()

# store the results to use in ensemble
y_hat_rf_orig <- y_hat_rf


# random forest (classification) - balanced dataset -> warning: long runtime 
set.seed(14, sample.kind = "Rounding")
# training with balanced dataset
fit_rf <- train(y ~ ., method = "rf", data = train.balanced, tuneGrid = data.frame(mtry = seq(1, 10)), 
                ntree = 100)
ggplot(fit_rf, highlight = TRUE)
# fit_rf$bestTune # mtry = 8 -> optimal value for mtry parameter
y_hat_rf <- predict(fit_rf, test.bank, type = "raw")
confusionMatrix(y_hat_rf, test.bank$y)$overall["Accuracy"] # 0.8398939
cmat_rf <- confusionMatrix(y_hat_rf, test.bank$y)
# yes is detected in many cases (high specificity: 0.9130) - due to better data balance
# importance of variables in the random forest model
imp <- varImp(fit_rf)
imp
# F-score
f_rf <- F_meas(data = y_hat_rf, reference = test.bank$y) 

model_results_bal <- add_result(model_results_bal, "Random Forest",  cmat_rf, f_rf)
# model_results_bal %>% knitr::kable()


##########################################################
# Model results
##########################################################
# Evaluate models
##########################################################
# original dataset
model_results %>% knitr::kable()

# balanced dataset
model_results_bal %>% knitr::kable()

# Note: the 'positive' class in R models is 'no' for the confusion matrix
# Random Forest/Classification Trees and SVM seem suitable with high balanced accuracy and high specificity

