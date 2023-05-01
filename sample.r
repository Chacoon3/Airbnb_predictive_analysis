# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file
source('Library\\data_cleaning.r')
source('Library\\utils.r')
library(xgboost)
library(ranger)
library(caret)
library(glmnet)
options(scipen = 999)
# replace the string with the directory of the project folder "Data"
# the original datasets MUST be placed under the Data folder for the following script to run correctly.
folder_dir = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data)"


# read
x_full_set <- get_cleaned(folder_dir)

y_train <- read.csv('Data\\airbnb_train_y_2023.csv')


hbr <- y_train$high_booking_rate %>% as.factor()
hbr <- ifelse(hbr == 'YES', 1, 0)
prs <- y_train$perfect_rating_score %>% as.factor()
prs <- ifelse(prs == 'YES', 1, 0)
x_train <- x_full_set[1:nrow(y_train),]
x_test <- x_full_set[(nrow(y_train) + 1): nrow(x_full_set),]
