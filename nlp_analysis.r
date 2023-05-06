source('Library\\data_cleaning.r')
source('Library\\utils.r')
library(tidyverse)
library(Metrics)
library(caret)
library(class)
library(readr)
library(text2vec)
library(tm)
library(SnowballC)
library(vip)
library(textdata)
library(tidytext)
library(quanteda)
library(glmnet)
library(ranger)

# read
folder_dir = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis_copy\Data)"
x <- get_cleaned(folder_dir, F)
wd <- getwd()
setwd(folder_dir)
y_train <- read.csv('airbnb_train_y_2023.csv')
y_train <- y_train %>%
  mutate(
    high_booking_rate = 
      ifelse(high_booking_rate== 'YES', 1, 0) %>% as.factor(),
    perfect_rating_score = 
      ifelse(perfect_rating_score== 'YES', 1, 0) %>% as.factor(),
  )
setwd(wd)


train_length = nrow(y_train)


# amenities dtm
dtm <- get_dtm(x$amenities, ngram = c(1L,2L))
dtm_train = dtm[1:train_length, ]
dtm_te = dtm[(train_length + 1): nrow(dtm), ]


ind_sample = sample(1:train_length, size = 0.75 * train_length)
dtm_tr = dtm_train[ind_sample, ]
dtm_va = dtm_train[-ind_sample, ]
y_tr = y_train$high_booking_rate[ind_sample]
y_va = y_train$high_booking_rate[-ind_sample]


# access dtm
dtm_access <- get_dtm(x$access)
dtm_access_train = dtm_access[1:train_length, ]
dtm_access_te = dtm_access[(train_length + 1): nrow(dtm), ]


ind_sample = sample(1:train_length, size = 0.75 * train_length)
dtm_access_tr = dtm_access_train[ind_sample, ]
dtm_access_va = dtm_access_train[-ind_sample, ]
y_tr = y_train$high_booking_rate[ind_sample]
y_va = y_train$high_booking_rate[-ind_sample]


# description dtm
dtm_desc <- get_dtm(x$description)
dtm_desc_train = dtm_desc[1:train_length, ]
dtm_desc_te = dtm_desc[(train_length + 1): nrow(dtm), ]


ind_sample = sample(1:train_length, size = 0.75 * train_length)
dtm_desc_tr = dtm_desc_train[ind_sample, ]
dtm_desc_va = dtm_desc_train[-ind_sample, ]
y_tr = y_train$high_booking_rate[ind_sample]
y_va = y_train$high_booking_rate[-ind_sample]



# logistic amenities ---------------------
md <- glmnet(
  x = dtm_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_va, type = 'response')
get_auc(pred, y_va)
vip(md, 35)


# logistic access ----------------------
# lambda 0 appears to be optimal for alpha = 1
lambda_search <- vec_search(
  vec_param = 0,
  x = dtm_access_train, y = y_train$high_booking_rate,
  trainer = \(x, y, param) {
    md <- glmnet(
      x = x, y = y,
      family = 'binomial',
      alpha = 1,
      lambda = param
    )
    return(md)
  },
  predictor = \(md, x) {
    pred <- predict(md, newx = x, type = 'response')
    return(pred)
  },
  measurer = \(y1, y2){
    return(
      get_auc(y1, y2)
    )
  } 
)
lambda_search[order(lambda_search$measurement, decreasing = T), ][1:5,]


md <- glmnet(
  x = dtm_access_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_access_va, type = 'response')
get_auc(pred, y_va)
vip(md, 35)



vec_auc <- cross_val(
  x = dtm_train,
  y = y_train$high_booking_rate,
  trainer = \(x_set, y_set) {
    model <- glmnet(
      x = x_set, y = y_set,
      family = 'binomial',
      alpha = 1,
      lambda = 10^-7
    )
    return(model)
  },
  predictor = \(model, x) {
    pred <- predict(model, newx = x, type = 'response')
    return(pred)
  },
  measurer = \(y1, y2) {
    return(
      get_auc(y1, y2)
    )
  }
)
vec_auc


vec_doc_min_prop
for (ind in 1:length(vec_doc_min_prop)) {
  
}



# logistic description ---------------------
md <- glmnet(
  x = dtm_desc_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_desc_va, type = 'response')
get_auc(pred, y_va)
vip(md, 35)