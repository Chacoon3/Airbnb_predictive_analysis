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
library(tree)
library(rpart)
library(xgboost)
library(e1071)


# read ------------------------------
folder_dir = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis_copy\Data)"


x <- get_cleaned(folder_dir, F)
y_train <- read.csv(r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis_copy\Data\airbnb_train_y_2023.csv)")
y_train <- y_train %>%
  mutate(
    high_booking_rate = 
      ifelse(high_booking_rate== 'YES', 1, 0) %>% as.factor(),
    perfect_rating_score = 
      ifelse(perfect_rating_score== 'YES', 1, 0) %>% as.factor(),
  )



train_length = nrow(y_train)


# create dtms --------------
# amenities
dtm_am_pruned <- get_dtm(
  x$amenities, tf_idf = T,
  doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_am_pruned_train = dtm_am_pruned[1:train_length, ]
dtm_am_pruned_te = dtm_am_pruned[(train_length + 1): nrow(dtm_am_pruned), ]


# description dtm
dtm_desc <- get_dtm(
  x$description, tf_idf = T, 
  doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_desc_train = dtm_desc[1:train_length, ]
dtm_desc_te = dtm_desc[(train_length + 1): nrow(dtm_desc), ]


# summary dtm
dtm_summary <- get_dtm(x$summary, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_summary_train = dtm_summary[1:train_length, ]
dtm_summary_te = dtm_summary[(train_length + 1): nrow(dtm_summary), ]


# transit dtm
dtm_transit <- get_dtm(
  x$transit, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_transit_train = dtm_transit[1:train_length, ]
dtm_transit_te = dtm_transit[(train_length + 1): nrow(dtm_transit), ]


# host verification dtm
dtm_hv <- get_dtm(
  x$host_verifications, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_hv_train = dtm_hv[1:train_length, ]
dtm_hv_te = dtm_hv[(train_length + 1): nrow(dtm_hv), ]


# house rule
dtm_hr <- get_dtm(
  x$house_rules, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_hr_train = dtm_hr[1:train_length, ]
dtm_hr_te = dtm_hr[(train_length + 1): nrow(dtm_hr), ]


# interaction dtm
dtm_itrt <- get_dtm(
  x$interaction, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_itrt_train = dtm_itrt[1:train_length, ]
dtm_itrt_te = dtm_itrt[(train_length + 1): nrow(dtm_itrt), ]


# split ---------------------
ind_sample = sample(1:train_length, size = 0.75 * train_length)

hbr_tr = y_train$high_booking_rate[ind_sample]
hbr_va = y_train$high_booking_rate[-ind_sample]
prs_tr = y_train$perfect_rating_score[ind_sample]
prs_va = y_train$perfect_rating_score[-ind_sample]

# dtm_am_pr_tr = dtm_am_pruned_train[ind_sample, ]
# dtm_am_pr_va = dtm_am_pruned_train[-ind_sample, ]
# 
# dtm_desc_tr = dtm_desc_train[ind_sample, ]
# dtm_desc_va = dtm_desc_train[-ind_sample, ]
# 
# dtm_transit_tr = dtm_transit_train[ind_sample, ]
# dtm_transit_va = dtm_transit_train[-ind_sample, ]
# 
# dtm_summary_tr = dtm_summary_train[ind_sample, ]
# dtm_summary_va = dtm_summary_train[-ind_sample, ]
# 
# dtm_hv_tr = dtm_hv_train[ind_sample, ]
# dtm_hv_va = dtm_hv_train[-ind_sample, ]
# 
# 
# dtm_hr_tr = dtm_hr_train[ind_sample, ]
# dtm_hr_va = dtm_hr_train[-ind_sample, ]
# 
# 
# dtm_inrt_tr = dtm_inrt_train[ind_sample, ]
# dtm_inrt_va = dtm_inrt_train[-ind_sample, ]


dtm_merged_train = cbind(
  dtm_am_pruned_train, dtm_desc_train, dtm_hr_train,
  dtm_hv_train, dtm_itrt_train, dtm_transit_train,
  dtm_summary_train
)
dtm_merged_tr = dtm_merged_train[ind_sample, ]
dtm_merged_va = dtm_merged_train[-ind_sample, ]


dtm_mergeed_te = cbind(
  dtm_am_pruned_te, dtm_desc_te, dtm_hr_te,
  dtm_hv_te, dtm_itrt_te, dtm_transit_te,
  dtm_summary_te
  )


# train -------------

md_hbr_xgb <- xgboost(
  data = dtm_merged_tr, label = ifelse(hbr_tr == 1, 1, 0),
  eta = 0.06,
  max_depth = 7,
  nrounds = 500,
  verbose = T,
  print_every_n = 100,
  nthread = 8,
  weight = ifelse(hbr_tr == 1, 7, 1),
  objective = 'binary:logistic',
  eval_metric = 'auc'
)

pred_hbr_xgb <- 
  predict(md_hbr_xgb, newdata = dtm_merged_va)
get_auc(pred_hbr_xgb, hbr_va)
# 0.7040643 \\ 0.708013 \\ 0.7690409 \\ 0.7817721


if_xgb <- get_important_feature(md_hbr_xgb, dtm_merged_va, quantile_threshold = 0.6)
md_hbr_xgb2 <- xgboost(
  data = dtm_merged_tr %>% subset_dtm(vec_col_name = if_xgb), 
  label = ifelse(hbr_tr == 1, 1, 0),
  eta = 0.06,
  max_depth = 7,
  nrounds = 500,
  verbose = T,
  print_every_n = 100,
  nthread = 8,
  weight = ifelse(hbr_tr == 1, 7, 1),
  objective = 'binary:logistic',
  eval_metric = 'auc'
)

pred_hbr_xgb2 <- 
  predict(md_hbr_xgb2, 
          newdata = dtm_merged_va %>% subset_dtm(vec_col_name = if_xgb))
get_auc(pred_hbr_xgb2, hbr_va)
# 0.7814226 \\ 0.7793241 \\ 0.781284



md_hbr_logit <- glmnet(
  x = dtm_merged_tr, y = hbr_tr,
  weights = ifelse(hbr_tr == 1, 1.6, 1),
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7,
  parallel = T
)
pred_hbr_logit <- 
  predict(md_hbr_logit, newx = dtm_merged_va, type = 'response')
get_auc(pred_hbr_logit, hbr_va)
# 0.7585049


if_logit <- get_important_feature(md_hbr_logit, dtm_merged_va, 0.6)
md_hbr_logit2 <- glmnet(
  x = dtm_merged_tr %>% subset_dtm(if_logit), 
  y = hbr_tr,
  weights = ifelse(hbr_tr == 1, 1.6, 1),
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7,
  parallel = T
)
pred_hbr_logit2 <- 
  predict(md_hbr_logit2, 
          newx = dtm_merged_va %>% subset_dtm(if_logit),
          type = 'response')
get_auc(pred_hbr_logit2, hbr_va)
# 0.7422923



md_hbr_ranger <- ranger(
  x = dtm_merged_tr,
  y = hbr_tr,
  num.trees=600,
  importance="impurity",
  probability = TRUE,
  num.threads = 12,
  max.depth	= 7,
  class.weights = c(1, 3),
  verbose = T
)

pred_hbr_ranger <- 
  predict(md_hbr_ranger, data = dtm_merged_va)$predictions[,2]
get_auc(pred_hbr_ranger, hbr_va)


md_hbr_ranger2 <- ranger(
  x = dtm_merged_tr %>% subset_dtm(if_xgb),
  y = hbr_tr,
  num.trees=600,
  importance="impurity",
  probability = TRUE,
  num.threads = 12,
  max.depth	= 7,
  class.weights = c(1, 3),
  verbose = T
)

pred_hbr_ranger2 <- 
  predict(md_hbr_ranger2, 
          data = dtm_merged_va %>% subset_dtm(if_xgb))$predictions[,2]
get_auc(pred_hbr_ranger2, hbr_va)
# 0.6526232 \\ 0.7118094


md_prs_ranger <- ranger(
  x = dtm_merged_tr,
  y = prs_tr,
  num.trees=600,
  importance="impurity",
  probability = TRUE,
  num.threads = 12,
  max.depth	= 7,
  class.weights = c(1, 2),
  verbose = T
)

pred_prs_ranger <- 
  predict(md_prs_ranger, data = dtm_merged_va)$predictions[,2]
get_auc(pred_prs_ranger, prs_va)
# 0.6647299


md_prs_xgb <- xgboost(
  data = dtm_merged_tr,
  label = ifelse(prs_tr == 1, 1, 0),
  max.depth = 7,
  eta = 0.06, 
  nrounds = 600,
  objective = "binary:logistic",
  eval_metric = "auc", 
  verbose = T,
  weight = ifelse(prs_tr == 1, 3, 1),
  print_every_n = 100,
  nthread = 12
)
pred_prs_xgb <- predict(md_prs_xgb, dtm_merged_va)
get_auc(pred_prs_xgb, prs_va)
# 0.6756432 \\ 0.7084855


md_logit <- glmnet(
  x = dtm_am_pr_tr,
  y = prs_tr,
  family = 'binomial',
  # weights = ifelse(prs_tr == 1, 3, 1),
  alpha = 1,
  lambda = 10^-6,
  parallel = T
)
logit_pred <- predict(md_logit, dtm_am_pr_va)
get_auc(logit_pred, prs_va)
# 0.6387255