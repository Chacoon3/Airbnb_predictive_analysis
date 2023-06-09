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

get_baseline_accuracy(y_train$perfect_rating_score)
get_baseline_accuracy(y_train$high_booking_rate)
summary(y_train$perfect_rating_score)
summary(y_train$high_booking_rate)
dtm_final <- get_final_dtm(x)


# create dtm -------------------
# amenities dtm
# dtm_am <- get_dtm(x$amenities, tf_idf = F)
# dtm_am_train = dtm_am[1:train_length, ]
# dtm_am_te = dtm_am[(train_length + 1): nrow(dtm_am), ]


dtm_am_pruned <- get_dtm(
    x$amenities, tf_idf = T,
    doc_prop_min = 0.01, doc_prop_max = 0.7, binary = T
  )
dtm_am_pruned_train = dtm_am_pruned[1:train_length, ]
dtm_am_pruned_te = dtm_am_pruned[(train_length + 1): nrow(dtm_am_pruned), ]


# access dtm
dtm_access <- get_dtm(x$access)
dtm_access_train = dtm_access[1:train_length, ]
dtm_access_te = dtm_access[(train_length + 1): nrow(dtm_access), ]



# description dtm
dtm_desc <- get_dtm(
    x$description, tf_idf = T, 
    doc_prop_min = 0.01, doc_prop_max = 0.7, binary = T
  )
dtm_desc_train = dtm_desc[1:train_length, ]
dtm_desc_te = dtm_desc[(train_length + 1): nrow(dtm_desc), ]


# host_about dtm
dtm_ha <- get_dtm(x$host_about)
dtm_ha_train = dtm_ha[1:train_length, ]
dtm_ha_te = dtm_ha[(train_length + 1): nrow(dtm_ha), ]


# house_rules dtm
dtm_hr <- get_dtm(x$house_rules, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_hr_train = dtm_hr[1:train_length, ]
dtm_hr_te = dtm_hr[(train_length + 1): nrow(dtm_hr), ]


# interaction dtm
dtm_itrt <- get_dtm(
  x$interaction, doc_prop_min = 0.01, doc_prop_max = 0.7, binary = T)
dtm_itrt_train = dtm_itrt[1:train_length, ]
dtm_itrt_te = dtm_itrt[(train_length + 1): nrow(dtm_itrt), ]


# neighborhood overview dtm
dtm_no <- get_dtm(x$neighborhood_overview)
dtm_no_train = dtm_no[1:train_length, ]
dtm_no_te = dtm_no[(train_length + 1): nrow(dtm_no), ]


# transit dtm
dtm_transit <- get_dtm(
  x$transit, doc_prop_min = 0.01, doc_prop_max = 0.7, binary = T)
dtm_transit_train = dtm_transit[1:train_length, ]
dtm_transit_te = dtm_transit[(train_length + 1): nrow(dtm_transit), ]


# summary dtm
dtm_summary <- get_dtm(x$summary, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_summary_train = dtm_summary[1:train_length, ]
dtm_summary_te = dtm_summary[(train_length + 1): nrow(dtm_summary), ]


# host verification dtm
dtm_hv <- get_dtm(x$host_verifications, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_hv_train = dtm_hv[1:train_length, ]
dtm_hv_te = dtm_hv[(train_length + 1): nrow(dtm_hv), ]


# train val splitting -----------------
ind_sample = sample(1:train_length, size = 0.75 * train_length)

y_tr = y_train$high_booking_rate[ind_sample]
y_va = y_train$high_booking_rate[-ind_sample]
hbr_tr = y_train$high_booking_rate[ind_sample]
hbr_va = y_train$high_booking_rate[-ind_sample]
prs_tr = y_train$perfect_rating_score[ind_sample]
prs_va = y_train$perfect_rating_score[-ind_sample]


dtm_final_train = dtm_final[1:train_length, ]
dtm_final_te = dtm_final[(1 +train_length): nrow(dtm_final), ]

dtm_final_tr = dtm_final_train[ind_sample,]
dtm_final_va = dtm_final_train[-ind_sample,]

# dtm_am_tr = dtm_am_train[ind_sample, ]
# dtm_am_va = dtm_am_train[-ind_sample, ]

dtm_am_pr_tr = dtm_am_pruned_train[ind_sample, ]
dtm_am_pr_va = dtm_am_pruned_train[-ind_sample, ]


dtm_access_tr = dtm_access_train[ind_sample, ]
dtm_access_va = dtm_access_train[-ind_sample, ]

dtm_desc_tr = dtm_desc_train[ind_sample, ]
dtm_desc_va = dtm_desc_train[-ind_sample, ]

dtm_ha_tr = dtm_ha_train[ind_sample, ]
dtm_ha_va = dtm_ha_train[-ind_sample, ]

dtm_hr_tr = dtm_hr_train[ind_sample, ]
dtm_hr_va = dtm_hr_train[-ind_sample, ]

dtm_inrt_tr = dtm_itrt_train[ind_sample, ]
dtm_inrt_va = dtm_itrt_train[-ind_sample, ]

dtm_no_tr = dtm_no_train[ind_sample, ]
dtm_no_va = dtm_no_train[-ind_sample, ]

dtm_transit_tr = dtm_transit_train[ind_sample, ]
dtm_transit_va = dtm_transit_train[-ind_sample, ]

dtm_summary_tr = dtm_summary_train[ind_sample, ]
dtm_summary_va = dtm_summary_train[-ind_sample, ]

dtm_hv_tr = dtm_hv_train[ind_sample, ]
dtm_hv_va = dtm_hv_train[-ind_sample, ]




# regularized logistic amenities ---------------------
# better without tf-idf
md <- glmnet(
  x = dtm_am_tr, y = hbr_tr,
  weights = ifelse(hbr_tr == 1, 2.5, 1),
  family = 'binomial',
  alpha = 1,
  lambda = 0.001,
  parallel = T,
  trace.it = T
)

pred <- predict(md, newx = dtm_am_va, type = 'response')
get_auc(pred, y_va)
# 0.6990129
vip(md, 30)
lambda_search <- vec_search(
  vec_param = seq(10^-200, 0.1, length.out = 100),
  x = dtm_train, y = y_train$high_booking_rate,
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
# 0.001 optimal




# ranger amenities ---------------
md <- ranger(
  x = dtm_am_tr, y = y_tr %>% as.factor(),
  importance = 'impurity',
  probability = T,
  class.weights = c(1, 2.5),
  num.threads = 4
)
pred <- predict(md, data = dtm_am_va)$predictions[,2]
get_auc(pred, y_va) 
# 0.7247184
vip(md, 35)


# xgb amenities -----------
ncol(dtm_am_pr_tr)
ncol(dtm_am)
md <- xgboost(
  data = dtm_am_tr, label = ifelse(hbr_tr == 1, 1, 0),
  eta = 0.05,
  max_depth = 8,
  nrounds = 50,
  verbose = T,
  print_every_n = 50,
  nthread = 12,
  weight = ifelse(hbr_tr == 1, 7, 1),
  objective = 'binary:logistic',
  eval_metric = 'auc'
)
pred <- predict(md, newdata = dtm_am_va)
get_auc(pred, y_va) 
# 0.7152156
vip(md, 35)



# regularized logistic pruned amenities ----------
# binary appears slightly better
md <- glmnet(
  x = dtm_am_pr_tr, y = hbr_tr,
  weights = ifelse(hbr_tr == 1, 2.5, 1),
  family = 'binomial',
  alpha = 1,
  lambda = 0.001,
  parallel = T,
  trace.it = T
)

pred <- predict(md, newx = dtm_am_pr_va, type = 'response')
get_auc(pred, y_va)
# 0.6990129
vip(md, 30)


# xgb pruned amenities ----------
md <- xgboost(
  data = dtm_am_pr_tr, label = ifelse(hbr_tr == 1, 1, 0),
  eta = 0.2,
  max_depth = 4,
  nrounds = 150,
  verbose = T,
  print_every_n = 50,
  nthread = 12,
  weight = ifelse(hbr_tr == 1, 7, 1),
  objective = 'binary:logistic',
  eval_metric = 'auc'
)
pred <- predict(md, newdata = dtm_am_pr_va)
get_auc(pred, hbr_va) 
# 0.7065387 \\ 0.7087925  \\ 0.7124978
vip(md, 35)

res_ms <- vector_search(
  x = rbind(dtm_am_pr_tr, dtm_am_pr_va),
  y = ifelse(c(hbr_tr, hbr_va) == 1, 1, 0),
  vec_param1 = 4:8, # max depth
  trainer = \(x, y, p1) {
    md <- xgboost(
      data = x, label = y,
      nrounds = 150,
      eta = 0.2,
      max_depth = p1,
      verbose = T,
      print_every_n = 50,
      nthread = 12,
      weight = ifelse(y == 1, 7, 1),
      objective = 'binary:logistic',
      eval_metric = 'auc'
    )
    return(md)
  },
  predictor = \(md, x) {
    pred <- predict(md, newdata = x)
    return(pred)
  },
  measurer = \(y1, y2) {
    return(get_auc(y1, y2))
  }
)
res_ms[order(res_ms$measurement, decreasing = T), ]


res_ms <- matrix_search(
  x = rbind(dtm_am_pr_tr, dtm_am_pr_va),
  y = ifelse(c(hbr_tr, hbr_va) == 1, 1, 0),
  vec_param1 = 3:4 * 50, # n round
  vec_param2 = 2:3 / 10, # eta
  trainer = \(x, y, p1, p2) {
    md <- xgboost(
      data = x, label = y,
      nrounds = p1,
      eta = p2,
      max_depth = 6,
      verbose = T,
      print_every_n = 50,
      nthread = 12,
      weight = ifelse(y == 1, 7, 1),
      objective = 'binary:logistic',
      eval_metric = 'auc'
    )
    return(md)
  },
  predictor = \(md, x) {
    pred <- predict(md, newdata = x)
    return(pred)
  },
  measurer = \(y1, y2) {
    return(get_auc(y1, y2))
  }
)
res_ms[order(res_ms$measurement, decreasing = T), ]

res_ms <- cube_search(
  x = rbind(dtm_am_pr_tr, dtm_am_pr_va),
  y = ifelse(c(hbr_tr, hbr_va) == 1, 1, 0),
  vec_param1 = 1:5 * 25 + 100, # n round
  vec_param2 = 1:10 * 2 / 100, # eta
  vec_param3 = 5:9, # max depth
  trainer = \(x, y, p1, p2, p3) {
    md <- xgboost(
      data = x, label = y,
      nrounds = p1,
      eta = p2,
      max_depth = p3,
      verbose = T,
      print_every_n = 50,
      nthread = 12,
      weight = ifelse(y == 1, 7, 1),
      objective = 'binary:logistic',
      eval_metric = 'auc'
    )
    return(md)
  },
  predictor = \(md, x) {
    pred <- predict(md, newdata = x)
    return(pred)
  },
  measurer = \(y1, y2) {
    return(get_auc(y1, y2))
  }
)
res_ms[order(res_ms$measurement, decreasing = T), ]


# svm amenities ----------
md <- svm(
  x = dtm_am_tr, y = y_tr,
  type = 'C-classification',
  kernel = 'linear',
  probability = T
)
pred <- predict(md, newdata = dtm_am_va)
get_auc(pred, y_va) 
# 0.7152156
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
# 0.5898382
plot_vip = vip(md, 35)
df_vip <- plot_vip$data %>% as.data.frame()
df_vip

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


# logistic description ---------------------
# better with binary
md <- glmnet(
  x = dtm_desc_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_desc_va, type = 'response')
get_auc(pred, y_va)
# 0.6914038 \\ 0.7128931
vip(md, 35)
confusionMatrix(
  data = ifelse(pred >= 0.5, 1, 0) %>% as.factor(),
  reference = y_va %>% as.factor(),
  positive = '1'
)


# logistic host about ---------------------
md <- glmnet(
  x = dtm_ha_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_ha_va, type = 'response')
get_auc(pred, y_va)
# 0.5634598
vip(md, 35)


# logistic host rule ---------------------
md <- glmnet(
  x = dtm_hr_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_hr_va, type = 'response')
get_auc(pred, y_va)
# 0.5868018
vip(md, 35)

# logistic interaction ---------------------
md <- glmnet(
  x = dtm_inrt_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_inrt_va, type = 'response')
get_auc(pred, y_va)
# 0.6077848 \\ 0.6321151 \\ 0.6395231
plot_vip <- get_vip_dataframe(md, dtm_inrt_va)
plot_vip$Importance %>% summary()
dtm_inrt_tr[, which(colnames(dtm_inrt_tr) == 'roommat')] 


# logistic neighborhood overview ---------------------
md <- glmnet(
  x = dtm_no_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_no_va, type = 'response')
get_auc(pred, y_va)
# 0.5879766
vip(md, 35)


# logistic summary ---------------------
# better with binary
md <- glmnet(
  x = dtm_summary_tr >0+0, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_summary_va>0+0, type = 'response')
get_auc(pred, y_va)
# 0.6151317 \\ 0.6622437
vip(md, 35)


# logistic transit ---------------------
# binary transit performs better
md <- glmnet(
  x = dtm_transit_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_transit_va, type = 'response')
get_auc(pred, y_va)
# 0.6032337 \\ 0.6131411 \\ 0.6215553
vip(md, 35)



# logistic host verification --------------
md <- glmnet(
  x = dtm_hv_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_hv_va, type = 'response')
get_auc(pred, y_va)
# 0.5578215
vip(md, 35)

hv_terms = c(
  'governmentid', 
  'jumio_offlinegovernmentid',
  'offlinegovernmentid', 
  'googl', 
  'jumio',
  'identitymanu',
  'review_jumio'
)

dtm_hv_pruned_tr = subset_dtm(dtm_hv_tr, hv_terms)
dtm_hv_pruned_va = subset_dtm(dtm_hv_va, hv_terms)
md <- glmnet(
  x = dtm_hv_pruned_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_hv_pruned_va, type = 'response')
get_auc(pred, y_va)




# feature engineering on original set -------------

feature_engineering_full_set <- function(df) {
  df <- df %>%
    group_by(city) %>%
    mutate(
      city_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      city = ifelse(
        city_count <= 200, 'Other', city # originally 50
      ) %>% as.factor()
    ) %>%
    select(!city_count) %>%
    group_by(market) %>%
    mutate(
      market_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      market = ifelse(
        market_count <= 25, 'Other', market
      ) %>% as.factor()
    ) %>%
    select(!market_count) %>%
    mutate(
      host_same_neighbor = 
        (neighbourhood == host_neighbourhood %>% as.character()) %>% as.factor(),
    ) %>%
    group_by(neighbourhood) %>%
    mutate(
      count_neighbor = n()
    ) %>%
    ungroup() %>%
    mutate(
      neighbourhood = ifelse(count_neighbor < 350, 'Other', neighbourhood) %>%
        as.factor()
    ) %>%
    select(!count_neighbor) %>%
    mutate(
      min_night_length = case_when(
        minimum_nights >= 365 ~ 'Year',
        minimum_nights >= 93 ~ 'Season',
        minimum_nights >= 30 ~ 'Month',
        minimum_nights >= 7 ~ 'Week',
        minimum_nights >= 2 ~ 'Days',
        TRUE ~ 'No'
      ),
      
      host_response_time = host_response_time %>% as.factor()
    ) %>%
    group_by(state) %>%
    mutate(
      state_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      state = ifelse(state_count <= 15, 'other', state) %>% as.factor()
    ) %>%
    select(
      !state_count
    )
  
  return(df)
}


feature_engineering <- function(x) {
  res <- x %>%
    select(
      accommodates,
      availability_30,
      availability_365,
      availability_60,
      availability_90,
      bathrooms,
      bedrooms,
      bed_type,
      beds,
      city,
      cancellation_policy,
      cleaning_fee,
      extra_people,
      first_review,
      guests_included,
      
      host_acceptance_rate,
      host_has_profile_pic,
      host_identity_verified,
      host_is_superhost,
      host_listings_count,
      host_same_neighbor,
      host_response_rate,
      host_response_time,
      host_since,
      
      longitude,
      latitude,
      market,
      min_night_length,
      neighbourhood,
      
      instant_bookable,
      is_business_travel_ready,
      require_guest_phone_verification,
      require_guest_profile_picture,
      room_type,
      # state,
      
      # removed 2023-4-17
      # license,
      
      
      
      property_category,
      price,

    ) %>%
    mutate(
      country = ifelse(x$country_code == 'US', 'US', 'Other') %>%
        as.factor(),
      
      host_response_time = as.factor(x$host_response_time),
      
      price_per_person = x$price / ifelse(x$accommodates == 0, 1, x$accommodates),
      
      ppp_ind = ifelse(price_per_person > median(price_per_person),1 , 0),
      
      price = ifelse(price < 1, 1, price) %>% # this is to prevent -inf
        log(), 
      
      price_per_sqfeet = x$price / 
        ifelse(x$square_feet == 0, median(x$square_feet), x$square_feet),
      
      monthly_price =
        ifelse(x$monthly_price < 1, 1, x$monthly_price) %>%
        log(),
      
      square_feet = 
        ifelse(x$square_feet == 0, median(x$square_feet), x$square_feet)
    )
  
  
  return(res)
}


x_all <- x %>%
  feature_engineering_full_set() %>%
  feature_engineering()

md_dummy = dummyVars(~., x_all, fullRank = T)
x_all_dum = predict(md_dummy, x_all)
x_dum_train = x_all_dum[1:nrow(y_train), ]

# split
x_dum_tr = x_dum_train[ind_sample, ]
x_dum_va = x_dum_train[-ind_sample, ]


# dtm for consideration: ------------
# very high: amenities pruned, description
# high: summary, transit, host verification
# medium: host rule, interaction

# merged dtm --------------

dtm_merged_train = cbind(
  dtm_am_pruned_train, dtm_desc_train,
  dtm_summary_train, dtm_transit_train, dtm_hv_train,
  dtm_hr_train, dtm_itrt_train
)

dtm_merged_te = cbind(
  dtm_am_pruned_te, dtm_desc_te,
  dtm_summary_te, dtm_transit_te, dtm_hv_te,
  dtm_hr_te, dtm_itrt_te
)

dtm_merged_tr = dtm_merged_train[ind_sample, ]
dtm_merged_va = dtm_merged_train[-ind_sample, ]
ncol(dtm_merged_train)

md <- glmnet(
    x = dtm_merged_tr, y = hbr_tr,
    weights = ifelse(hbr_tr == 1, 5, 1),
    family = 'binomial',
    alpha = 1,
    lambda = 0.001,
    parallel = T
  )

pred <- predict(md, newx = dtm_merged_va, type = 'response')
get_auc(pred, y_va)
# 0.6990129 \\ 0.7681656 \\ 0.7759867

get_important_feature(md, dtm_merged_va, 0.75)
get_vip_dataframe(md, dtm_merged_va) %>%
  select(Importance) %>%
  boxplot()


md <- glmnet(
  x = dtm_merged_tr, y = prs_tr,
  weights = ifelse(prs_tr == 1, 5, 1),
  family = 'binomial',
  alpha = 1,
  lambda = 0.001,
  parallel = T
)

pred <- predict(md, newx = dtm_merged_va, type = 'response')
get_auc(pred, prs_va)
# 0.7696559 \\ 0.7035021


df_vip = get_vip_dataframe(md, dtm_merged_tr)
df_vip %>%
  arrange(desc(Sign), desc(Importance)) %>%
  head(20)




# subset merged dtm -------------------
important_token = get_important_feature(md, dtm_merged_tr, 0.75)


dtm_subset_train <- subset_dtm(
  dtm_merged_train, important_token
)

dtm_subset_te <- subset_dtm(
  dtm_merged_te, important_token
)


get_baseline_accuracy(c(hbr_tr, hbr_va))

dtm_subset_tr = dtm_subset_train[ind_sample, ]
dtm_subset_va = dtm_subset_train[-ind_sample, ]


md <- glmnet(
  x = dtm_subset_tr, y = hbr_tr,
  weights = ifelse(hbr_tr == 1, 1.6, 1),
  family = 'binomial',
  alpha = 1,
  lambda = 0.000000001,
  parallel = T
)
pred <- predict(md, newx = dtm_subset_va, type = 'response')
get_auc(pred, y_va) 
# 0.755716  \\ 0.7633405
get_cutoff_dataframe(pred, y_va) %>%
plot_cutoff_dataframe()


vs_res <- vector_search(
  x = rbind(dtm_subset_tr, dtm_subset_va),
  y = c(hbr_tr, hbr_va),
  vec_param1 = seq(0.000000001, 1, length.out = 50),
  trainer = \(x,y,p) {
    md <- glmnet(
      x = x, y = y,
      weights = ifelse(y == 1, 5, 1),
      family = 'binomial',
      alpha = 1,
      lambda = p,
      parallel = T,
      trace.it = 0
    )
    
    return(md)
  },
  predictor = \(m,x) {
    return(
      predict(m, newx = x, type = 'response')
    )
  },
  measurer = \(y1,y2) {
    return(
      get_auc(y1, y2)
    )
  },
  n_per_round = 2,
  verbose = T
)

vs_res %>%
  arrange(
    desc(measurement)
  )
# 1.6 appears optimal for weight
#       param1 measurement
# 1  0.000000001   0.7685309
# 2  0.020408164   0.6999819
# 3  0.040816327   0.6682916


# logistic dummy test ----------
# colnames(x_dum_tr)
vec_col_names = c('availability_30', 'availability_60', 'price_per_sqfeet')
which(colnames(x_dum_tr) %in% vec_col_names)
md <- glmnet(
  x = x_dum_tr[, which(colnames(x_dum_tr) %in% vec_col_names)],
  y = hbr_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 0
)

pred <- predict(
  md, 
  newx = x_dum_va[, which(colnames(x_dum_tr) %in% vec_col_names)], 
  type = 'response')
get_auc(pred, y_va)
# 0.6151317


# combining column data with nlp ------------------

# x_all[1:nrow(y_train), ] %>%
#   mutate(
#     hbr = y_train$high_booking_rate
#   ) %>%
#   group_by(host_response_time) %>%
#   summarise(
#     count = n(),
#     p = sum(hbr == 1),
#     n = sum(hbr != 1),
#     pn_ratio = p /n,
#     p_ratio = p / count,
#     n_ratio = n/ count
#   )

summary(x$availability_30 %>% as.numeric())
x_tr_av30 <- x_all[1:nrow(y_train), ] %>%
  mutate(
    hbr = y_train$high_booking_rate
  ) %>%
  group_by(cancellation_policy) %>%
  summarise(
    count = n(),
    p = sum(hbr == 1),
    n = sum(hbr != 1),
    pn_ratio = p /n,
    p_ratio = p / count,
    n_ratio = n/ count
  ) %>%
  arrange(desc(pn_ratio))
x_tr_av30$pn_ratio %>% boxplot()


x_selected <- x_all %>%
  select(
    availability_30,
    host_response_time,
    monthly_price,
    min_night_length,
    host_listings_count,
    price,
    price_per_person,
    price_per_sqfeet,
    country,
    room_type,
    neighbourhood,
    is_business_travel_ready,
    cancellation_policy
  ) %>%
  mutate(
    availability_30 = availability_30 %>% as.numeric(),
    
    host_response_time = case_when(
      host_response_time == 'within an hour' ~ 'an hour',
      host_response_time == 'within a few hours' ~ 'a few hours',
      TRUE ~ 'Other'
    )
  )

md_dum2 = dummyVars(~., x_selected, fullRank = T)
x_dum_selected = predict(md_dum2, x_selected)

merged_dum = cbind(
  rbind(
    dtm_subset_train, dtm_subset_te
  ),
  x_dum_selected
)

merged_train = merged_dum[1:train_length, ]
merged_te = merged_dum[(train_length + 1): nrow(merged_dum), ]


merged_tr = merged_train[ind_sample, ]
merged_va = merged_train[-ind_sample, ]
merged_hbr_tr = y_train$high_booking_rate[ind_sample]
merged_hbr_va = y_train$high_booking_rate[-ind_sample]
merged_prs_tr = y_train$perfect_rating_score[ind_sample]
merged_prs_va = y_train$perfect_rating_score[-ind_sample]
ncol(merged_tr)

md_merged <- glmnet(
  x = merged_tr, y = merged_hbr_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7,
  weights = ifelse(merged_hbr_tr == 1, 5, 1)
)

pred <- predict(md_merged, newx = merged_va, type = 'response')
get_auc(pred, merged_hbr_va)
# 0.7871135 \\ 0.8005933 \\ 0.8108367
vip(md_merged, 35)


md_ranger <- ranger(
  x = merged_tr,
  y = merged_hbr_tr,
  num.trees=600,
  importance="impurity",
  probability = TRUE,
  num.threads = 12,
  max.depth	= 7,
  class.weights = c(1, 10),
  verbose = T
)

hbr_prob_ranger <- 
  predict(md_ranger, data = merged_va)$predictions[,2]
get_auc(hbr_prob_ranger, merged_hbr_va)
# 0.7894064


md_xgb <- xgboost(
  data = merged_tr,
  label = ifelse(merged_hbr_tr == 'YES', 1, 0),
  max.depth = 7,
  eta = 0.06, 
  nrounds = 600,
  objective = "binary:logistic",
  eval_metric = "auc", 
  verbose = T,
  weight = ifelse(merged_hbr_tr == 'YES', 9, 1),
  print_every_n = 100,
  nthread = 12
)
xgb_pred <- predict(md_xgb, merged_va)
get_auc(xgb_pred, merged_hbr_va)
# 0.8530816 \\ 0.8600508

plot_roc(xgb_pred, merged_hbr_va %>% as.factor())

# cannot weight positive cases too high
md_prs_xgb <- xgboost(
  data = merged_tr,
  label = ifelse(merged_prs_tr == 1, 1, 0),
  max.depth = 7,
  eta = 0.06, 
  nrounds = 600,
  objective = "binary:logistic",
  eval_metric = "auc", 
  verbose = T,
  weight = ifelse(merged_prs_tr == 1, 1.5, 1),
  print_every_n = 100,
  nthread = 12
)
xgb_prs_pred <- predict(md_prs_xgb, merged_va)
get_auc(xgb_prs_pred, merged_prs_va)
get_cutoff_dataframe(xgb_prs_pred, merged_prs_va, step = 0.01) %>%
  plot_cutoff_dataframe()


