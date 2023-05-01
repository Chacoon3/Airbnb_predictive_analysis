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

# This line needs only to be run once, which exports two csv files, one for the training X's, the other for the testing X's. Once the two files were created you only need to run the read.csv statements under this line.
# export_cleaned(folder_dir)


feature_engineering_full_set <- function(df) {
  df <- df %>%
    group_by(city) %>%
    mutate(
      city_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      city = ifelse(
        city_count <= 50, 'OTHER', city
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
        market_count <= 25, 'OTHER', market
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
      neighbourhood = ifelse(count_neighbor < 350, 'Other', neighbourhood)
    ) %>%
    select(!count_neighbor)
  
  return(df)
}


# read
# x_train <- read.csv('Data\\x_train_clean.csv')
# x_test <- read.csv('Data\\x_test_clean.csv')
x_full_set <- get_cleaned(folder_dir)
y_train <- read.csv('Data\\airbnb_train_y_2023.csv')

x <- feature_engineering_full_set(x_full_set)

hbr <- y_train$high_booking_rate %>% as.factor()
prs <- y_train$perfect_rating_score %>% as.factor()
x_train <- x[1:nrow(y_train),]
x_test <- x[(nrow(y_train) + 1): nrow(x),]


# test
nrow(x_train) == nrow(y_train)
nrow(x_test) == 12205
(length(hbr) == length(prs)) && (length(hbr) == nrow(y_train))


#view -----------------
summary(x_train$host_neighbourhood)
tar_fac <- y_train$perfect_rating_score %>% as.factor()

obj_test <- x_train %>%
  cbind(tar_fac) %>% 
  group_by(neighbourhood) %>%
  mutate(
    inst_count = n()
  ) %>%
  ungroup() %>%
  filter(tar_fac == 'YES') %>%
  group_by(neighbourhood) %>%
  mutate(
    p_count = n(),
    p_rate = p_count / inst_count
  ) %>%
  select(neighbourhood, inst_count, p_rate) %>%
  arrange(inst_count) %>%
  distinct()


ggplot(data = obj_test, aes(x = neighbourhood, y = inst_count)) +
  geom_bar(stat = "identity")

boxplot(obj_test$inst_count)
summary(obj_test$inst_count)
hist(x$monthly_price)
summary(x$price)


# train-validation split -------------------------
sampled = sample(1:nrow(x_train), 0.85 * nrow(x_train))
x_tr = x_train[sampled, ]
x_va = x_train[-sampled, ]
hbr_tr = hbr[sampled]
hbr_va = hbr[-sampled]
prs_tr = prs[sampled]
prs_va = prs[-sampled]


# summary(x$host_neighbourhood)
# codes start here ----------------------------

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
      # host_location,
      host_response_rate,
      # host_response_time, in mutate 
      host_since,
      
      longitude,
      latitude,
      market,
      neighbourhood,
      host_same_neighbor,
      
      # added 2023-4-16




      instant_bookable,
      # Internet,
      is_business_travel_ready,
      # TV,
      room_type,
      
      # added 2023-4-17
      `self check-in`,
      `hair dryer`,
      hangers,
      washer,
      shampoo,
      kitchen,
      `first aid kit`,
      `24-hour check-in`,
      `lock on bedroom door`,
      # `air conditioning`, cannot select
      `free parking on premises`,
      `fire extinguisher`,
      tv,
      `carbon monoxide detector`,
      # `hot water`,
      # `buzzer`,
      # family,
      
      
      # removed 2023-4-17
      # license,

      
      
      property_category,
      price,

      
      access_snmt,
      desc_snmt,
      host_about_snmt,
      house_rules_snmt,
      interaction_snmt,
      neighborhood_snmt,
      notes_snmt,
      summary_snmt
    ) %>%
    mutate(
      country = ifelse(x$country_code == 'US', 'US', 'Other') %>%
        as.factor(),
      
      host_response_time = as.factor(x$host_response_time),

      price = ifelse(price < 1, mean(price), price) %>% # this is to prevent -inf
              log(), 

      price_per_sqfeet = x$price / 
        ifelse(x$square_feet == 0, median(x$square_feet), x$square_feet),
      
      monthly_price =
        ifelse(x$monthly_price < 1, mean(x$monthly_price), x$monthly_price) %>%
        log(),
      
      square_feet = 
        ifelse(x$square_feet == 0, median(x$square_feet), x$square_feet)
      
    )
  return(res)
}


x_tr_rf <- feature_engineering(x_tr) 
x_va_rf <- feature_engineering(x_va)
x_te_rf <- feature_engineering(x_test)
hbr_tr_rf <- hbr_tr
prs_tr_rf <- prs_tr
md_dummy <- dummyVars(formula = ~., x_tr_rf, fullRank = TRUE)
x_tr_rf_dummy <- predict(md_dummy, x_tr_rf)
x_va_rf_dummy <- predict(md_dummy, x_va_rf)
x_te_rf_dummy <- predict(md_dummy, x_te_rf)
prs_tr_dummy = ifelse(prs_tr_rf == 'YES', 1, 0)
hbr_tr_dummy = ifelse(hbr_tr_rf == 'TES', 1, 0)
prs_va_dummy = ifelse(prs_va == 'YES', 1, 0)
hbr_va_dummy = ifelse(hbr_va == 'TES', 1, 0)


# xgb prs ----------------------------
md_prs_xgb <- xgboost(
  data = as.matrix(x_tr_rf_dummy),
  label = prs_tr_dummy,
  nrounds = 2000,
  print_every_n = 100,
  objective = 'binary:logistic',
  eval_metric = "auc"
)
y_pred_prob_xgb <- predict(md_prs_xgb, newdata = x_va_rf_dummy)
plot_roc(y_pred_prob_xgb, prs_va)
get_auc(y_pred_prob_xgb, prs_va)

get_cutoff_dataframe(y_pred_prob_xgb, prs_va_dummy, level = c(0,1)) %>%
  plot_cutoff_dataframe()
# 0.7906043



# rf ----------------------------
md_prs_rf <- randomForest::randomForest(
  x = x_tr_rf_dummy, 
  y = prs_tr_rf, 
  maxnodes = 80,
  nodesize = 1,
  mtry = 20
)

y_pred_rf <- predict(md_prs_rf, newdata = x_va_rf_dummy, 'prob')

plot_roc(y_pred_rf[,2], prs_va)
get_auc(y_pred_rf[,2], prs_va)



# ranger hbr ---------------------------------
md_hbr_rf_ranger <- ranger(x = x_tr_rf_dummy, y = hbr_tr_rf,
                 mtry=22, num.trees=500,
                 importance="impurity",
                 probability = TRUE)
y_pred_prob_hbr_ranger <- 
  predict(md_hbr_rf_ranger, data = x_va_rf_dummy)$predictions[,2]
plot_roc(y_pred_prob_hbr_ranger, hbr_va)
get_auc(y_pred_prob_hbr_ranger, hbr_va)
# 0.8733637 \\ 0.873 \\ 0.8663719


# ranger prs ----------------------------
# optimal num tree = 800, mtry = 26
md_hbr_rf_ranger <- ranger(x = x_tr_rf_dummy, y = prs_tr_rf,
                           mtry=26, 
                           num.trees=800,
                           importance="impurity",
                           probability = TRUE)
y_pred_prob_prs_ranger <- 
  predict(md_hbr_rf_ranger, data = x_va_rf_dummy)$predictions[,2]
plot_roc(y_pred_prob_prs_ranger, prs_va)
get_auc(y_pred_prob_prs_ranger, prs_va)
# last 0.7924
# highest 0.804

df_cutoff <- 
  get_cutoff_dataframe(y_pred_prob_prs_ranger, prs_va, 
                       level = c('NO', 'YES'),
                       max_fpr = 0.08)

plot_cutoff_dataframe(df_cutoff)

df_cutoff$cutoff_bound[1]

final_pred_prob = predict(md_hbr_rf_ranger, data = x_te_rf_dummy)$predictions[,2]
final_pred_cls = ifelse(final_pred_prob > df_cutoff$cutoff_bound[1], 'YES', 'NO')
wd <- getwd()
setwd(folder_dir)
write.table(final_pred_cls, "perfect_rating_score_group5.csv", row.names = FALSE)
setwd(wd)


# new code testing ----------------
vec_param = 0:3 * 3 + 20
vec_auc <- iterate_on(
  on = vec_param,
  action = \(param) {
    md_hbr_rf_ranger <- ranger(
      x = x_tr_rf_dummy, 
      y = prs_tr_rf,
      num.trees=800, # 800 appears optimal
      mtry = param,
      importance="impurity",
      probability = TRUE
    )
    
    y_pred_prob_prs_ranger <- 
      predict(md_hbr_rf_ranger, data = x_va_rf_dummy)$predictions[,2]
    return(get_auc(y_pred_prob_prs_ranger, prs_va))
  }
)


ggplot(
  data = data.frame(
    mtry = vec_param,
    auc = vec_auc
  ),
  aes(x = mtry, y = auc)
) + geom_line()



# logistic lasso ------------------
# best lambda appears to be 10^-7
vec_lasso_lambda = seq(from = 10^-7, to = 2, length.out = 100)
vec_lass_auc <- iterate_on(
  on = vec_lasso_lambda,
  action = \(param) {
    logistic_lasso_prs <- glmnet(
      x = x_tr_rf_dummy, 
      y = prs_tr_dummy,
      lambda = param, 
      alpha = 0,
      family="binomial"
    )
    
    pred_prob <- predict(logistic_lasso_prs, newx = x_va_rf_dummy, type = "response")
    return(
      get_auc(pred_prob, prs_va_dummy)
    )
  },
  
  verbose = FALSE
)


ggplot(
  data = data.frame(
    lambda = vec_lasso_lambda,
    auc = vec_lass_auc
  ),
  aes(x = lambda, y = auc)
) + geom_line()


# fpr checking flow -----------------------
check_time = 5
sample_size = 30000
vec_fpr = rep(0, check_time)
vec_tpr = rep(0, check_time)
set_cutoff = 0.3
x_source = rbind(x_tr_rf_dummy, x_va_rf_dummy)
for (ind in 1:check_time) {
  sampled = sample(1:nrow(x_source), size = sample_size, replace = FALSE)
  x_sampled = x_source[sampled,]
  y_sampled = prs[sampled]
  # y_pred_prob = predict(md_hbr_rf_ranger, data = x_sampled)$predictions[,2]
  # y_pred_prob = predict(md_prs_xgb, newdata = x_sampled)
  y_pred_prob = predict(md_prs_logit, newdata = data.frame(x_sampled))
  y_pred_cls = ifelse(y_pred_prob > set_cutoff, 'YES', 'NO')
  
  
  n = 'NO'
  p = 'YES'
  get_fpr <- function(y_pred, y_valid) {
    count_fp = sum((y_valid == n) & (y_pred == p))
    count_tn = sum(y_valid == n)
    return(count_fp / count_tn)
  }
  get_tpr <- function(y_pred, y_valid) {
    count_tp = sum((y_valid == p) & (y_valid == y_pred))
    count_p = sum(y_valid == p)
    return(count_tp / count_p)
  }
  
  fpr = get_fpr(y_pred_cls, y_sampled)
  vec_fpr[ind] = fpr
  
  tpr = get_tpr(y_pred_cls, y_sampled)
  vec_tpr[ind] = tpr
}



# logistic hbr --------------------------
md_logistic <- glm(data.frame(x_tr_rf_dummy, hbr_tr), formula = hbr_tr~., family = 'binomial')

y_pred_prob_logit <- predict(md_logistic, newdata = data.frame(x_va_rf_dummy), type = 'response')
plot_roc(y_pred_prob_logit, hbr_va)



# logistic prs --------------------------
md_prs_logit <- glm(data.frame(x_tr_rf_dummy, prs_tr), formula = prs_tr~., family = 'binomial')

y_pred_prob_prs_logit <- predict(md_prs_logit, newdata = data.frame(x_va_rf_dummy), type = 'response')
plot_roc(y_pred_prob_prs_logit, prs_va)




# analysis ------------------------


summary(x_tr$room_type)
summary(x_tr_rf$price)
boxplot(x_tr_rf$price)
boxplot(x_tr_rf$monthly_price)
summary(x_tr_rf$price_per_sqfeet)
nrow(x_tr)
nrow(x_tr_rf)
nrow(x_tr_rf_dummy)

x_tr %>%
  filter(hbr_tr=='YES') %>%
  group_by(is_business_travel_ready) %>%
  summarise(
    number = n()) 


x_tr %>%
  filter(prs_tr=='YES') %>%
  group_by(is_business_travel_ready) %>%
  summarise(
    number = n()) 


sort(x_tr$maximum_nights %>% unique())
summary(x_tr$maximum_nights)
max_nit <- x_tr %>%
  group_by(maximum_nights) %>%
  summarise(count = n())
barplot(height = max_nit$count, names.arg = max_nit$maximum_nights)

sort(x_tr$minimum_nights %>% unique())
summary(x_tr$minimum_nights)
boxplot(x_tr$minimum_nights)
max_nit <- x_tr %>%
  group_by(minimum_nights) %>%
  summarise(count = n())
barplot(height = max_nit$count, names.arg = max_nit$minimum_nights)


summary(x_tr$country %>% as.factor())
summary(x_tr$country_code %>% as.factor())
summary(x_va$country %>% as.factor())
summary(x_tr_rf$cancellation_policy)
summary(x_tr$square_feet)
histogram(x_tr$square_feet)
mean(x_tr$square_feet)
boxplot(x_tr$square_feet)
sf <- x_tr$square_feet
sf <- ifelse(sf > 1500, 1500, sf)
# sf <- ifelse(sf <= 1, log(median(sf)), log(sf))
boxplot(sf)

names(x_tr)
summary(x_tr$host_total_listings_count)

histogram(x_tr$host_total_listings_count, nint = max(x_tr$host_total_listings_count) / 8)
boxplot(x_tr$host_listings_count)


listing_freq <- x_tr %>% 
  group_by(host_listings_count) %>%
  summarise(total = n())

barplot(height = listing_freq$total, names.arg = listing_freq$host_listings_count)




