# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file
source('Library\\data_cleaning.r')
source('Library\\utils.r')
library(xgboost)
library(ranger)
library(caret)
options(scipen = 999)
# replace the string with the directory of the project folder "Data"
# the original datasets MUST be placed under the Data folder for the following script to run correctly.
folder_dir = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data)"

# This line needs only to be run once, which exports two csv files, one for the training X's, the other for the testing X's. Once the two files were created you only need to run the read.csv statements under this line.
# export_cleaned(folder_dir)


# read
# x_train <- read.csv('Data\\x_train_clean.csv')
# x_test <- read.csv('Data\\x_test_clean.csv')
x <- get_cleaned(folder_dir)
y_train <- read.csv('Data\\airbnb_train_y_2023.csv')

hbr <- y_train$high_booking_rate %>% as.factor()
prs <- y_train$perfect_rating_score %>% as.factor()
x_train <- x[1:nrow(y_train),]
x_test <- x[(nrow(y_train) + 1): nrow(x),]


# test
nrow(x_train) == nrow(y_train)
nrow(x_test) == 12205
(length(hbr) == length(prs)) && (length(hbr) == nrow(y_train))


#view
summary(x_train)


# train-validation split
sampled = sample(1:nrow(x_train), 0.85 * nrow(x_train))
x_tr = x_train[sampled, ]
x_va = x_train[-sampled, ]
hbr_tr = hbr[sampled]
hbr_va = hbr[-sampled]
prs_tr = prs[sampled]
prs_va = prs[-sampled]


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
      cancellation_policy,
      cleaning_fee,
      
      guests_included,
      host_acceptance_rate,
      host_has_profile_pic,
      host_identity_verified,
      host_response_rate,
      host_response_time,
      longitude,
      latitude,
      
      
      # added 2023-4-16

      extra_people,
      first_review,
      host_is_superhost,
      host_listings_count,
      host_since,
      instant_bookable,
      # Internet,
      is_business_travel_ready,
      # TV,
      monthly_price,
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
      # neighbourhood,
      
      
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
      host_response_time = as.factor(x$host_response_time),

      price = ifelse(price < 1, mean(price), price) %>% # this is to prevent -inf
              log(), 

      price_per_sqfeet = x$price / 
        ifelse(x$square_feet == 0, median(x$square_feet), x$square_feet),
      
      monthly_price = 
        ifelse(monthly_price < 1, mean(monthly_price), monthly_price) %>%
        log(),
      
      square_feet = 
        ifelse(x$square_feet == 0, median(x$square_feet), x$square_feet),
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
  objective = 'binary:logistic'
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
md_hbr_rf_ranger <- ranger(x = x_tr_rf_dummy, y = prs_tr_rf,
                           # mtry=22, 
                           # num.trees=500,
                           importance="impurity",
                           probability = TRUE)
y_pred_prob_prs_ranger <- 
  predict(md_hbr_rf_ranger, data = x_va_rf_dummy)$predictions[,2]
plot_roc(y_pred_prob_prs_ranger, prs_va)
get_auc(y_pred_prob_prs_ranger, prs_va)
# 0.8023 \\ 0.8003  --> tpr ~ 0.4

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


# single parameter tuning flow ------------------------
vec_param = 1:5 * 50
vec_tr_aucs = rep(0, length(vec_param))
vec_va_aucs = rep(0, length(vec_param))
vec_tr_accs = rep(0, length(vec_param))
vec_va_accs = rep(0, length(vec_param))
for (ind in 1:length(vec_param)) {
  param = vec_param[ind]
  
  md <- ranger(x = x_tr_rf_dummy, 
               y = prs_tr_rf, 
               # mtry = 22, 
               num.trees = param,
               importance="impurity",
               probability = TRUE)
  
  y_va_pred_prob <- 
    predict(md, data = x_va_rf_dummy)$predictions[,2]
  
  y_tr_pred_prob <-
    predict(md, data = x_tr_rf_dummy)$predictions[,2]
  
  va_auc = get_auc(y_va_pred_prob, prs_va)
  tr_auc = get_auc(y_tr_pred_prob, prs_tr)
  
  vec_va_aucs[ind] = va_auc
  vec_tr_aucs[ind] = tr_auc
  
  y_va_cls <- ifelse(y_va_pred_prob >= 0.5, 'YES', 'NO') %>%
    get_accuracy(prs_va)
  y_tr_cls <- ifelse(y_tr_pred_prob >= 0.5, 'YES', 'NO') %>%
    get_accuracy(prs_tr)
  
  
  vec_va_accs[ind] = y_va_cls
  vec_tr_accs[ind] = y_tr_cls
}


df_plot <- data.frame(
  vec_param = vec_param,
  vec_va_accs = vec_va_accs,
  vec_tr_accs = vec_tr_accs,
  vec_va_aucs = vec_va_aucs,
  vec_tr_aucs = vec_tr_aucs
)

ggplot(data = df_plot, aes(x = vec_param)) +
  # ylim(0, 1) +
  geom_line(aes(y = vec_va_aucs, color = 'red')) +
  geom_line(aes(y = vec_tr_aucs, color = 'blue')) +
  geom_line(aes(y = vec_va_accs, color = 'red')) +
  geom_line(aes(y = vec_tr_accs, color = 'blue'))


# cross val flow
cross_val_n = 5
cv_tr_aucs = rep(0, cross_val_n)
cv_va_aucs = rep(0, cross_val_n)
cv_tr_accs = rep(0, cross_val_n)
cv_va_accs = rep(0, cross_val_n)
for (ind in 1:cross_val_n) {
  # train-validation split
  sampled = sample(1:nrow(x_train), 0.75 * nrow(x_train))
  x_tr = x_train[sampled, ]
  x_va = x_train[-sampled, ]
  hbr_tr = hbr[sampled]
  hbr_va = hbr[-sampled]
  prs_tr = prs[sampled]
  prs_va = prs[-sampled]
  
  
  x_tr_rf <- feature_engineering(x_tr) 
  x_va_rf <- feature_engineering(x_va)
  hbr_tr_rf <- hbr_tr
  prs_tr_rf <- prs_tr
  md_dummy <- dummyVars(formula = ~., x_tr_rf, fullRank = TRUE)
  x_tr_rf_dummy <- predict(md_dummy, x_tr_rf)
  x_va_rf_dummy <- predict(md_dummy, x_va_rf)
  prs_tr_dummy = ifelse(prs_tr_rf == 'YES', 1, 0)
  hbr_tr_dummy = ifelse(hbr_tr_rf == 'TES', 1, 0)
  
  
  md <- ranger(x = x_tr_rf_dummy, 
               y = prs_tr_rf, 
               mtry = 22, 
               num.trees = 500,
               importance="impurity",
               probability = TRUE)
  
  y_va_pred_prob <- 
    predict(md, data = x_va_rf_dummy)$predictions[,2]
  
  y_tr_pred_prob <-
    predict(md, data = x_tr_rf_dummy)$predictions[,2]
  
  va_auc = get_auc(y_va_pred_prob, prs_va)
  tr_auc = get_auc(y_tr_pred_prob, prs_tr)
  
  cv_va_aucs[ind] = va_auc
  cv_tr_aucs[ind] = tr_auc
  
  y_va_cls <- ifelse(y_va_pred_prob >= 0.5, 'YES', 'NO') %>%
    get_accuracy(prs_va)
  y_tr_cls <- ifelse(y_tr_pred_prob >= 0.5, 'YES', 'NO') %>%
    get_accuracy(prs_tr)
  
  
  cv_va_accs[ind] = y_va_cls
  cv_tr_accs[ind] = y_tr_cls
}

df_plot2 <- data.frame(
  param = 1:cross_val_n,
  va_accs = cv_va_accs,
  tr_accs = cv_tr_accs,
  va_aucs = cv_va_aucs,
  tr_aucs = cv_tr_aucs
)

ggplot(data = df_plot2, aes(x = param)) +
  # ylim(0, 1) +
  geom_line(aes(y = va_aucs)) +
  geom_line(aes(y = tr_aucs)) +
  geom_line(aes(y = va_accs)) +
  geom_line(aes(y = tr_accs))



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


# draft -------------------------------


# should refine the function by returning the cutoff that maximizes the tpr
# given a maximum fpr
find_cutoff(y_pred_prob_hbr_ranger, hbr_va, level = c('NO', 'YES'), max_fpr = 0.1)



