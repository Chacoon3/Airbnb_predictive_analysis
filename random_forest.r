# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file
source('Library\\data_cleaning.r')
source('Library\\utils.r')
library(xgboost)
library(ranger)
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


# train-validation split
sampled = sample(1:nrow(x_train), 0.75 * nrow(x_train))
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
      bed_type,
      bedrooms,
      beds,
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
      cancellation_policy,
      extra_people,
      first_review,
      host_is_superhost,
      host_listings_count,
      host_since,
      instant_bookable,
      Internet,
      is_business_travel_ready,
      TV,
      monthly_price,
      room_type,
      
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
hbr_tr_rf <- hbr_tr
prs_tr_rf <- prs_tr
md_dummy <- dummyVars(formula = ~., x_tr_rf, fullRank = TRUE)
x_tr_rf_dummy <- predict(md_dummy, x_tr_rf)
x_va_rf_dummy <- predict(md_dummy, x_va_rf)
prs_tr_dummy = ifelse(prs_tr_rf == 'YES', 1, 0)
hbr_tr_dummy = ifelse(hbr_tr_rf == 'TES', 1, 0)


# xgb ----------------------------
md_prs_xgb <- xgboost(
  data = as.matrix(x_tr_rf_dummy),
  label = prs_tr_rf,
  nrounds = 1700
)


y_pred_prob_xgb <- predict(md_prs_xgb, newdata = x_va_rf_dummy)
plot_roc(y_pred_prob_xgb, prs_va)
get_auc(y_pred_prob_xgb, prs_va)


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
# 0.873 \\ 0.8663719


# ranger prs ----------------------------
md_hbr_rf_ranger <- ranger(x = x_tr_rf_dummy, y = prs_tr_rf,
                           mtry=22, num.trees=500,
                           importance="impurity",
                           probability = TRUE)
y_pred_prob_hbr_ranger <- 
  predict(md_hbr_rf_ranger, data = x_va_rf_dummy)$predictions[,2]
plot_roc(y_pred_prob_hbr_ranger, prs_va)
get_auc(y_pred_prob_hbr_ranger, prs_va)
# 0.8003  --> tpr ~ 0.4


# logistic --------------------------
md_logistic <- glm(data.frame(x_tr_rf_dummy, hbr_tr), formula = hbr_tr~., family = 'binomial')

y_pred_prob_logit <- predict(md_logistic, newdata = data.frame(x_va_rf_dummy), type = 'response')

plot_roc(y_pred_prob_logit, hbr_va)




# analysis ------------------------


summary(x_tr$room_type)
summary(x_tr_rf$price)
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

find_cutoff <- 
  function(y_pred_prob, y_valid_factor, level, max_fpr = 0.085, step = 0.01) {
  if (length(y_pred_prob) != length(y_valid_factor)) {
    stop('prediction and validation have different lengths.')
  }
    
  p = level[2]
  n = level[1]
  
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
  
  
  tpr_prev = 0
  fpr_prev = 0
  cutoff_prev = 0
  tpr = 0
  fpr = 0
  cut_off = 1
  while (fpr < max_fpr && cut_off >= 0 ) {
    cut_off = cut_off - step
    print(cut_off)
    
    y_pred = ifelse(y_pred_prob >= cut_off, level[2], level[1])
    
    tpr = get_tpr(y_pred, y_valid_factor)
    fpr = get_fpr(y_pred, y_valid_factor)
  }
  return(cut_off)
  }


# should refine the function by returning the cutoff that maximizes the tpr
# given a maximum fpr
find_cutoff(y_pred_prob_hbr_ranger, hbr_va, level = c('NO', 'YES'), max_fpr = 0.1)
