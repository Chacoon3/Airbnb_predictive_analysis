# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file
source('Library\\data_cleaning.r')
source('Library\\utils.r')
library(xgboost)
library(ranger)

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
      extra_people, # added 2023-4-16
      first_review, # added 2023-4-16
      guests_included,
      host_acceptance_rate,
      host_has_profile_pic,
      host_identity_verified,
      host_is_superhost, # added 2023-4-16
      host_listings_count, # added 2023-4-16
      host_response_rate,
      host_response_time,
      host_since,
      Internet, # added 2023-4-16
      TV, # added 2023-4-16
      
      longitude,
      latitude,
      
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
      cancellation_policy = as.factor(x$cancellation_policy),
      host_response_time = as.factor(x$host_response_time),

      price = ifelse(price < 1, 1, price), # this is to prevent -inf
      price = log(price)
    )
  return(res)
}


x_tr_rf <- feature_engineering(x_tr) 
x_va_rf <- feature_engineering(x_va)
md_dummy <- dummyVars(formula = ~., x_tr_rf, fullRank = TRUE)
x_tr_rf_dummy <- predict(md_dummy, x_tr_rf)
x_va_rf_dummy <- predict(md_dummy, x_va_rf)
prs_tr_dummy = ifelse(prs_tr == 'YES', 1, 0)
hbr_tr_dummy = ifelse(hbr_tr == 'TES', 1, 0)


# xgb ----------------------------
md_prs_xgb <- xgboost(
  data = as.matrix(x_tr_rf_dummy),
  label = prs_tr,
  nrounds = 1700
)


y_pred_prob_xgb <- predict(md_prs_xgb, newdata = x_va_rf_dummy)
plot_roc(y_pred_prob_xgb, prs_va)
get_auc(y_pred_prob_xgb, prs_va)


# rf ----------------------------
md_prs_rf <- randomForest::randomForest(
  x = x_tr_rf_dummy, 
  y = prs_tr, 
  maxnodes = 80,
  nodesize = 1,
  mtry = 20
)

y_pred_rf <- predict(md_prs_rf, newdata = x_va_rf_dummy, 'prob')

plot_roc(y_pred_rf[,2], prs_va)
get_auc(y_pred_rf[,2], prs_va)



# ranger ---------------------------------
md_hbr_rf_ranger <- ranger(x = x_tr_rf_dummy, y = hbr_tr,
                 mtry=22, num.trees=500,
                 importance="impurity",
                 probability = TRUE)
y_pred_prob_hbr_ranger <- 
  predict(md_hbr_rf_ranger, data = x_va_rf_dummy)$predictions[,2]
plot_roc(y_pred_prob_hbr_ranger, hbr_va)
get_auc(y_pred_prob_hbr_ranger, hbr_va)


# logistic --------------------------
md_logistic <- glm(data.frame(x_tr_rf_dummy, hbr_tr), formula = hbr_tr~., family = 'binomial')

y_pred_prob_logit <- predict(md_logistic, newdata = data.frame(x_va_rf_dummy), type = 'response')

plot_roc(y_pred_prob_logit, hbr_va)




# analysis ------------------------

t_diff <- x_tr_rf$host_since - x_tr_rf$first_review
mean(t_diff)
summary(x_tr_rf$host_since)
histogram(x_tr_rf$host_since - x_tr_rf$first_review)


names(x_tr)
summary(x_tr$host_listings_count)

histogram(x_tr$host_listings_count, nint = max(x_tr$host_listings_count) / 8)
boxplot(x_tr$host_listings_count)

listing_freq <- x_tr %>% 
  group_by(host_listings_count) %>%
  summarise(total = n())

barplot(height = listing_freq$total, names.arg = listing_freq$host_listings_count)
