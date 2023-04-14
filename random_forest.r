# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file.
source('Library\\data_cleaning.r')
source('Library\\utils.r')

library(xgboost)

# replace the string with the directory of the project folder "Data"
# the original datasets MUST be placed under the Data folder for the following script to run correctly.
folder_dir = r"(C:\Users\Chaconne\Documents\学业\UMD\Courses\758T Predictive\785T_Pred_Assignment\GA\Airbnb_predictive_analysis\Data)"

# This line needs only to be run once, which exports two csv files, one for the training X's, the other for the testing X's. Once the two files were created you only need to run the read.csv statements under this line.
export_cleaned(folder_dir)


# read
x_train <- read.csv('Data\\x_train_clean.csv')
x_test <- read.csv('Data\\x_test_clean.csv')
y_train <- read.csv('Data\\airbnb_train_y_2023.csv')
hbr <- y_train$high_booking_rate
prs <- y_train$perfect_rating_score


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
fe_random_forest <- function(x) {
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
      extra_people,
      guests_included,
      host_acceptance_rate,
      host_has_profile_pic,
      host_identity_verified,
      host_response_rate,
      host_response_time,
      
      longitude,
      latitude,
      
      price
    ) %>%
    mutate(
      cancellation_policy = as.factor(x$cancellation_policy),
      host_response_time = as.factor(x$host_response_time),
      
      price = ifelse(price < 1, 1, price), # this is to prevent -inf
      price = log(price)
    )
  return(res)
}


x_tr_rf <- fe_random_forest(x_tr)
x_va_rf <- fe_random_forest(x_va)
md_dummy <- dummyVars(formula = ~., x_tr_rf, fullRank = TRUE)
x_tr_rf_dummy <- predict(md_dummy, x_tr_rf)
x_va_rf_dummy <- predict(md_dummy, x_va_rf)
prs_tr_dummy = ifelse(prs_tr == 'YES', 1, 0)


md_prs_xgb <- xgboost(
  data = as.matrix(x_tr_rf_dummy),
  label = prs_tr_dummy,
  nrounds = 1700
)


y_pred_prob_xgb <- predict(md_prs_xgb, newdata = x_va_rf_dummy)
plot_roc(y_pred_prob_xgb, prs_va)
get_auc(y_pred_prob_xgb, prs_va)


x_tr_rf_dummy %>%
  apply(2, FUN = \(col) {sum(is.na(col))}) %>%
  sum()

md_prs_rf <- randomForest::randomForest(
  x = x_tr_rf_dummy, 
  y = as.factor(prs_tr), 
  maxnodes = 80,
  nodesize = 1,
  mtry = 20
)

y_pred_rf <- predict(md_prs_rf, newdata = x_va_rf_dummy, 'prob')

plot_roc(y_pred_rf[,2], prs_va)
get_auc(y_pred_rf[,2], prs_va)
