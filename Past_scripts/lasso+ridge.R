# This line references the data_cleaning.r script so that you can call functions that are written in that r file.
# Libraries called in the referenced file will automatically be included in this file.
source('Library//data_cleaning.r')
source('Library//utils.r')
library(xgboost)
library(caret)
library(glmnet)
options(scipen = 999)

# replace the string with the directory of the project folder "Data"
# the original datasets MUST be placed under the Data folder for the following script to run correctly.
folder_dir = r"(/Users/nuomihan/Desktop/758T group project/airbnb/Data)"

# read
# x_train <- read.csv('Data\\x_train_clean.csv')
# x_test <- read.csv('Data\\x_test_clean.csv')
x <- get_cleaned(folder_dir, FALSE)
y_train <- read.csv('Data//airbnb_train_y_2023.csv')
hbr <- y_train$high_booking_rate %>% as.factor()
prs <- y_train$perfect_rating_score %>% as.factor()
x_train <- x[1:nrow(y_train),]
x_test <- x[(nrow(y_train) + 1): nrow(x),]
# test
nrow(x_train) == nrow(y_train)
nrow(x_test) == 12205
(length(hbr) == length(prs)) && (length(hbr) == nrow(y_train))

# train-validation split
sampled = sample(1:nrow(x_train), 0.7 * nrow(x_train))
x_tr = x_train[sampled, ]
x_va = x_train[-sampled, ]
hbr_tr = hbr[sampled]
hbr <- y_train$high_booking_rate %>% as.factor()
hbr <- ifelse(hbr == 'YES', 1, 0)
prs <- y_train$perfect_rating_score %>% as.factor()
prs <- ifelse(prs == 'YES', 1, 0)
############################data cleaning ###########################
cleaning_test <- function(df){
  df <- df %>%
    mutate(bed_category = as.factor(bed_category),
           property_category = as.factor(property_category),
           cancellation_policy = as.factor(cancellation_policy),
           host_response_time = as.factor(ifelse(host_response_time == "character","MISSING", host_response_time)),
           price_per_person = price/accommodates,
           ppp_ind = ifelse(price_per_person > median(price_per_person),1 , 0),
           room_type = as.factor(room_type),
           host_acceptance = as.factor(host_acceptance),
           host_response = as.factor(host_response),
           license = ifelse(license == 'Missing',0,1),
           price = log(price))
  df<- df%>%
    mutate(time_since_first_review= difftime('2023-01-01',first_review, unit='days'),
           time_since_first_review=as.duration(time_since_first_review)/dyears(x=1),
           time_host_since= difftime('2023-01-01',host_since, unit='days'),
           time_host_since=as.duration(time_host_since)/dyears(x=1),
           host_local= ifelse(as.character(host_neighbourhood)==as.character(neighbourhood), TRUE, FALSE)
    )
  df <- df %>%
    group_by(market) %>%
    mutate(market_freq = n()) %>%
    ungroup() %>%
    mutate(market = ifelse(market_freq >= quantile(market_freq,0.25), as.character(market), 'Other'),
           market = as.factor(market))
  df <- df %>%
    group_by(city_name) %>%
    mutate(city_freq = n()) %>%
    ungroup() %>%
    mutate(city_name = ifelse(city_freq >=quantile (city_freq,0.25), as.character(city_name), 'Other'),
           city_name=as.factor(city_name))
  df$city <- NULL
  df$host_acceptance_rate <- NULL
  df$host_response_rate <-NULL
  df$country <-NULL
  df$country_code <-NULL
  df$first_review <- NULL
  df$host_location <- NULL
  df$host_neighbourhood <-NULL
  df$zipcode <- NULL
  df$state <- NULL
  df$X <-NULL
  df$smart_location <- NULL
  df$experiences_offered <- NULL
  df$neighbourhood <-NULL
  df$property_type <- NULL
  df$bed_type <- NULL
  df$access<-NULL
  df$host_about <-NULL
  df$house_rules <- NULL
  df$interaction <- NULL
  df$jurisdiction_names <- NULL
  df$name <- NULL
  df$neighborhood_overview <-NULL
  df$notes<-NULL
  df$space <-NULL
  df$street <- NULL
  df$summary <- NULL
  df$transit <- NULL
  df$host_name <-NULL
  df$host_verifications <-NULL
  df$description <-NULL
  for (i in length(df)){
    if (max(df[i]) == 1 & min(df[i] == 0)){
      df[i] = as.factor(df[i])
    }
  }
  return(df)}
x_all <- cleaning_test(x) %>%
  select(!c('amenities', 'latitude', 'longitude'))
tr = x_all[1:nrow(x_train), ]
te = x_all[(nrow(x_train) + 1) : nrow(x_all), ]
#create dummy variables
dummy <- dummyVars(formula=~., data = rbind(tr, te),fullRank = T)
train_dum <- predict(dummy, newdata = tr)
te_dum <- predict(dummy, newdata = te)
#split training set into train and valid
sampled = sample(1:nrow(train_dum), 0.7 * nrow(train_dum))
x_tr = train_dum[sampled, ]
x_va = train_dum[-sampled, ]
hbr_tr = hbr[sampled]
hbr_va = hbr[-sampled]

###################train a xgboost model#####################
xgb <-xgboost(
  data = x_tr,
  label = hbr_tr,
  max.depth = 8,
  eta = 0.06,
  nrounds = 600,
  objective = "binary:logistic",
  eval_metric = "auc",
  verbose = T,
  print_every_n = 100,
  nthread = 12,
  weight = ifelse(hbr_tr == 1, 10, 1)
)

#make predictions
xgb_va<- predict(xgb, newdata=x_va)
xgb_tr <- predict(xgb, newdata=x_tr )

#get auc
xgb_auc = get_auc(xgb_va, hbr_va)
xgb_auc

#########################train a ridge logistic model##################
grid <- 10^seq(-7,7,length=100)

#storage vector
aucs <- rep(0, length(grid))

#grid search
for(i in c(1:length(grid))){
  lam = grid[i] #current value of lambda
  
  #train a ridge model with lambda = lam
  ridge <- glmnet(x_tr, hbr_tr, family = "binomial", alpha = 1, lambda = lam)
  
  #make predictions as usual
  preds <- predict(glmout, newx = ridge, type = "response")
  auc_temp <- get_auc(preds, hbr_va)
  aucs[i] <- auc_temp
}


#grid search
for(i in c(1:length(grid))){
  lam = grid[i] #current value of lambda
  #train a ridge model with lambda = lam
  ridge <- glmnet(x_tr, hbr_tr, family = "binomial", alpha = 1, lambda = lam)
  #make predictions as usual
  preds <- predict(ridge, newx = x_va, type = "response")
  auc_temp <- get_auc(preds, hbr_va)
  aucs[i] <- auc_temp
}


#get best lambda for ridge model
grid[which.max(aucs)]
aucs

#get best model
ridge<- glmnet(x_tr, hbr_tr, family = "binomial", alpha = 1, lambda = 0.0000001)

#make predictions
ridge_tr <- predict(ridge, newx = x_tr, type = "response")

#################################lasso##############################
grid <- 10^seq(-7,7,length=100)

#storage vector
aucs_lasso <- rep(0, length(grid))

#grid search
for(i in c(1:length(grid))){
  lam = grid[i] #current value of lambda
  #train a ridge model with lambda = lam
  lasso <- glmnet(x_tr, hbr_tr, family = "binomial", alpha = 0, lambda = lam)
  #make predictions as usual
  preds <- predict(lasso, newx = x_va, type = "response")
  auc_temp <- get_auc(preds, hbr_va)
  aucs_lasso[i] <- auc_temp
}


#get best lambda for ridge model
grid[which.max(aucs_lasso)]

#get best model
lasso<- glmnet(x_tr, hbr_tr, family = "binomial", alpha =0, lambda = 0.00000097701)

#make predictions
lasso_tr <- predict(lasso, newx = x_tr, type = "response")


########################train a logistic model based on xgb/ridge###############
#logistic regression based on xgb prediction
log_xgb <- glm(data = data.frame(cbind(xgb_tr,hbr_tr)), formula = hbr_tr~xgb_tr, family = "binomial")

#unify the colname of xgb_va and xgb_tr
xgb_va_df = data.frame(xgb_va)
xgb_va_df$xgb_tr <- xgb_va_df$xgb_va
xgb_va_df$xgb_va <- NULL

#make prediction
preds_xgb_va <- predict(log_xgb, newdata = xgb_va_df,type = "response")
#calculate auc
get_auc(preds_xgb_va, hbr_va)

##logistic regression based on ridge prediction
log_ridge <- glm(data = data.frame(cbind(ridge_tr,hbr_tr)), formula = hbr_tr~s0, family = "binomial")

#make prediction
preds_ridge_va <- predict(log_ridge, newdata = data.frame(ridge_va),type = "response")

#calculate auc
get_auc(preds_ridge_va, hbr_va)

#auc of original ridge model
max(aucs)

pred_te <- predict(xgb, newdata = te_dum)
write.table(y_te_pred, "high_booking_rate_group5.csv", row.names = FALSE)
