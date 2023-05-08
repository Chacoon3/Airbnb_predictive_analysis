# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file.
source('Library//data_cleaning.r')
source('Library//utils.r')
library(xgboost)
library(ranger)
library(lubridate)
library(glmnet)

# replace the string with the directory of the project folder "Data"
# the original datasets MUST be placed under the Data folder for the following script to run correctly.
folder_dir = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data)"



# This line needs only to be run once, which exports two csv files, one for the training X's, the other for the testing X's. Once the two files were created you only need to run the read.csv statements under this line.
# export_cleaned(folder_dir)


# read
# x_train <- read.csv('Data\\x_train_clean.csv')
# x_test <- read.csv('Data\\x_test_clean.csv')
x <- get_cleaned(folder_dir, FALSE)
y_train <- read.csv('Data/airbnb_train_y_2023.csv')
hbr <- y_train$high_booking_rate %>% as.factor()
prs <- y_train$perfect_rating_score %>% as.factor()
x_train <- x[1:nrow(y_train),]
x_test <- x[(nrow(y_train) + 1): nrow(x),]


# test
nrow(x_train) == nrow(y_train)
nrow(x_test) == 12205
(length(hbr) == length(prs)) && (length(hbr) == nrow(y_train))

# train-validation split
# sampled = sample(1:nrow(x_train), 0.75 * nrow(x_train))
# x_tr = x_train[sampled, ]
# x_va = x_train[-sampled, ]
# hbr_tr = hbr[sampled]
# hbr_va = hbr[-sampled]
# prs_tr = prs[sampled]
# prs_va = prs[-sampled]
# 

# codes start here ----------------------------


# purely amenities based --------------
# x_am_train = x_train[, 70:107]
# 
# md_dummy_ridge = dummyVars(~., x_am_train)
# x_am_dummy_train = predict(md_dummy_ridge, x_am_train)
# 
# 
# md_ridge = glmnet(
#   x = x_am_dummy_tr, y = hbr_tr, alpha = 1, family = 'binomial',
#   lambda = 10^-7
#   )
# pred_hbr_ridge = predict(md_ridge, newx = x_am_dummy_va, type = 'response')
# get_auc(pred_hbr_ridge, hbr_va)
# 0.6932712

# data cleaning ------------------

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
           license = ifelse(license == 'Missing',0,1))
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


#check if there any character col
sum(sapply(tr, is.character))

colnames(tr)

#create dummy variables
dummy <- dummyVars(formula=~., data = rbind(tr, te),fullRank = T)
train_dum <- predict(dummy, newdata = tr)
te_dum <- predict(dummy, newdata = te)

#convert hbr into 1-0 variable
y_train_hbr_1 <- ifelse(hbr == 'YES',1,0)
prs <- ifelse(prs == 'YES',1,0)

sampled = sample(1:nrow(train_dum), 0.75 * nrow(train_dum))
x_tr = train_dum[sampled, ]
x_va = train_dum[-sampled, ]
hbr_tr = y_train_hbr_1[sampled]
hbr_va = y_train_hbr_1[-sampled]
prs_tr = prs[sampled]
prs_va = prs[-sampled]


# grid search ---------------------
#      depth nround eta_set       auc
# 6      5    600     0.2 0.9059210
# 14     6    500     0.2 0.9057336
# 44     9    800     0.2 0.9057282
cs_res <- cube_search(
  x = rbind(x_tr, x_va),
  y = c(hbr_tr, hbr_va), # always remember use c to contacnate two vectors
  vec_param1 = 4:8, # depth
  vec_param2 = 3:7 * 100, # round
  vec_param3 = 1:3 * 5 / 100, # eta
  trainer = \(x,y,p1,p2,p3) {
    model <- xgboost(
      data = x,
      label = y,
      max.depth = p1,
      nrounds = p2,
      eta = p3, 
      objective = "binary:logistic",
      eval_metric = "auc", 
      verbose = T,
      print_every_n = 100,
      nthread = 12,
      weight = ifelse(y == 1, 10, 1)
    )
    return(model)
  },
  predictor = \(m, x) {
    pred <- predict(m, newdata = x)
    return(pred)
  },
  measurer = \(y1, y2) {
    return(get_auc(y1, y2))
  },
  n_per_round = 1
)
cs_res %>% arrange(measurement %>% desc())
#       param1 param2 param3 measurement
# 1       8    600   0.10   0.9030510
# 2       8    700   0.10   0.9028660
# 3       8    700   0.05   0.9027338
# 4       8    500   0.10   0.9027207


vs_res <- vector_search(
  x = rbind(x_tr, x_va),
  y = c(hbr_tr, hbr_va), # always remember use c to contacnate two vectors
  vec_param1 = 1:10 * 2 / 100, # eta
  trainer = \(x,y,p1) {
    model <- xgboost(
      data = x,
      label = y,
      max.depth = 8,
      nrounds = 600,
      eta = p1, 
      objective = "binary:logistic",
      eval_metric = "auc", 
      verbose = T,
      print_every_n = 100,
      nthread = 12,
      weight = ifelse(y == 1, 10, 1)
    )
    return(model)
  },
  predictor = \(m, x) {
    pred <- predict(m, newdata = x)
    return(pred)
  },
  measurer = \(y1, y2) {
    return(get_auc(y1, y2))
  },
  n_per_round = 2
)
vs_res %>% arrange(measurement %>% desc())
#     param1 measurement
# 1    0.06   0.9027252
# 2    0.04   0.9020486
# 3    0.12   0.9019057
# 4    0.10   0.9018063
# 5    0.08   0.9017903


vs_res <- vector_search(
  x = rbind(x_tr, x_va),
  y = c(hbr_tr, hbr_va), # always remember use c to contacnate two vectors
  vec_param1 = 2:5 * 3, # weight
  trainer = \(x,y,p1) {
    model <- xgboost(
      data = x,
      label = y,
      max.depth = 8,
      nrounds = 600,
      eta = 0.06, 
      objective = "binary:logistic",
      eval_metric = "auc", 
      verbose = T,
      print_every_n = 100,
      nthread = 12,
      weight = ifelse(y == 1, p1, 1)
    )
    return(model)
  },
  predictor = \(m, x) {
    pred <- predict(m, newdata = x)
    return(pred)
  },
  measurer = \(y1, y2) {
    return(get_auc(y1, y2))
  },
  n_per_round = 1
)
vs_res %>% arrange(measurement %>% desc())
# optimal positive weight 9
#     param1 measurement
# 1      9   0.8981199
# 2      6   0.8981117
# 3     12   0.8967573
# 4     15   0.8965645


# train validation model --------------
best_model <- xgboost(
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
pred <- predict(best_model, newdata = x_va)
plot_roc(pred, hbr_va)
get_auc(pred, hbr_va) 
# 0.8989933 \\ 0.8987783 \\ 0.9003966 \\ 0.9020315
vip(best_model, 35)
summary(best_model)


cn <- colnames(x_tr)
cn


# train final model --------------------------
x_bind = rbind(x_tr, x_va)
hbr_bind = c(hbr_tr, hbr_va)
final_model <- xgboost(
  data = x_bind,
  label = hbr_bind,
  max.depth = 8,
  eta = 0.06, 
  nrounds = 600,
  objective = "binary:logistic",
  eval_metric = "auc", 
  verbose = T,
  print_every_n = 50,
  nthread = 12,
  weight = ifelse(hbr_bind == 1, 10, 1)
)

final_pred = predict(final_model, newdata = te_dum)

write.table(final_pred, "high_booking_rate_group5.csv", row.names = FALSE)

old_pred = read.csv(
  r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\s4_predict.csv)"
)

last_pred = old_pred$x
(final_pred - last_pred) %>% mean()
