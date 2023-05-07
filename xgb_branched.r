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
sampled = sample(1:nrow(x_train), 0.75 * nrow(x_train))
x_tr = x_train[sampled, ]
x_va = x_train[-sampled, ]
hbr_tr = hbr[sampled]
hbr_va = hbr[-sampled]
prs_tr = prs[sampled]
prs_va = prs[-sampled]


# codes start here ----------------------------


# prely amenities based --------------
names(x_va)

x_am_tr = x_tr[, 70:107]
x_am_va = x_va[, 70:107]

md_dummy_ridge = dummyVars(~., x_am_tr)
x_am_dummy_tr = predict(md_dummy_ridge, x_tr)
x_am_dummy_va = predict(md_dummy_ridge, x_va)

md_ridge = glmnet(
  x = x_am_dummy_tr, y = hbr_tr, alpha = 1, family = 'binomial',
  lambda = 10^-7
  )
pred_hbr_ridge = predict(md_ridge, newx = x_am_dummy_va, type = 'response')
get_auc(pred_hbr_ridge, hbr_va)
# 0.6932712

#data cleaning
#To build the input matrix of xgboost, I need to remove the character columns
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

te <-cleaning_test(x_test)

tr <- cleaning_test(x_train)
#check if there any character col
sapply(tr, is.character)

colnames(tr)

#create dummy variables
dummy_tr <- dummyVars(formula=~., data = tr,fullRank = TRUE)
dummy_te <- dummyVars(formula=~., data = te,fullRank = TRUE) 
tr <- predict(dummy_tr, newdata = tr)
te <- predict(dummy_te, newdata = te)

#convert hbr into 1-0 variable
y_train_hbr_1 <- ifelse(hbr == 'YES',1,0)

############################nested loop for tuning parameter###############


k=5
fold_auc_df = data.frame(depth = rep(0,k),
                         nround = rep(0,k),
                         eta_set= rep(0,k),
                         auc = rep(0,k))
grid_search_res = NULL
for(i in 1:k){
  folds <- cut(seq(1,nrow(tr)),breaks=k,labels=FALSE)
  #Segment your data by fold using the which() function 
  valid_inds <- which(folds==i,arr.ind=TRUE)
  valid_fold <- tr[valid_inds, ]
  hbr_valid_fold <- y_train_hbr_1[valid_inds]
  train_fold <- tr[-valid_inds, ]
  hbr_train_fold <- y_train_hbr_1[-valid_inds]
  grid_search_res = grid_search_xgb(
    x_tr = train_fold, 
    y_tr = hbr_train_fold,
    x_va = valid_fold,
    y_va = hbr_valid_fold,
    vec_tree_depth = 5:6,
    vec_nround = 5:7 * 100,
    vec_eta_set = 2:4 /4 * 0.01
  )
  fold_auc_df[i,] = 
    grid_search_res[order(grid_search_res$auc,decreasing = TRUE),][1,]
  break
}

grid_search_res[order(grid_search_res$auc, decreasing = T), ]
#      depth nround eta_set       auc
# 6      5    600     0.2 0.9059210
# 14     6    500     0.2 0.9057336
# 44     9    800     0.2 0.9057282


best_model <- xgboost()

summary(best_model)
vip(best_model)
##################make predictions for test####################
#colnames in test and train are different
#Compare col names and get missing cols
union_names <-union(colnames(tr),colnames(te))
test_missing_cols <-setdiff(union_names,colnames(te))
for (new_col in test_missing_cols){
  a <- rep(0,nrow(te))
  te <- cbind(te, a)
  colnames(te)[ncol(te)] <- new_col
}
idx<-match(union_names,colnames(te))
te_match <-te[,idx]
colnames(te_match)

train_missing_cols <-setdiff(union_names,colnames(tr))
for (new_col in train_missing_cols){
  print(new_col)
  a <- rep(0,nrow(tr))
  tr <- cbind(tr, a)
  colnames(tr)[ncol(tr)] <- new_col
}

colnames(tr)

idx<-match(union_names,colnames(tr))
tr_match <-tr[,idx]

#Check if test and training dataset have the same features
match(colnames(te_match),colnames(tr_match))
hbr_xgboost <- xgboost(data = tr_match, label = y_train_hbr_1, nround = 150, max_depth = 6,objective = "reg:logistic", eval_metric = "auc")
y_te_pred <- predict(hbr_xgboost, newdata = te_match)



length(te_match[,1])

max(y_te_pred)

model <- xgb.dump(hbr_xgboost, with.stats = T)
#compute feature importance matrix for selecting variables 
importance_matrix <- xgb.importance(model = hbr_xgboost)
nrow(importance_matrix)
write.table(importance_matrix, "importance_matrix.csv")

write.table(y_te_pred, "high_booking_rate_group5.csv", row.names = FALSE)

