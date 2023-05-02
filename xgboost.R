# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file.
source('Library//data_cleaning.r')
source('Library//utils.r')
library(xgboost)
library(ranger)

# replace the string with the directory of the project folder "Data"
# the original datasets MUST be placed under the Data folder for the following script to run correctly.
folder_dir = r"(/Users/nuomihan/Desktop/758T group project)"



# This line needs only to be run once, which exports two csv files, one for the training X's, the other for the testing X's. Once the two files were created you only need to run the read.csv statements under this line.
# export_cleaned(folder_dir)


# read
# x_train <- read.csv('Data\\x_train_clean.csv')
# x_test <- read.csv('Data\\x_test_clean.csv')
x <- get_cleaned(folder_dir)
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


length(x_test[,1])

accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}
summary(x_train)

#data cleaning
summary(x_test)


cleaning_test <- function(df){
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
  df <- df %>%
    mutate(bed_category = as.factor(bed_category),
           property_category = as.factor(property_category),
           cancellation_policy = as.factor(cancellation_policy),
           host_response_time = as.factor(ifelse(host_response_time == "character","MISSING", host_response_time)),
           host_since = year(host_since),
           price_per_person = price/accommodates,
           ppp_ind = ifelse(price_per_person > median(price_per_person),1 , 0),
           room_type = as.factor(room_type),
           host_acceptance = as.factor(host_acceptance),
           host_response = as.factor(host_response),
           license = ifelse(license == 'Missing',0,1))
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
  
  for (i in length(df)){
    if (max(df[i]) == 1 & min(df[i] == 0)){
      df[i] = as.factor(df[i])
    }
  }
  return(df)}

te <-cleaning_test(x_test)
tr <- cleaning_test(x_train)
sapply(tr, is.character)
tr_matrix <- as.matrix(tr)
te_matrix <-as.matrix(te)
tr1<- tr[1:38,]
#create dummy teriables
dummy_tr <- dummyVars(formula=~., data = tr,fullRank = TRUE)

dummy_te <- dummyVars(formula=~., data = te,fullRank = TRUE) 
tr <- predict(dummy_tr, newdata = tr)
te <- predict(dummy_te, newdata = te)

tr<- read.csv('x_train_dummy.csv')
te<- read.csv('x_test_dummy.csv')
y_cleaned <- read.csv('cleaned_y_train.csv')
y_train_prs <- y_cleaned[,2]
y_train_hbr <- y_cleaned[,3]
#############hbr##########################
#split train and validation
train_insts = sample(nrow(tr), .7*nrow(tr))
data_train <- tr[train_insts,]
data_valid <- tr[-train_insts,]
y_train_hbr_1 <- ifelse(hbr == 'YES',1,0)
y_train <- y_train_hbr_1[train_insts]
y_valid <- y_train_hbr_1[-train_insts]

#build model for hbr
#tree_depth = c(5,10,15,25,35,50,75)
tree_depth = c(1:10)
tr_auc = rep(0,length(tree_depth))
va_auc = rep(0,length(tree_depth))
#hbr_xgboost <- xgboost(data = data_train, label = y_train, nround = 50,objective = "reg:logistic")
for (i in 1:length(tree_depth)){
  hbr_xgboost <- xgboost(data = data_train, label = y_train, nround = 150, max_depth = tree_depth[i],objective = "reg:logistic")
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  #calculate auc
  auc_train <- get_auc( y_train_pred,y_train)
  tr_auc[i] = auc_train
  y_va_pred <- predict(hbr_xgboost, newdata = data_valid)
#  classification_va <- ifelse(y_va_pred > .5, 1, 0)
  auc_valid <- get_auc(y_va_pred, y_valid)
  va_auc[i] = auc_valid}
#get the max_depth with highest va acc
tr_auc
va_auc
tree_depth[which.max(va_auc)]
#auc = 0.8933235,round = 150
###################set the nround#######################
round = c(5,10,20,30,40,50,60,70,80,90,100,125,150,200)
tr_auc = rep(0,length(round))
va_auc = rep(0,length(round))
for (i in 1:length(round)){
  hbr_xgboost <- xgboost(data = data_train, label = y_train, nround = round[i], max_depth = 6,objective = "reg:logistic")
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  #calculate auc
  auc_train <- get_auc( y_train_pred,y_train)
  tr_auc[i] = auc_train
  y_va_pred <- predict(hbr_xgboost, newdata = data_valid)
  #  classification_va <- ifelse(y_va_pred > .5, 1, 0)
  auc_valid <- get_auc(y_va_pred, y_valid)
  va_auc[i] = auc_valid}
get_auc(y_va_pred,y_valid)
#get the max_depth with highest va acc
tr_auc
va_auc
round[which.max(va_auc)]
max(va_auc)


#############change model evaluation metrics#############
hbr_xgboost <- xgboost(data = data_train, label = y_train, nround = round[i], max_depth = 9,objective = "reg:logistic", eval_metric = "auc")


###############choose objectives##############
obj = c( 'binary:logistic','reg:linear')

tr_auc = rep(0,length(obj))
va_auc = rep(0,length(obj))
for (i in 1:length(obj)){
  hbr_xgboost <- xgboost(data = data_train, label = y_train, nround = 150, max_depth = 6,objective = obj[i])
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  #calculate auc
  auc_train <- get_auc( y_train_pred,y_train)
  tr_auc[i] = auc_train
  y_va_pred <- predict(hbr_xgboost, newdata = data_valid)
  #  classification_va <- ifelse(y_va_pred > .5, 1, 0)
  auc_valid <- get_auc(y_va_pred, y_valid)
  va_auc[i] = auc_valid}

va_auc
###################set eta#################
eta_set = c(0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5)
tr_auc_eta = rep(0,length(eta_set))
va_auc_eta = rep(0,length(eta_set))
for (i in 1:length(eta_set)){
  hbr_xgboost <- xgboost(data = data_train, label = y_train, nround = 150, max_depth = 6,objective = "reg:logistic", eval_metric = "auc",eta = eta_set[i])
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  #calculate auc
  auc_train <- get_auc( y_train_pred,y_train)
  tr_auc_eta[i] = auc_train
  y_va_pred <- predict(hbr_xgboost, newdata = data_valid)
  #  classification_va <- ifelse(y_va_pred > .5, 1, 0)
  auc_valid <- get_auc(y_va_pred, y_valid)
  va_auc_eta[i] = auc_valid}

va_auc_eta # max = 0.8947089
eta_set[which.max(va_auc_eta)] #eta=0.2

#####################tuning gamma####################
gm <- c(0,1,2,4,6,8,10,20)
tr_auc_gm = rep(0,length(gm))
va_auc_gm = rep(0,length(gm))
for (i in 1:length(gm)){
  hbr_xgboost <- xgboost(data = data_train, label = y_train, nround = 150, max_depth = 6,objective = "reg:logistic", eval_metric = "auc",eta = 0.2,gamma = gm[i])
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  #calculate auc
  auc_train <- get_auc( y_train_pred,y_train)
  tr_auc_gm[i] = auc_train
  y_va_pred <- predict(hbr_xgboost, newdata = data_valid)
  #  classification_va <- ifelse(y_va_pred > .5, 1, 0)
  auc_valid <- get_auc(y_va_pred, y_valid)
  va_auc_gm[i] = auc_valid}
va_auc_gm # max = 0.8951530
gm[which.max(va_auc_gm)] #gm=2
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

hbr_xgboost <- xgboost(data = tr_match, label = y_train_hbr, nround = 150, max_depth = 9,objective = "reg:logistic")
y_te_pred <- predict(hbr_xgboost, newdata = te_match)


length(te_match[,1])

max(y_te_pred)

model <- xgb.dump(hbr_xgboost, with.stats = T)
#compute feature importance matrix for selecting variables 
importance_matrix <- xgb.importance(model = hbr_xgboost)

write.table(importance_matrix, "importance_matrix.csv")

write.table(y_te_pred, "high_booking_rate_group5.csv", row.names = FALSE)

########################prs########################
#split train and validation
train_insts = sample(nrow(tr), .7*nrow(tr))
data_train <- tr[train_insts,]
data_valid <- tr[-train_insts,]
y_train_prs_1 <- as.numeric(y_train_prs)-1
y_train <- y_train_prs_1[train_insts]
y_valid <- y_train_prs_1[-train_insts]
y_train_hbr
#build model for hbr
tree_depth = c(1:10,15,25,35,50)
tr_accs = rep(0,length(tree_depth))
va_accs = rep(0,length(tree_depth))
for (i in 1:length(tree_depth)){
  hbr_xgboost <- xgboost(data = data_train, label = y_train, nround = 25, max_depth = tree_depth[i],objective = "reg:logistic")
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  classification_train <- ifelse(y_train_pred > .5, 1, 0)
  #calculate accuracy
  accuracy_train <- accuracy(classification_train, y_train)
  tr_accs[i] = accuracy_train
  y_va_pred <- predict(hbr_xgboost, newdata = data_valid)
  classification_va <- ifelse(y_va_pred > .5, 1, 0)
  accuracy_va <- accuracy(classification_va, y_valid)
  va_accs[i] = accuracy_va}
#get the max_depth with highest va acc
tr_accs
va_accs
tree_depth[which.max(va_accs)]

###################set the nround#######################
round = c(5,10,20,30,40,50,60,70,80,90,100,125,150,200)
tr_accs = rep(0,length(round))
va_accs = rep(0,length(round))
for (i in 1:length(round)){
  hbr_xgboost <- xgboost(data = data_train, label = y_train, nround = round[i], max_depth = 9,objective = "reg:logistic")
  #make prediction
  y_train_pred <- predict(hbr_xgboost, newdata = data_train)
  classification_train <- ifelse(y_train_pred > .5, 1, 0)
  #calculate accuracy
  accuracy_train <- accuracy(classification_train, y_train)
  tr_accs[i] = accuracy_train
  
  y_va_pred <- predict(hbr_xgboost, newdata = data_valid)
  classification_va <- ifelse(y_va_pred > .5, 1, 0)
  accuracy_va <- accuracy(classification_va, y_valid)
  va_accs[i] = accuracy_va}

#get the max_depth with highest va acc
tr_accs
va_accs
round[which.max(va_accs)] #20
max(va_accs) #0.7356471


