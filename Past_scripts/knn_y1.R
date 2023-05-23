library(caret)
library(pROC)
library(class)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(stats)

data_all <- read.csv("cleaned.csv") %>%
select( c(accommodates,
          bed_type,
          cleaning_fee,
          extra_people,
          guests_included,
          host_acceptance_rate,
          host_has_profile_pic,
          host_identity_verified,
          host_is_superhost,
          host_listings_count,
          host_response_rate,
          instant_bookable,
          is_business_travel_ready,
          latitude,
          longitude,
          market,
          monthly_price,
          property_category,
          room_type,
          security_deposit,
          square_feet,
          weekly_price,
          high_booking_rate,
          perfect_rating_score)) %>%
mutate(
  bed_type = as.factor(bed_type),
  market = as.factor(market),
  room_type = as.factor(room_type),
  property_category = as.factor(property_category),
  high_booking_rate = as.factor(high_booking_rate),
  perfect_rating_score = as.factor(perfect_rating_score))
  

###
data_all <- read.csv("cleaned.csv") %>%
  select_if(function(x) length(unique(x)) > 1) %>%
  select(-c(city,host_verifications)) %>%
  mutate(
    bed_type = as.factor(bed_type),
         cancellation_policy = as.factor(cancellation_policy),
         city_name = as.factor(city_name),
         country = as.factor(country),
         country_code = as.factor(country_code),
         first_review = as.factor(first_review),
         host_location = as.factor(host_location),
         host_neighbourhood = as.factor(host_neighbourhood),
         host_response_time = as.factor(host_response_time),
         host_since = as.factor(host_since),
         market = as.factor(market),
         neighbourhood = as.factor(neighbourhood),
         property_type = as.factor(property_type),
         room_type = as.factor(room_type),
         smart_location = as.factor(smart_location),
         state = as.factor(state),
         zipcode = as.factor(zipcode),
         bed_category = as.factor(bed_category),
         property_category = as.factor(property_category),
         host_acceptance = as.factor(host_acceptance),
         host_response = as.factor(host_response),
         high_booking_rate = as.factor(high_booking_rate),
         perfect_rating_score = as.factor(perfect_rating_score))
    
##
data_y1 <- data_all$high_booking_rate

y_train_hbr <- read.csv("cleaned_y_train_hbr.csv")


#split to a small data-set
x_train_small <- data_all %>% sample_n(1000)
data_y1 <- x_train_small$high_booking_rate
data_y2 <- x_train_small$perfect_rating_score
dummy <- dummyVars( ~ . , data=x_train_small, fullRank = TRUE)
x_train_dummy <- data.frame(predict(dummy, newdata = x_train_small)) 

#set train and valid model
set.seed(1)
valid_instn = sample(nrow(x_train_dummy), 0.30*nrow(x_train_dummy))
train_X <- x_train_dummy[-valid_instn,]
valid_X <- x_train_dummy[valid_instn,]

train_y1 <- data_y1[-valid_instn]
valid_y1 <- data_y1[valid_instn]

#accuracy function
accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

#Make predictions
knn.pred=knn(train_X, valid_X, train_y1, k=20, prob = TRUE)
knn.pred
#prediction performance
table(valid_y1,knn.pred)
accuracy(knn.pred, valid_y1)
knn.probs <- attr(knn.pred, "prob")
knn.probs
knn_probofYES <- ifelse(knn.pred == "YES", knn.probs, 1-knn.probs)
knn_probofYES
#fitting curve: k vs. accuracy
kvec <- c(1:20)*2 - 1 
kvec
#initialize storage
va_acc <- rep(0, length(kvec))
tr_acc <- rep(0, length(kvec))
#for loop
for(i in 1:length(kvec)){
  inner_k <- kvec[i]
  
  inner_tr_preds <- knn(train_X, train_X, train_y1, k=inner_k, prob = TRUE) 
  inner_tr_acc <- accuracy(inner_tr_preds, train_y1)
  tr_acc[i] <- inner_tr_acc
  
  #repeat for predictions in the validation data
  inner_va_preds <- knn(train_X, valid_X, train_y1, k=inner_k, prob = TRUE) 
  inner_va_acc <- accuracy(inner_va_preds, valid_y1)
  va_acc[i] <- inner_va_acc}

plot(kvec, tr_acc, col = "blue", type = 'l', ylim = c(.6,1))
lines(kvec, va_acc, col = "red")
#the best k
best_validation_index <- which.max(va_acc)
best_k <- kvec[best_validation_index] 
#best k and retrieve the probability that Y=1
best_k_preds <- knn(train_X, valid_X, train_y1, best_k, prob=TRUE)
best_probs <- attr(best_k_preds, "prob")
best_probofYES <- ifelse(best_k_preds == "YES", best_probs, 1-best_probs)
#given a cutoff assess accuracy, TPR, and TNR
classify_evaluate <- function(predicted_probs, actual_y, cutoff){
  
  classifications <- ifelse(predicted_probs > cutoff, "YES", "NO")
  classifications <- factor(classifications, levels = levels(actual_y))
  
  CM <- confusionMatrix(data = classifications,
                        reference = actual_y,
                        positive = "YES")
  
  CM_accuracy <- as.numeric(CM$overall["Accuracy"])
  CM_TPR <- as.numeric(CM$byClass["Sensitivity"])
  CM_TNR <- as.numeric(CM$byClass["Specificity"])
  
  return(c(CM_accuracy, CM_TPR, CM_TNR))}

#performance for each cutoff
classify_evaluate(best_probofYES, valid_y1, .25)
classify_evaluate(best_probofYES, valid_y1, .5)
classify_evaluate(best_probofYES, valid_y1, .75)

train_pred <- knn(train_X, train_X, train_y1, k = best_k, prob = TRUE)
train_acc <- accuracy(train_pred, train_y1)

valid_pred <- knn(train_X, valid_X, train_y1, k = best_k, prob = TRUE)
valid_acc <- accuracy(valid_pred, valid_y1)

print(paste("Training Accuracy:", train_acc))
print(paste("Generalization Accuracy:", valid_acc))

#AUC
roc_obj <- roc(valid_y1, best_probofYES)
auc <- auc(roc_obj)
plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate", col = "blue")
print(auc)

# Fitting Curve
plot(kvec, va_acc, type = 'l', col = "red", xlab = "k", ylab = "Accuracy", main = "Fitting Curve")
lines(kvec, tr_acc, type = 'l', col = "blue")
legend("bottomright", legend = c("Validation Accuracy", "Training Accuracy"), col = c("red", "blue"), lty = 1)
