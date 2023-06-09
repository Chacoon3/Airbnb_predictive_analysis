# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file.
source('Library\\data_cleaning.r')

# replace the string with the directory of the project folder "Data"
# the original datasets MUST be placed under the Data folder for the following script to run correctly.
folder_dir = r"(C:\Users\quinc\OneDrive - University of Maryland\My Drive\College\5PlusOne\Semester 2\BUDT758T MIN\Final Project\Airbnb_predictive_analysis\Data)"

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




##############QUINN TREE MODEL CODE BELOW##################


#xy_train <- cbind(x_train, y_train_hbr)
#xy_train <- cbind(xy_train,y_train_prs)
#summary(xy_train)

library(tree)

#valid_instn = sample(nrow(x_train), 0.30*nrow(x_train))
#x_train_va <- x_train[valid_instn,]
#x_train_tr <- x_train[-valid_instn,]
#y_train_hbr_va <- y_train_hbr[valid_instn]
#y_train_hbr_tr <- y_train_hbr[-valid_instn]


xy_train <- cbind(x_train, y_train)

train_inst <- sample(nrow(xy_train), .7*nrow(xy_train))
train <- xy_train[train_inst,]
valid <- xy_train[-train_inst,]



mycontrol = tree.control(nrow(train), mincut = 5, minsize = 10, mindev = 0.0005)

full_tree=tree(y_train ~ accommodates + availability_30 + 
                 availability_365 + availability_60 + availability_90 + bathrooms+
                 bed_type + bedrooms + beds + cancellation_policy+
                 cleaning_fee+experiences_offered+extra_people +
                 guests_included+host_acceptance_rate+host_has_profile_pic+
                 host_identity_verified+host_is_superhost+host_listings_count+
                 host_response_rate+
                 host_total_listings_count+instant_bookable+
                 is_business_travel_ready+is_location_exact+latitude+longitude+
                 maximum_nights+minimum_nights+monthly_price+price+
                 require_guest_phone_verification+require_guest_profile_picture+
                 requires_license+room_type+security_deposit+square_feet+
                 state+weekly_price+email+phone+google+reviews+
                 jumio+kba+work_email+facebook+linkedin+selfie+government_id+
                 identity_manual+amex+offline_government_id+sent_id+manual_offline+
                 None+weibo+sesame+sesame_offline+manual_online+photographer+
                 zhima_selfie+has_interaction+is_note+property_category+has_about+
                 host_acceptance+has_host_name+host_response+has_house_rules,
               #had to remove all character vars, all vars with factor levels>32, all vars with spaces in name
               data = train, control = mycontrol)
summary(full_tree)
plot(full_tree)
text(full_tree,pretty=1)

##code to make the one node tree
#pruned_tree_1=prune.tree(full_tree, best = 1)
#summary(pruned_tree_1)
#plot(pruned_tree_1)
#text(pruned_tree_1,pretty=1)

#pruned_tree_1




accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

tree_sizes = c(2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 60)
tr_accs = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
va_accs = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

num_sizes <- length(tree_sizes)
for (i in c(1:num_sizes)){
  va_accs[i] <- accuracy(ifelse(predict(prune.tree(full_tree, best = tree_sizes[i]),newdata=valid)[,2]>.5,"YES","NO"),valid$y_train_hbr)
  tr_accs[i] <- accuracy(ifelse(predict(prune.tree(full_tree, best = tree_sizes[i]),newdata=train)[,2]>.5,"YES","NO"),train$y_train_hbr)
}

plot(tree_sizes, tr_accs, col = "blue", type = 'l')
lines(tree_sizes, va_accs, col = "red")

tr_accs
va_accs

#tr_accs ######run this to see accuracy on training data at different tree_sizes
# ranges from 0.7486547 to 0.7895337





#####RANDOM FORESTS##########

library(randomForest)


rf.mod <- randomForest(y_train_hbr ~ accommodates + availability_30 + 
                         availability_365 + availability_60 + availability_90 + bathrooms+
                         bed_type + bedrooms + beds + cancellation_policy+
                         cleaning_fee+experiences_offered+extra_people,
                       data=xy_train,
                       subset=train_inst,
                       mtry=4, ntree=100,
                       importance=TRUE)

rf_preds <- predict(rf.mod, newdata=valid)
rf_acc <- mean(ifelse(rf_preds==valid$y_train_hbr,1,0))

rf.mod
rf_acc

#plot the variable importances (the average decrease in impurity when splitting across that variable)
importance(rf.mod)
varImpPlot(rf.mod)
