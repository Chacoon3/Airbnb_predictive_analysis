# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file.
source('Library\\data_cleaning.r')

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

fe_naivebayes <- function(x) {
  res <- x %>%
    select(!c(
      first_review,
      host_since,
      host_verifications,
      city_name,
      country,
      country_code,
      host_location,
      host_neighbourhood,
      neighbourhood,
      require_guest_phone_verification,
      requires_license
    )) %>%
    mutate(
      accommodates = scale(accommodates),
      bed_type = as.factor(bed_type),
      city = as.factor(city),
      cleaning_fee = scale(cleaning_fee),
      extra_people = scale(extra_people),
      host_acceptance_rate = scale(host_acceptance_rate),
      host_has_profile_pic = as.factor(host_has_profile_pic),
      host_identity_verified = as.factor(host_identity_verified),
      host_is_superhost = as.factor(host_is_superhost),
      host_response_rate = scale(host_response_rate),
      host_response_time = as.factor(host_response_time),
      instant_bookable = as.factor(instant_bookable),
      is_business_travel_ready = as.factor(is_business_travel_ready),
      latitude = scale(latitude),
      longitude = scale(longitude),
      market = as.factor(market),
      monthly_price = scale(monthly_price),
      square_feet = scale(square_feet),
      weekly_price = scale(weekly_price)
    )
  return(res)
}


train_naivebayes <- function(x, y) {
  md <- e1071::naiveBayes(x = x, y = y, laplace = 1)
  return(md)
}


x_tr_nb <- fe_naivebayes(x_tr)
x_va_nb <- fe_naivebayes(x_va)
# get_shape(x_tr_nb)
# get_shape(x_va_nb)
# high booking rate
md_hbr_nb <- train_naivebayes(x_tr_nb, hbr_tr)
pred_hbr_nb <- predict(md_hbr_nb, newdata = x_va_nb)
acc_hbr_nb <- get_accuracy(pred_hbr_nb, hbr_va)
acc_hbr_nb
# perfect rating score
md_prs_nb <- train_naivebayes(x_tr_nb, prs_tr)
pred_prs_nb <- predict(md_prs_nb, newdata = x_va_nb)
acc_prs_nb <- get_accuracy(pred_prs_nb, prs_va)
acc_prs_nb


vec_laplace = 1:20
accs = rep(0, length(vec_laplace))
for (ind in 1:length(vec_laplace)) {
  lap = vec_laplace[ind]
  md <- e1071::naiveBayes(x = x_tr, y = hbr_tr, laplace = lap)
  pred_prob = predict(md, newdata = x_va_nb, type = "raw")
  pred_cls = ifelse(pred_prob[,2] >= pred_prob[, 1], 'YES', 'NO')
  accs[ind] = get_accuracy(pred_cls, hbr_va)
}
accs
# [1] 0.4109858 0.4087054 0.4075052 0.4063450 0.4053849 0.4045447 0.4037446 0.4027844 0.4020243
# [10] 0.4017443 0.4011042 0.4009041 0.4004241 0.3997440 0.3991039 0.3986238 0.3981037 0.3972636
# [19] 0.3969035 0.3965034

vec_laplace = 1:10 / 10
accs_2 = rep(0, length(vec_laplace))
for (ind in 1:length(vec_laplace)) {
  lap = vec_laplace[ind]
  md <- e1071::naiveBayes(x = x_tr, y = hbr_tr, laplace = lap)
  pred_prob = predict(md, newdata = x_va_nb, type = "raw")
  pred_cls = ifelse(pred_prob[,2] >= pred_prob[, 1], 'YES', 'NO')
  accs_2[ind] = get_accuracy(pred_cls, hbr_va)
}
accs_2
# [1] 0.4137862 0.4131461 0.4125460 0.4123860 0.4120259 0.4117459 0.4115859 0.4115058 0.4112658
# [10] 0.4109858