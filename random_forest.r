# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file
source('Library\\data_cleaning.r')
source('Library\\utils.r')
# source('Library\\external_dataset_hotel.r')
library(xgboost)
library(ranger)
library(caret)
library(glmnet)
options(scipen = 999)
# replace the string with the directory of the project folder "Data"
# the original datasets MUST be placed under the Data folder for the following script to run correctly.
folder_dir = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis_copy\Data)"

# read -----------------
x_full_set <- get_cleaned(folder_dir, FALSE)
y_train <- read.csv('Data\\airbnb_train_y_2023.csv')

# feature engineering ------------

feature_engineering_full_set <- function(df) {
  df <- df %>%
    group_by(city) %>%
    mutate(
      city_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      city = ifelse(
        city_count <= 200, 'Other', city # originally 50
      ) %>% as.factor()
    ) %>%
    select(!city_count) %>%
    group_by(market) %>%
    mutate(
      market_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      market = ifelse(
        market_count <= 25, 'Other', market
      ) %>% as.factor()
    ) %>%
    select(!market_count) %>%
    mutate(
      host_same_neighbor = 
        (neighbourhood == host_neighbourhood %>% as.character()) %>% as.factor(),
    ) %>%
    group_by(neighbourhood) %>%
    mutate(
      count_neighbor = n()
    ) %>%
    ungroup() %>%
    mutate(
      neighbourhood = ifelse(count_neighbor < 350, 'Other', neighbourhood) %>%
        as.factor()
    ) %>%
    select(!count_neighbor) %>%
    mutate(
      min_night_length = case_when(
        minimum_nights >= 365 ~ 'Year',
        minimum_nights >= 93 ~ 'Season',
        minimum_nights >= 30 ~ 'Month',
        minimum_nights >= 7 ~ 'Week',
        minimum_nights >= 2 ~ 'Days',
        TRUE ~ 'No'
      ),
      
      host_response_time = host_response_time %>% as.factor()
    ) %>%
    group_by(state) %>%
    mutate(
      state_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      state = ifelse(state_count <= 15, 'other', state) %>% as.factor()
    ) %>%
    select(
      !state_count
    )
  
  return(df)
}


feature_engineering <- function(x) {
  
  high_hotel_rate_city = x_full_set$city %in% c(
    'new york', 'miami', 'chicago', 'las vegas', 'san fransisco'
  )
  
  res <- x %>%
    select(
      accommodates,
      availability_30,
      availability_365,
      availability_60,
      availability_90,
      bathrooms,
      bedrooms,
      bed_type,
      beds,
      city,
      cancellation_policy,
      cleaning_fee,
      extra_people,
      first_review,
      guests_included,
      
      host_acceptance_rate,
      host_has_profile_pic,
      host_identity_verified,
      host_is_superhost,
      host_listings_count,
      host_same_neighbor,
      host_response_rate,
      host_response_time,
      host_since,
      
      longitude,
      latitude,
      market,
      min_night_length,
      neighbourhood,

      instant_bookable,
      is_business_travel_ready,
      require_guest_phone_verification,
      require_guest_profile_picture,
      room_type,
      # state,
      
      # added 2023-4-17
      `self check-in`,
      `hair dryer`,
      hangers,
      washer,
      shampoo,
      kitchen,
      `first aid kit`,
      `24-hour check-in`,
      `lock on bedroom door`,
      # `air conditioning`, cannot select
      `free parking on premises`,
      `fire extinguisher`,
      tv,
      `carbon monoxide detector`,
      # `hot water`,
      # `buzzer`,
      # family,
      
      
      # removed 2023-4-17
      # license,
      
      
      
      property_category,
      price,
      
      
      # access_snmt,
      # desc_snmt,
      # host_about_snmt,
      # house_rules_snmt,
      # interaction_snmt,
      # neighborhood_snmt,
      # notes_snmt,
      # summary_snmt
    ) %>%
    mutate(
      country = ifelse(x$country_code == 'US', 'US', 'Other') %>%
        as.factor(),
      
      host_response_time = as.factor(x$host_response_time),
      
      price_per_person = x$price / ifelse(x$accommodates == 0, 1, x$accommodates),

      ppp_ind = ifelse(price_per_person > median(price_per_person),1 , 0),
      
      price = ifelse(price < 1, 1, price) %>% # this is to prevent -inf
        log(), 
      
      price_per_sqfeet = x$price / 
        ifelse(x$square_feet == 0, median(x$square_feet), x$square_feet),
      
      monthly_price =
        ifelse(x$monthly_price < 1, 1, x$monthly_price) %>%
        log(),
      
      square_feet = 
        ifelse(x$square_feet == 0, median(x$square_feet), x$square_feet),
      
      high_hotel_rate_city = high_hotel_rate_city
  )

  
  return(res)
}


x <- x_full_set %>%
  feature_engineering_full_set() %>%
  feature_engineering()


hbr <- y_train$high_booking_rate %>% as.factor()
hbr <- ifelse(hbr == 'YES', 1, 0)
prs <- y_train$perfect_rating_score %>% as.factor()
prs <- ifelse(prs == 'YES', 1, 0)
x_train <- x[1:nrow(y_train),]
x_test <- x[(nrow(y_train) + 1): nrow(x),]

md_dummy <- dummyVars(formula = ~., x, fullRank = TRUE)
x_tr_dummy <- predict(md_dummy, x_train)
x_te_dummy <- predict(md_dummy, x_test)


# test
nrow(x_train) == nrow(y_train)
nrow(x_test) == 12205
(length(hbr) == length(prs)) && (length(hbr) == nrow(y_train))


# train-validation split -------------------------
sampled = base::sample(1:nrow(x_train), 0.75 * nrow(x_train))
x_tr_rf = x_tr_dummy[sampled, ]
x_va_rf = x_tr_dummy[-sampled, ]
hbr_tr = hbr[sampled]
hbr_va = hbr[-sampled]
prs_tr = prs[sampled]
prs_va = prs[-sampled]


# feature comparing -----------------------
x1 <- x_train %>%
  select(!high_hotel_rate_city)
d1 <- dummyVars(formula = ~., x1, fullRank = T)
x1 <- predict(d1, x1)


x2 <- x_train
x2 <- predict(md_dummy, x2)


df_cf = compare_feature(
  x1, x2, prs, 
  trainer = \(x, y) {
    model <- ranger(x = x, y = y,
                    mtry = 26, 
                    max.depth = 10, # seemingly optimal
                    num.trees=800,
                    importance="impurity",
                    probability = TRUE,
                    verbose = F
                    )
    return(model)
  },
  predictor = \(model, x) {
    pred = predict(model, data = x)$predictions[,2]
    return(pred)
  },
  measurer = \(y1, y2) {
    return(get_auc(y1, y2))
  }
)



#view -----------------
x_view <- x[1:nrow(y_train),]
# feature of interest
foi <- x_view$state
summary(foi) # useful
boxplot(foi)


tar_fac <- y_train$perfect_rating_score %>% as.factor()
obj_test <- x_view %>%
  mutate(
    host_response_time = host_response_time %>% as.factor()
  ) %>%
  cbind(tar_fac) %>% 
  group_by(high_hotel_rate_city) %>%
  mutate(
    inst_count = n()
  ) %>%
  ungroup() %>%
  filter(tar_fac == 'YES') %>%
  group_by(high_hotel_rate_city) %>%
  mutate(
    p_count = n(),
    p_rate = p_count / inst_count
  ) %>%
  select(high_hotel_rate_city, inst_count, p_rate) %>%
  arrange(inst_count) %>%
  distinct()


ggplot(data = obj_test, aes(x = neighbourhood, y = inst_count)) +
  geom_bar(stat = "identity")

boxplot(obj_test$inst_count)
summary(obj_test$inst_count)
hist(x$monthly_price)
summary(x$price)


# codes start here ----------------------------


# learning curve using ranger -----------------

# create held-out data (holding 10% out)

folds <- cut(
    1:nrow(x_train) %>% sample(size = nrow(x_train)), 
    breaks = tr_fold_max + 1, 
    labels = FALSE
  )
# summary(folds)
x_heldout <- x_train[which(folds == tr_fold_max + 1, arr.ind = TRUE),]
prs_heldout <- prs[which(folds == tr_fold_max + 1, arr.ind = TRUE)]
# nrow(x_heldout) == length(prs_heldout)

vec_sample_size = rep(0, tr_fold_max)
vec_acc_heldout = rep(0, tr_fold_max)
for (ind in 1:tr_fold_max) {
  
  indice_tr = which(folds <= ind, arr.ind = TRUE)
  sample_size = length(indice_tr)
  vec_sample_size[ind] = sample_size
  
  x_tr = x_train[indice_tr, ]
  prs_tr = prs[indice_tr]
  # x_va = x_use[-sampled, ]
  # hbr_tr = hbr[sampled]
  # hbr_va = hbr[-sampled]
  # prs_va = prs_use[-sampled]
  
  
  md_ranger <- ranger(x = x_tr, y = prs_tr,
                             mtry=26, 
                             num.trees=800,
                             importance="impurity",
                             probability = TRUE)
  
  # validation set
  # y_pred_prob <- 
  #   predict(md_hbr_rf_ranger, data = x_va)$predictions[,2]
  # y_pred = ifelse(y_pred_prob >= 0.5, 1, 0)
  # acc = get_accuracy(y_pred, prs_va)
  # vec_acc[ind] = acc
  
  
  # heldout set
  y_pred_prob <- 
    predict(md_ranger, data = x_heldout)$predictions[,2]
  y_pred = ifelse(y_pred_prob >= 0.5, 1, 0)
  acc_heldout = get_accuracy(y_pred, prs_heldout)
  vec_acc_heldout[ind] = acc_heldout
}


ggplot(
  data = data.frame(
    sample_size = vec_sample_size, 
    accuracy = vec_acc_heldout
  ),
  aes(x = sample_size, y = accuracy)
) + geom_line()



# xgb prs ----------------------------
md_prs_xgb <- xgboost(
  data = as.matrix(x_tr_rf_dummy),
  label = prs_tr_dummy,
  nrounds = 2000,
  print_every_n = 100,
  objective = 'binary:logistic',
  eval_metric = "auc"
)
y_pred_prob_xgb <- predict(md_prs_xgb, newdata = x_va_rf_dummy)
plot_roc(y_pred_prob_xgb, prs_va)
get_auc(y_pred_prob_xgb, prs_va)

get_cutoff_dataframe(y_pred_prob_xgb, prs_va_dummy, level = c(0,1)) %>%
  plot_cutoff_dataframe()
# 0.7906043



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
md_hbr_ranger <- ranger(x = x_tr_rf, y = hbr_tr,
                 mtry=26, num.trees=800,
                 importance="impurity",
                 probability = TRUE)
y_pred_prob_hbr_ranger <- 
  predict(md_hbr_ranger, data = x_va_rf)$predictions[,2]
plot_roc(y_pred_prob_hbr_ranger, hbr_va)
get_auc(y_pred_prob_hbr_ranger, hbr_va)
# 0.8733637 \\ 0.873 \\ 0.8663719 \\ 0.8788666


# ranger prs ----------------------------
# optimal num tree = 800, mtry = 26
md_prs_ranger <- ranger(x = x_tr_rf, y = prs_tr,
                           mtry = 26, 
                           max.depth = 24, # seemingly optimal
                           num.trees=800,
                           importance="impurity",
                           probability = TRUE)
y_pred_prob_prs_ranger <- 
  predict(md_prs_ranger, data = x_va_rf)$predictions[,2]
plot_roc(y_pred_prob_prs_ranger, prs_va)
get_auc(y_pred_prob_prs_ranger, prs_va)
plot_roc_2(y_pred_prob_prs_ranger, prs_va)
# last 0.8131754 \\ 0.8049635 \\ 0.8144257 \\ 0.814458 \\ 
# highest 0.8131754


vec_auc_cv <- cross_val(
  trainer = \(x, y) {
    md <- ranger(
        x = x, y = y,
        mtry = 26, 
        max.depth = 24, # seemingly optimal
        num.trees=800,
        importance="impurity",
        probability = TRUE
      )
    return(md)
  },
  predictor = \(model, x) {
    y_pred <- predict(model, data = x)$predictions[,2]
    return(y_pred)
  },
  measurer = \(y_pred, y_va) {
    return(get_auc(y_pred, y_va))
  },
  x = x_tr_dummy,
  y = prs
)
vec_auc_cv


vec_depth = (1:6) * 3 + 12
vec_auc_by_depth = iterate_on(
  on = vec_depth,
  action = \(depth) {
    md_prs <- ranger(
      x = x_tr_rf, y = prs_tr,
      mtry = 26, 
      max.depth = depth,
      num.trees=800,
      importance="impurity",
      probability = TRUE)
    pred_prob = predict(md_prs, data = x_va_rf)$predictions[,2]
    return(get_auc(pred_prob, prs_va))
  },
)
vec_depth[which.max(vec_auc_by_depth)]
vec_auc_by_depth


df_cutoff <- get_cutoff_dataframe(y_pred_prob_prs_ranger, prs_va, 
                       level = c(0, 1),
                       max_fpr = 0.068)
df_cutoff %>% plot_cutoff_dataframe()
df_cutoff$cutoff_bound[1]



final_pred_prob = predict(md_prs_ranger, data = x_te_dummy)$predictions[,2]
final_pred_cls = ifelse(final_pred_prob > df_cutoff$cutoff_bound[1], 'YES', 'NO')
wd <- getwd()
setwd(folder_dir)
write.table(final_pred_cls, "perfect_rating_score_group5.csv", row.names = FALSE)
setwd(wd)

saveRDS(md_prs_ranger,
        file = 
          r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Models\md_prs_ranger_0501_auc_081.rsd)")




# new code testing ----------------
vec_param = 0:3 * 3 + 20
vec_auc <- iterate_on(
  on = vec_param,
  action = \(param) {
    md_hbr_rf_ranger <- ranger(
      x = x_tr_rf_dummy, 
      y = prs_tr_rf,
      num.trees=800, # 800 appears optimal
      mtry = param,
      importance="impurity",
      probability = TRUE
    )
    
    y_pred_prob_prs_ranger <- 
      predict(md_hbr_rf_ranger, data = x_va_rf_dummy)$predictions[,2]
    return(get_auc(y_pred_prob_prs_ranger, prs_va))
  }
)


ggplot(
  data = data.frame(
    mtry = vec_param,
    auc = vec_auc
  ),
  aes(x = mtry, y = auc)
) + geom_line()



# logistic lasso ------------------
# best lambda appears to be 10^-7
vec_lasso_lambda = seq(from = 10^-7, to = 2, length.out = 100)
vec_lass_auc <- iterate_on(
  on = vec_lasso_lambda,
  action = \(param) {
    logistic_lasso_prs <- glmnet(
      x = x_tr_rf, 
      y = prs_tr,
      lambda = param, 
      alpha = 0,
      family="binomial"
    )
    
    pred_prob <- predict(logistic_lasso_prs, newx = x_va_rf, type = "response")
    return(
      get_auc(pred_prob, prs_va)
    )
  },
  
  verbose = FALSE
)

logistic_lasso_prs <- glmnet(
  x = x_tr_rf, 
  y = prs_tr,
  lambda = 10^-7, 
  alpha = 0,
  family="binomial"
)

pred_prob <- predict(logistic_lasso_prs, newx = x_va_rf, type = "response")
get_auc(pred_prob, prs_va)

ggplot(
  data = data.frame(
    lambda = vec_lasso_lambda,
    auc = vec_lass_auc
  ),
  aes(x = lambda, y = auc)
) + geom_line()


# fpr checking flow -----------------------

check_time = 4
set_cutoff = 0.515
sample_size = 60000
vec_fpr = rep(0, check_time)
vec_tpr = rep(0, check_time)
x_source = rbind(x_tr_rf, x_va_rf)
y_source = c(prs_tr, prs_va)
for (ind in 1:check_time) {
  sampled = sample(1:nrow(x_source), size = sample_size, replace = FALSE)
  x_sampled = x_source[sampled,]
  y_sampled = y_source[sampled]

  p = 1
  n = 0
  
  y_pred_prob = predict(md_prs_ranger, data = x_sampled)$predictions[,2]
  y_pred_cls = ifelse(y_pred_prob >= set_cutoff, p, n)
  

  # cm = caret::confusionMatrix(
  #   data = y_pred_cls %>% as.factor(), #predictions
  #   reference = y_sampled %>% as.factor(), #actuals
  #   positive = '1') 
  
  get_tpr <- function(y_pred, y_val, n, p) {
    tp = sum((y_pred == y_val) & (y_pred == p))
    return(
      tp / sum(y_val == p)
    )
  }
  
  
  get_fpr <- function(y_pred, y_val, n, p) {
    fp = sum((y_pred == p) & (y_pred != y_val)) 
    return(
      fp / sum(y_val == n)
    )
  }
  
  
  vec_fpr[ind] = get_fpr(y_pred_cls, y_sampled, n, p)
  vec_tpr[ind] = get_tpr(y_pred_cls, y_sampled, n, p)
}


ggplot(
  data = data.frame(
    fpr = vec_fpr,
    tpr = vec_tpr
  ),
  aes(x = fpr, y = tpr)
) + geom_point()



# logistic hbr --------------------------
md_logistic <- glm(data.frame(x_tr_rf_dummy, hbr_tr), formula = hbr_tr~., family = 'binomial')

y_pred_prob_logit <- predict(md_logistic, newdata = data.frame(x_va_rf_dummy), type = 'response')
plot_roc(y_pred_prob_logit, hbr_va)



# logistic prs --------------------------
md_prs_logit <- glm(data.frame(x_tr_rf_dummy, prs_tr), formula = prs_tr~., family = 'binomial')

y_pred_prob_prs_logit <- predict(md_prs_logit, newdata = data.frame(x_va_rf_dummy), type = 'response')
plot_roc(y_pred_prob_prs_logit, prs_va)




# analysis ------------------------


summary(x_tr$room_type)
summary(x_tr_rf$price)
boxplot(x_tr_rf$price)
boxplot(x_tr_rf$monthly_price)
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









# new dataset ------------------
# df_redfin <- read.csv('Data\\city_market_tracker.tsv000', sep = '\t')
# 
# 
# df_redfin_2023_mar <- df_redfin %>%
#   mutate(
#     period_begin = period_begin %>% as.Date(),
#     period_end = period_end %>% as.Date(),
#   ) %>%
#   select(
#     !period_duration
#   ) %>%
#   filter(
#     (period_begin %>% data.table::year()) == 2023 &
#       (period_begin %>% data.table::month()) == 3
#   )
# 
# 
# find_monotonous(df_redfin_2023_mar)

df_redfin_2023_mar <- read.csv('Data\\Redfin_city_market_tracker_2023_3.csv')

names(df_redfin_2023_mar)

df_redfin_current <- df_redfin_2023_mar %>%
  group_by(city) %>%
  mutate(
    city_occurence = n()
  ) %>%
  arrange(
    city_occurence
  ) %>%
  ungroup() %>%
  mutate(
    city = 
      ifelse(city_occurence < 5, 'other', city),
    state = state_code
  ) %>%
  select(
    !c(
      city_occurence,
      period_begin,
      period_end,
      region_type,
      region_type_id,
      is_seasonally_adjusted,
      last_updated,
      state_code
    )
  ) %>%
  group_by(state, city) %>%
  mutate(
    median_price = mean(median_sale_price, na.rm = TRUE)
  ) %>%
  ungroup()

names(df_redfin_current)
summary(df_redfin_current$median_price)
boxplot(df_redfin_current$median_price)
summary(x$state)

x_full_set$state[1:10]
df_redfin_current$state[1:10]
sum(x_full_set$city %in% df_redfin_2023_mar$city)
sum(x_full_set$state %in% df_redfin_2023_mar$state)
nrow(x_view)


summary(x$city)
summary(df_redfin_current$state)
