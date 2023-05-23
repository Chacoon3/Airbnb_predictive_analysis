source('Library\\data_cleaning.r')
source('Library\\utils.r')
library(tidyverse)
library(Metrics)
library(caret)
library(class)
library(readr)
library(text2vec)
library(tm)
library(SnowballC)
library(vip)
library(textdata)
library(tidytext)
library(quanteda)
library(glmnet)
library(ranger)
library(tree)
library(rpart)
library(xgboost)
library(e1071)



# read ------------------------------
folder_dir = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data)"


x <- get_cleaned(folder_dir, F)

y_train <- read.csv(r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\airbnb_train_y_2023.csv)") %>%
  mutate(
    high_booking_rate = 
      ifelse(high_booking_rate== 'YES', 1, 0) %>% as.factor(),
    perfect_rating_score = 
      ifelse(perfect_rating_score== 'YES', 1, 0) %>% as.factor(),
  )

hbr = y_train$high_booking_rate
prs = y_train$perfect_rating_score

train_length = nrow(y_train)



# feature engineering ------------
feature_eng <- function(df, price_log = T) {
  verifications = df$host_verifications %>% as.character() %>% tolower()
  amenities = df$amenities %>% as.character() %>% tolower()
  top_dest = c('paris',
               'beijing',
               'orlando',
               'shanghai',
               'las vegas',
               'new york',
               'tokyo', 
               'mexico city', 
               'london',
               'guangzhou') %>% tolower()
  
  df <- df %>%
    # value correction and addition ---------
  mutate(
    accommodates = ifelse(accommodates == 0, max(1, bedrooms), accommodates),
    is_top_dest = (city_name %>% tolower()) %in% top_dest,
  ) %>%
    # city processing ----------
  group_by(city) %>%
    mutate(
      city_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      city = ifelse(
        city_count <= quantile(city_count, 0.25), 'Other', city
      ) %>% as.factor()
    ) %>%
    select(!city_count) %>%
    # market processing ----------
  group_by(market) %>%
    mutate(
      market_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      market = ifelse(
        market_count < quantile(market_count, 0.65), 'Other', market
      ) %>% as.factor()
    ) %>%
    select(!market_count) %>%
    # neighbourhood processing ----------
  group_by(neighbourhood) %>%
    mutate(
      count_neighbor = n()
    ) %>%
    ungroup() %>%
    mutate(
      neighbourhood =
        ifelse(count_neighbor < quantile(count_neighbor, 0.75), 'Other',
               neighbourhood) %>%
        as.factor()
    ) %>%
    select(!count_neighbor) %>%
    # state value correction ---------------
  mutate(
    state = state %>% as.character(),
    state = case_when(
      state == 'NEW YORK' ~ 'NY',
      state == 'BAJA CALIFORNIA' ~ 'CA',
      T ~ state
    ) %>% as.factor()
  ) %>%
    # state processing ----------
  group_by(state) %>%
    mutate(
      state_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      state = ifelse(state_count <= quantile(state_count, 0.25), 'Other', state) %>% as.factor()
    ) %>%
    select(
      !state_count
    )
  
  
  
  res <- df %>%
    # feature selection -------------
  select(
    availability_30,
    availability_60,
    availability_90,
    availability_365,
    accommodates,
    beds,
    bed_type,
    bedrooms,
    bathrooms,
    # country,
    cleaning_fee,
    extra_people,
    first_review,
    guests_included,
    host_response_time,
    host_response_rate,
    host_acceptance_rate,
    host_is_superhost,
    host_listings_count,
    host_total_listings_count,
    host_since,
    instant_bookable,
    is_business_travel_ready,
    # is_top_dest,
    license,
    latitude,
    longitude,
    require_guest_phone_verification,
    require_guest_profile_picture,
    room_type,
    security_deposit,
    neighbourhood,
    minimum_nights,
    maximum_nights,
    market,
    property_category,
    price,
    weekly_price,
    monthly_price
  ) %>%
    # factor mutation ---------------
  mutate(
    availability_30 = case_when(
      availability_30 >= 28 ~ 'monthly',
      availability_30 >= 21 ~ '3 weeks',
      availability_30 >= 14 ~ '2 weeks',
      availability_30 >= 7 ~ 'a week',
      availability_30 >= 1 ~ 'within a week',
      T ~ 'always'
    ) %>% as.factor(),
    
    
    availability_60 = case_when(
      availability_60 == 0 ~ '0',
      availability_60 >= 45 ~ '>=45',
      availability_60 >= 7 ~ '[7, 45)',
      T ~ '[1,7)'
    ) %>% as.factor(),
    
    
    availability_365 = case_when(
      availability_365 == 0 ~ '0',
      availability_365 >= 364 ~ '>=364',
      availability_365 >= quantile(availability_365, 0.75) ~ '>=3rd qu.',
      availability_365 >= quantile(availability_365, 0.5) ~ '>=2nd qu.', 
      availability_365 >= quantile(availability_365, 0.25) ~ '>=1st qu.', 
      T ~ '[1,1st qu.)'
    ) %>% as.factor(),
    
    
    bathrooms = ifelse(bathrooms == 0, 1, bathrooms),
    bathroom_pp = case_when(
      (accommodates / bathrooms) <= 1 ~ '[0,1]',
      (accommodates / bathrooms) <= 3 ~ '(1,3]',
      T ~ '(3,inf)'
    ) %>% as.factor(),
    
    bedroom_pp = accommodates / bedrooms,
    bedroom_pp = case_when(
      bedroom_pp <= 1 ~ '<= 1',
      bedroom_pp <= 2 ~ '<= 2',
      bedroom_pp <= 3 ~ '<= 3',
      T ~ '>3'
    ) %>% as.factor(),
    
    # extra_people = (extra_people / accommodates),
    # extra_people = case_when(
    #   extra_people == 0 ~ '0',
    #   extra_people > quantile(extra_people, 0.75) ~ 'side',
    #   extra_people < quantile(extra_people, 0.25) ~ 'side',
    #   T ~ 'Other'
    # ) %>% as.factor(),
    
    license = 
      ifelse(grepl(pattern = 'pending', license, ignore.case = T),
             'pending', license),
    license = ifelse(
      license != 'pending' &
        license != '6240',
      'Other',
      license
    ) %>% as.factor(),
    
    
    minimum_nights = case_when(
      minimum_nights >= 365 ~ 'Year',
      minimum_nights >= 93 ~ 'Season',
      minimum_nights >= 30 ~ 'Month',
      minimum_nights >= 7 ~ 'Week',
      minimum_nights >= 2 ~ 'Days',
      TRUE ~ 'No'
    ) %>% as.factor(),
    
    
    maximum_nights = case_when(
      maximum_nights <= 7 ~ 'weekly',
      maximum_nights <= 30 ~ 'monthly',
      maximum_nights <= 365 ~ 'yearly',
      T ~ 'more than a year'
    ) %>% as.factor(),
    
    
    # country = ifelse(country == 'United States', 'US', 'Other') %>%
    #   as.factor(),
    
    bed_pp = case_when(
      accommodates / beds < 1 ~ '<1',
      accommodates / beds < 2 ~ '[1,2)',
      T ~ '[2, inf)'
    ) %>% as.factor(),
    
    
    bed_type = bed_type == 'Real Bed',
    
    
    cleaning_fee = case_when(
      cleaning_fee >= quantile(cleaning_fee, 0.75) ~ '>=3rd qu.',
      cleaning_fee >= quantile(cleaning_fee, 0.5) ~ '>=2nd qu.',
      cleaning_fee >= quantile(cleaning_fee, 0.25) ~ '>=1st qu.',
      cleaning_fee > 0 ~ '>0',
      T ~ '0'
    ) %>% as.factor(),
    
    
    first_review = first_review %>% 
      as.Date() %>% 
      lubridate::year() %>% as.numeric(),
    first_review = case_when(
      first_review <= 2013 ~ '<=2013',
      first_review == 2015 | first_review == 2014 ~ '2014-2015',
      T ~ '>2015'
    ) %>% as.factor(),
    
    
    host_response_time = case_when(
      host_response_time == 'within an hour' ~ 'within an hour',
      host_response_time == 'within a few hours' ~ 'within a few hours',
      T ~ 'Other',
    ) %>% as.factor(),
    
    
    price_per_guest = price / guests_included,
    price_per_guest = price_per_guest %>% 
      cut(quantile(price_per_guest, 0:4*25/100), include.lowest = T),
    
    
    host_acceptance_rate= case_when(
      # host_acceptance_rate == 100 ~ '100',
      host_acceptance_rate >= 90 ~ '>=90',
      host_acceptance_rate >= 70 ~ '>=70',
      host_acceptance_rate >= 0 ~ '>=0',
      T ~ 'Missing'
    ) %>% as.factor(),
    
    
    host_response_rate = case_when(
      host_response_rate >= 90 ~ ">=90", 
      host_response_rate > 0 ~ '>0',
      T ~ "MISSING or 0",
    ) %>% as.factor(),
    
    
    host_listings_count = ifelse(host_listings_count ==0, 1, host_listings_count),
    host_listings_count = case_when(
      host_listings_count >= 36 ~ 'many',
      host_listings_count >= 2 ~ 'more than one',
      T ~ 'one'
    ) %>% as.factor(),
    
    
    host_total_listings_count = case_when(
      host_total_listings_count >= quantile(host_total_listings_count,0.975)
      ~ 'many',
      T ~ 'other'
    ) %>% as.factor(),
    
    host_since = host_since %>% as.Date() %>% 
      lubridate::year() %>% as.numeric(),
    
    security_deposit = case_when(
      security_deposit >= quantile(security_deposit,0.9) ~ 'vh',
      security_deposit >= quantile(security_deposit,0.75) ~ 'h',
      security_deposit >= quantile(security_deposit, 0.5) ~ 'm',
      security_deposit == 0 ~ 'no',
      T ~ 'other'
    ) %>% as.factor(),
    
    property_category = case_when(
      property_category == 'other' ~ 'other',
      property_category == 'house' ~ 'house',
      T ~ 'hotel like'
    ) %>% as.factor(),
    
    # price_higher_avg = price > df$average_night_price
  ) %>%
    # categorical list mutation ------------
  mutate(
    # verifications 
    offline_verify = grepl(pattern = 'offline', verifications),
    id_verify = grepl(pattern = 'id|identity', verifications),
    email_verify = grepl(pattern = 'email|mail', verifications),
    phone_verify = grepl(pattern = 'phone', verifications),
    reviews_verify = grepl(pattern = 'review', verifications),
    social_media_verify = grepl(
      pattern = 'google|weibo|facebook|linkedin', verifications),
    third_party_verify = grepl(
      pattern = 'kba|jumio|sesame', verifications),
    
    
    # amenities
    air_detector = amenities %>% grepl(
      pattern = 'detector'
    ),
    air_conditioner = amenities %>% grepl(
      pattern = 'air conditioning|air conditioner'
    ),
    all_day_service = amenities %>% grepl(
      pattern = '24-hour'
    ),
    self_service = amenities %>% grepl(
      pattern = 'self'
    ),
    bathtub = amenities %>% grepl(
      pattern = 'bath|tub'
    ),
    bedroom_lock = amenities %>% grepl(
      pattern = 'lock on bedroom door'
    ),
    extinguisher = amenities %>% grepl(
      pattern = 'extinguisher'
    ),
    first_aid = amenities %>% grepl(
      pattern = 'first aid kit'
    ),
    free_parking = amenities %>% grepl(
      pattern = 'free parking|free [a-z]+ parking'
    ),
    internet = amenities %>% grepl(
      pattern = 'internet|wifi|wireless'
    ),
    heating = amenities %>% grepl(pattern = 'heating'),
    hair_dryer = amenities %>% grepl(
      pattern = 'hair dryer|dryer'
    ),
    hanger = amenities %>% grepl(
      pattern = 'hanger'
    ),
    kitchen = amenities %>% grepl(
      pattern = 'kitchen'
    ),
    shampoo = amenities %>% grepl(
      pattern = 'shampoo'
    ),
    tv = amenities %>% grepl(
      pattern = 'tv',
    ),
    washer = amenities %>% grepl(
      pattern = 'washer'
    ),
  ) %>%
    # numerical mutation -------------
  mutate(
    # availability_60 = ifelse(availability_60 < 1, 1, availability_60) %>% log(),
    
    price = ifelse(
      price > quantile(price, 0.75, na.rm = T), 
      quantile(price, 0.75, na.rm = T), price),
    price = ifelse(price <= 1, median(price, na.rm = T), price),
    price_pp = price / accommodates,
    price_pp_ind = price_pp > median(price_pp),
    
    
    weekly_price = ifelse(weekly_price <= 1, 7 * price, weekly_price),
    monthly_price = ifelse(monthly_price <= 1, 30 * price, monthly_price)
  )
  # column binding --------------
  # cbind(
  #   df[, 70:84]
  # )
  
  # parameter dependent operations --------
  if (price_log) {
    res <- res %>%
      mutate(
        price = log(price),
        weekly_price = log(weekly_price),
        monthly_price = log(monthly_price),
        price_pp = log(price_pp)
      )
  }
  
  return(res)
}


to_dummy <- function(df) {
  md_dum = dummyVars(~., df, fullRank = T)
  return(
    predict(md_dum, df)
  )
}


x_for_hbr <- x %>%
  feature_eng(price_log = F)

x_for_hbr %>% summary()
# x_for_hbr[1:train_length, ] %>% data.frame(
#   hbr = hbr
# ) %>%
#   group_by(tv) %>%
#   summarise(
#     inst = n(),
#     p_count = sum(hbr == 1),
#     p_rate = p_count / inst
#   )

# sum(x$country != 'United States')
# sum(x_train_raw$country!= 'United States')

x_for_hbr_dum <- to_dummy(x_for_hbr)

x_for_hbr_dum_te = x_for_hbr_dum[(train_length + 1):nrow(x), ] 
x_for_hbr_dum_train = x_for_hbr_dum[1:train_length, ] 


# remove country outlier ----------------
# note that there is no country outlier in testing set

ind_country_outliers = which(x[1:train_length, ]$country != 'United States' )
x_for_hbr_dum_train <- x_for_hbr_dum_train[-ind_country_outliers, ]
hbr_filtered = hbr[-ind_country_outliers]


# splitting -------------
ind_sample <- sample(1:length(hbr_filtered), 0.7 * length(hbr_filtered))
x_for_hbr_tr = x_for_hbr_dum_train[ind_sample, ]
x_for_hbr_va = x_for_hbr_dum_train[-ind_sample, ]

hbr_tr = hbr_filtered[ind_sample]
hbr_va = hbr_filtered[-ind_sample]



# training xgb -----------------
# colnames(x_for_hbr_tr)
md_hbr_xgb <- xgboost(
  data = x_for_hbr_tr,
  label = ifelse(hbr_tr == 1, 1, 0),
  max.depth = 8,
  eta = 0.1,
  nrounds = 600,
  objective = "binary:logistic",
  verbose = T,
  weight = ifelse(hbr_tr == 1, 1.6, 1),
  print_every_n = 100,
  nthread = 12,
  eval_metric = "auc",
  base_score = mean(ifelse(hbr_tr == 1, 1, 0))
)
pred_hbr_xgb <- predict(md_hbr_xgb, x_for_hbr_va)
get_auc(pred_hbr_xgb, hbr_va)
# 0.7 ratio
# 0.8917648 \\ 0.8946285 (added top dest)



# cross validating xgb -------------------

cv_res <- cross_val(
  # trainer = example_trainer,
  trainer = \(x,y) {
    return( xgboost(
      data = x,
      label = ifelse(y == 1, 1, 0),
      max.depth = 8,
      eta = 0.1,
      nrounds = 550,
      objective = "binary:logistic",
      verbose = T,
      weight = ifelse(y == 1, 1.6, 1),
      print_every_n = 100,
      nthread = 12,
      eval_metric = "auc",
      base_score = mean(ifelse(y == 1, 1, 0))
    ))
  },
  predictor = \(m,x) {
    return(predict(m, x))
  },
  measurer = \(y1,y2) {
    return(get_auc(y1,y2))
  },
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va)
)
cv_res %>% max()
# with top dest     avg 0.8927897 max 0.8941699 
# without top dest  avg 0.8928667 max 0.895 
# without tv avg 0.8926228 max 0.896694

# tuning xgb  -----------------
# optimal nrounds = 800, eta = 0.1, max.depth =6
df_cs <- cube_search(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  vec_param1 = 1:8 * 0.05, #eta
  vec_param2 = 1:4 * 4 + 5, # depth
  vec_param3 = 1:3 * 400, # rounds
  trainer = \(x,y,p1,p2,p3) {
    return(
      xgboost(
        data = x,
        label = ifelse(y == 1, 1, 0),
        eta = p1,
        max.depth = p2,
        nrounds = p3,
        objective = "binary:logistic",
        eval_metric = "auc",
        verbose = T,
        weight = ifelse(y == 1, 2, 1),
        print_every_n = 100,
        nthread = 12
      )
    )
  },
  predictor = \(m,x) {
    return(
      predict(m,x)
    )
  },
  measurer = \(y1,y2) {
    return(
      get_auc(y1, y2)
    )
  },
  n_per_round = 2
)
df_cs
write.csv(
  x = df_cs,
  file = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Models\cube_search_edn_xgb.csv)"
)









df_ms <- matrix_search(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  vec_param1 = 4:8 * 100, # rounds
  vec_param2 = 6:8, # depth
  trainer = \(x,y,p1,p2) {
    return(
      xgboost(
        data = x,
        label = ifelse(y == 1, 1, 0),
        eta = 0.1,
        nrounds = p1,
        max.depth = p2,
        objective = "binary:logistic",
        eval_metric = "auc",
        verbose = T,
        weight = ifelse(y == 1, 2, 1),
        print_every_n = 100,
        nthread = 12
      )
    )
  },
  predictor = \(m,x) {
    return(
      predict(m,x)
    )
  },
  measurer = \(y1,y2) {
    return(
      get_auc(y1, y2)
    )
  },
  n_per_round = 1
)
df_ms


write.csv(
  x = df_ms,
  file = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Models\matrix_search_xgb.csv)"
)



# training ridge ---------------------
md_hbr_ridge <- glmnet(
  x = x_for_hbr_tr,
  y = hbr_tr,
  parallel= T,
  lambda = 1e-7,
  family = 'binomial',
  alpha  = 1,
  weights = ifelse(hbr_tr == 1, 1.6, 1)
)
pred_ridge_hbr = predict(md_hbr_ridge, x_for_hbr_va)
get_auc(pred_ridge_hbr, hbr_va)
# 0.8661152
length(pred_ridge_hbr) == length(xgb_x_pred)
pred_avg = (pred_ridge_hbr + xgb_x_pred) / 2
get_auc(pred_avg, hbr_va)


# training ranger ----------------
# class weights helps
ncol(x_for_hbr_tr)
pn_rate = sum(hbr_tr == 1) / sum(hbr_tr != 1)
md_ranger_hbr <- ranger(
  x = x_for_hbr_tr, y = hbr_tr,
  max.depth = 18,
  num.trees = 2000,
  min.bucket = 10,
  importance = 'impurity',
  probability = T,
  class.weights = c(1, 1 / pn_rate),
  num.threads = 12
)
ranger_hbr_pred <- predict(md_ranger_hbr, x_for_hbr_va)$predictions[,2]
get_auc(ranger_hbr_pred, hbr_va)
get_accuracy(ifelse(ranger_hbr_pred >= 0.5, 1, 0), hbr_va)


# 0.8573872 \\ 0.8707894 depth from 10 to 15
# 0.8661867 min.bucket 10 cls weight n > p
# 0.8672256 cls weight n < p
# 0.8696877 (increase depth and ntree)



