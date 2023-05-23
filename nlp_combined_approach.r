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

x_train_raw <- read.csv(r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\airbnb_train_x_2023.csv)"
                        )
x <- get_cleaned(folder_dir, F)
y_train <- read.csv(r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\airbnb_train_y_2023.csv)")
y_train <- y_train %>%
  mutate(
    high_booking_rate = 
      ifelse(high_booking_rate== 'YES', 1, 0) %>% as.factor(),
    perfect_rating_score = 
      ifelse(perfect_rating_score== 'YES', 1, 0) %>% as.factor(),
  )

hbr = y_train$high_booking_rate
prs = y_train$perfect_rating_score

train_length = nrow(y_train)


# create dtms --------------
# amenities
dtm_am_pruned <- get_dtm(
  x$amenities, 
  custom_stop_words = c('will'),
  tf_idf = T,
  doc_prop_min = 0.01, 
  doc_prop_max = 0.7,
  )
dtm_am_pruned_train = dtm_am_pruned[1:train_length, ]
dtm_am_pruned_te = dtm_am_pruned[(train_length + 1): nrow(dtm_am_pruned), ]


# description dtm
dtm_desc <- get_dtm(
  x$description, tf_idf = T, 
  custom_stop_words = c('will'),
  doc_prop_min = 0.01, doc_prop_max = 0.7
  )
dtm_desc_train = dtm_desc[1:train_length, ]
dtm_desc_te = dtm_desc[(train_length + 1): nrow(dtm_desc), ]


# summary dtm
dtm_summary <- get_dtm(
  custom_stop_words = c('will'),
  x$summary, doc_prop_min = 0.01, doc_prop_max = 0.7
  )
dtm_summary_train = dtm_summary[1:train_length, ]
dtm_summary_te = dtm_summary[(train_length + 1): nrow(dtm_summary), ]


# transit dtm
dtm_transit <- get_dtm(
  custom_stop_words = c('will'),
  x$transit, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_transit_train = dtm_transit[1:train_length, ]
dtm_transit_te = dtm_transit[(train_length + 1): nrow(dtm_transit), ]


# host verification dtm
dtm_hv <- get_dtm(
  custom_stop_words = c('will'),
  x$host_verifications, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_hv_train = dtm_hv[1:train_length, ]
dtm_hv_te = dtm_hv[(train_length + 1): nrow(dtm_hv), ]


# house rule
dtm_hr <- get_dtm(
  custom_stop_words = c('will'),
  x$house_rules, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_hr_train = dtm_hr[1:train_length, ]
dtm_hr_te = dtm_hr[(train_length + 1): nrow(dtm_hr), ]


# interaction dtm
dtm_itrt <- get_dtm(
  custom_stop_words = c('will'),
  x$interaction, doc_prop_min = 0.01, doc_prop_max = 0.7)
dtm_itrt_train = dtm_itrt[1:train_length, ]
dtm_itrt_te = dtm_itrt[(train_length + 1): nrow(dtm_itrt), ]



# read rental price data set ---------
# df_rental <- read.csv(
#   r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\rental_price_house_price.csv)"
# ) %>% select(x.zipcode, average_night_price) %>%
#   rename(zipcode = x.zipcode)
# split ---------------------
ind_sample = sample(1:train_length, size = 0.75 * train_length)

hbr_tr = y_train$high_booking_rate[ind_sample]
hbr_va = y_train$high_booking_rate[-ind_sample]
prs_tr = y_train$perfect_rating_score[ind_sample]
prs_va = y_train$perfect_rating_score[-ind_sample]


dtm_merged_train = cbind(
  dtm_am_pruned_train, dtm_desc_train, dtm_hr_train,
  dtm_hv_train, dtm_itrt_train, dtm_transit_train,
  dtm_summary_train
)
dtm_merged_tr = dtm_merged_train[ind_sample, ]
dtm_merged_va = dtm_merged_train[-ind_sample, ]


dtm_merged_te = cbind(
  dtm_am_pruned_te, dtm_desc_te, dtm_hr_te,
  dtm_hv_te, dtm_itrt_te, dtm_transit_te,
  dtm_summary_te
)



# filter dtm by training model and selecting important features --------

#  md_am_hbr_xgb <- xgboost(
#    data = dtm_am_pruned_train[ind_sample,],
#    label = ifelse(hbr_tr == 1, 1, 0),
#    eta = 0.06,
#    max_depth = 7,
#    nrounds = 500,
#    verbose = T,
#    print_every_n = 100,
#    nthread = 12,
#    weight = ifelse(hbr_tr == 1, 3, 1),
#    objective = 'binary:logistic',
#    eval_metric = 'auc',
#    base_score = mean(ifelse(hbr_tr == 1, 1, 0))
#  )
# 
#  pred_am_hbr_xgb <-
#    predict(md_am_hbr_xgb, newdata = dtm_am_pruned_train[-ind_sample,])
#  get_auc(pred_am_hbr_xgb, hbr_va)
# # 0.7040643 \\ 0.708013 \\ 0.7690409 \\ 0.7817721 \\ 0.7815328 \\ 0.7826265
# vip(md_am_hbr_xgb, 30)
# 
# md_am_hbr_ridge <- glmnet(
#   x = dtm_am_pruned_train[ind_sample,],
#   y = hbr_tr,
#   parallel = T,
#   weights = ifelse(hbr_tr == 1, 2, 1),
#   lambda = 10^-8,
#   family = 'binomial'
# )
# pred_am_hbr_ridge <-
#   predict(md_am_hbr_ridge, newx = dtm_am_pruned_train[-ind_sample,])
# get_auc(pred_am_hbr_ridge, hbr_va)
# # 0.7040643 \\ 0.708013 \\ 0.7690409 \\ 0.7817721 \\ 0.7815328 \\ 0.7826265
# vip(md_am_hbr_ridge, 30)


# if_xgb <- get_important_feature(md_hbr_xgb, dtm_merged_va, quantile_threshold = 0.75)
# saveRDS(if_xgb,
#      file = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Models\if_xgb)")

if_xgb =  readRDS(file = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Models\if_xgb)")
# if_xgb
dtm_subset_tr = dtm_merged_tr %>% subset_dtm(if_xgb)
dtm_subset_va = dtm_merged_va %>% subset_dtm(if_xgb)
dtm_subset_ve = dtm_merged_te %>% subset_dtm(if_xgb)



# original x features ----------------
# x$city %>% boxplot()
# x$state %>% boxplot()
# x$neighbourhood %>% boxplot()
# x$country %>% as.factor() %>% summary()
# x$host_response_time %>% as.factor() %>% histogram()

x_train_raw$neighbourhood %>% get_mode()
x_train_raw$cleaning_fee %>% 
  parse_number() %>% 
  boxplot()
sum(is.na(x_train_raw$cleaning_fee))
x_train_raw$longitude %>% hist()
x_train_raw %>% select(latitude, longitude) %>% plot()
x_train_raw$host_acceptance_rate %>% as.factor() %>% summary()
x_train_raw$host_response_rate %>% as.factor() %>% summary()

x_raw_fe_view$inst %>% quantile(0:20*0.05)
x_raw_fe_view <- x_train_raw %>%
  mutate(
    prs = prs,
    hbr = hbr,
    host_response_rate = parse_number(host_response_rate, na = c("", "NA")),
    host_response_rate = case_when(
      host_response_rate >= 90 ~ ">=90", 
      host_response_rate > 0 ~ '>0',
      T ~ "MISSING or 0",
      ),
    
    host_acceptance_rate = parse_number(host_acceptance_rate, na = c("", "NA")),
    # host_acceptance_rate = # 2023-4-5 fixed
    #   ifelse(
    #     is.na(host_acceptance_rate), 'Missing', host_acceptance_rate),
    host_acceptance_rate= case_when(
      # host_acceptance_rate == 100 ~ '100',
      host_acceptance_rate >= 90 ~ '>=90',
      host_acceptance_rate >= 70 ~ '>=70',
      host_acceptance_rate >= 0 ~ '>=0',
      T ~ 'Missing'
    ),
    
    cleaning_fee = cleaning_fee <- parse_number(cleaning_fee),
    cleaning_fee = ifelse(is.na(cleaning_fee), 0, cleaning_fee),
    cleaning_fee = cleaning_fee %>%
      cut(breaks = quantile(cleaning_fee, 0:4 * 0.25, na.rm = T), include.lowest=T)
  ) %>%
  group_by(market) %>%
  summarise(
    inst = n(),
    hbr_p = sum(hbr == 1) / inst,
    hbr_p_stdev = sd(hbr == 1),
    hbr_n = sum(hbr != 1) / inst,
    # hbr_odds = hbr_p / hbr_n,
    prs_p = sum(prs == 1) / inst,
    prs_p_stdev = sd(prs == 1),
    prs_n = sum(prs != 1) / inst,
    # prs_odds = prs_p / prs_n,
  )


quantile(x$maximum_nights, 0:4*25/100)
x_group_fe_view <- x[1:train_length, ] %>%
  data.frame(
    hbr = hbr,
    prs = prs
  ) %>%
  mutate(
    host_same_neighbor = 
      (neighbourhood == host_neighbourhood %>% as.character()) %>% as.factor(),
    
    bed_per_person = case_when(
      accommodates / beds < 1 ~ '<1',
      accommodates / beds < 2 ~ '[1,2)',
      T ~ 'plenty'
    ),
    
    accommodates = ifelse(accommodates == 0, max(1, bedrooms), accommodates),

    # bathroom_pp = (accommodates / bathrooms) %>% cut(0:5, include.lowest = T),
    bathroom_pp = case_when(
      (accommodates / bathrooms) <= 1 ~ '[0,1]',
      (accommodates / bathrooms) <= 3 ~ '(1,3]',
      T ~ '(3,inf)'
    ),
  
    
    bathroom_pp_ind = accommodates / bathrooms > median(accommodates / bathrooms),
    
    bedroom_pp = accommodates / bedrooms,
    bedroom_pp = case_when(
      bedroom_pp <= 1 ~ '<= 1',
      bedroom_pp <= 2 ~ '<= 2',
      bedroom_pp <= 3 ~ '<= 3',
      T ~ '>3'
    ),
    
    bin_cleaning_fee = cleaning_fee %>% 
      cut(quantile(cleaning_fee, c(0:4*25/100))) ,
    
    availability_365 = case_when(
      availability_365 == 0 ~ '0',
      availability_365 >= 364 ~ '>=364',
      availability_365 >= quantile(availability_365, 0.75) ~ '>=3rd qu.',
      availability_365 >= quantile(availability_365, 0.5) ~ '>=2nd qu.',
      availability_365 >= quantile(availability_365, 0.25) ~ '>=1st qu.',
      T ~ '[1,1st qu.)'
    ) %>% as.factor(),
    
    cleaning_fee = case_when(
      cleaning_fee >= quantile(cleaning_fee, 0.75) ~ '>=3rd qu.',
      cleaning_fee >= quantile(cleaning_fee, 0.5) ~ '>=2nd qu.',
      cleaning_fee >= quantile(cleaning_fee, 0.25) ~ '>=1st qu.',
      cleaning_fee > 0 ~ '>0',
      T ~ '0'
    )%>% as.factor(),
    
    availability_60 = case_when(
      availability_60 == 0 ~ '0',
      availability_60 >= 45 ~ '>=45',
      availability_60 >= 7 ~ '[7, 45)',
      T ~ '[1,7)'
    ) %>% as.factor(),
    
    # extra_over_accom = (extra_people / accommodates) %>%
    #   cut(0:10, include.lowest = T),
    extra_over_accom = (extra_people / accommodates),
    extra_over_accom = case_when(
      extra_over_accom == 0 ~ '0',
      extra_over_accom > quantile(extra_over_accom, 0.75) ~ 'side',
      extra_over_accom < quantile(extra_over_accom, 0.25) ~ 'side',
      T ~ 'Other'
    ),
    
    extra_minus_accom = (extra_people - accommodates) %>%
      cut(0:5, include.lowest = T),
    
    first_review = first_review %>% 
      as.Date() %>% 
      lubridate::year() %>% as.numeric(),
    first_review = case_when(
      first_review <= 2013 ~ '<=2013',
      first_review == 2015 | first_review == 2014 ~ '2014-2015',
      T ~ '>2015'
    ),
    
    price_per_guest = price / guests_included,
    price_per_guest = price_per_guest %>% 
      cut(quantile(price_per_guest, 0:4*25/100), include.lowest = T),
    
    # host_listings_count = ifelse(host_listings_count ==0, 1, host_listings_count),
    # host_listings_count = case_when(
    #   host_listings_count >= quantile(host_listings_count) ~ 'many',
    #   host_listings_count >= 2 ~ 'more than one',
    #   T ~ 'one'
    # ),
    
    host_total_listings_count = case_when(
      host_total_listings_count >= quantile(host_total_listings_count,0.975) ~ 'many',
      host_total_listings_count > 0 ~ 'more than one',
      T ~ 'first time'
    ),
    
    host_since = host_since %>% as.Date() %>% lubridate::year(),
    
    property_category = case_when(
      property_category == 'other' ~ 'other',
      property_category == 'house' ~ 'house',
      T ~ 'hotel like'
    ),
    
    # security_deposit_diff = security_deposit / price,
    # security_deposit_diff = security_deposit_diff %>%
    #   cut(quantile(security_deposit_diff, 0:4*25/100), include.lowest = T),
    
    security_deposit = case_when(
      security_deposit >= quantile(security_deposit,0.9) ~ 'vh',
      security_deposit >= quantile(security_deposit,0.75) ~ 'h',
      security_deposit >= quantile(security_deposit, 0.5) ~ 'm',
      security_deposit == 0 ~ 'no',
      T ~ 'other'
    ),

    license = 
      ifelse(grepl(pattern = 'pending', license, ignore.case = T),
             'pending', license),
    license = ifelse(
      license != 'pending' &
        license != '6240',
      'Other',
      license
    ),
    
    maximum_nights = case_when(
      maximum_nights <= 7 ~ 'weekly',
      maximum_nights <= 30 ~ 'monthly',
      maximum_nights <= 365 ~ 'yearly',
      T ~ 'more than a year'
    )
  ) %>%
  group_by(is_location_exact) %>%
  summarise(
    inst = n(),
    hbr_p = sum(hbr == 1) / inst,
    hbr_p_stdev = sd(hbr == 1),
    hbr_n = sum(hbr != 1) / inst,
    hbr_odds = hbr_p / hbr_n,
    prs_p = sum(prs == 1) / inst,
    prs_p_stdev = sd(prs == 1),
    prs_n = sum(prs != 1) / inst,
    prs_odds = prs_p / prs_n,
  ) %>%
  arrange(desc(hbr_p))
write.csv(x_group_fe_view,
          r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\x_extra_people.csv)")


x$price %>% boxplot()
x$price %>% summary()
x$square_feet %>% boxplot()

x_group_label_view <- x[1:train_length, ] %>%
  data.frame(
    hbr = hbr,
    prs = prs
  ) %>%
  mutate(
    host_same_neighbor = 
      (neighbourhood == host_neighbourhood %>% as.character()) %>% as.factor(),
    price_sqrt = price / square_feet
  ) %>%
  group_by(hbr) %>%
  summarise(
     avg_price = mean(price),
     med_price = median(price),
     min_price = min(price),
     max_price = max(price),
     avg_sqft_price = mean(price_sqrt),
     med_sqft_price = median(price_sqrt)
  )



x_train_raw %>% # square feet is a invalid column
  apply(MARGIN = 2, FUN = \(c) {sum(is.na(c))}) %>%
  sort(decreasing = T)
x_train_raw$host_response_rate %>% median(na.rm = T)
# feature engineering ------------
merge_rental_dataset <- function(df, df_rental){
  df <- df %>%
    merge(
      df_rental, all.x = T, by = 'zipcode'
    ) %>%
    mutate(
      average_night_price =
        ifelse(is.na(average_night_price), 
               median(average_night_price, na.rm = T),
               average_night_price)
    )
  
  return(df)
}


feature_eng <- function(df, price_log = F) {
  verifications = df$host_verifications %>% as.character() %>% tolower()
  amenities = df$amenities %>% as.character() %>% tolower()
  
  df <- df %>%
    # value correction ---------
    mutate(
      accommodates = ifelse(accommodates == 0, max(1, bedrooms), accommodates),
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
      country,
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
      
      
      country = ifelse(country == 'United States', 'US', 'Other') %>%
        as.factor(),
      
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

# important amenity cols 70:84
# colnames(x)
# x$host_verifications
x_for_hbr <- x %>%
  # merge_rental_dataset(df_rental = df_rental) %>%
  feature_eng(price_log = T)
x_for_hbr %>% summary()

df_over_sample  <- over_sampling(
  x = x_for_hbr[1:train_length,], y = hbr, p = 1
)
x_for_hbr <- rbind(x_for_hbr, df_over_sample %>% select(!hbr))

x_cleaned_fac_view <- x_for_hbr[1:train_length, ] %>%
  mutate(
    hbr = hbr,
    prs = prs
  ) %>%
  group_by(neighbourhood) %>%
  summarise(
    count = n(),
    hbr_rate = sum(hbr == 1) / count,
    prs_rate = sum(prs== 1) / count
  ) %>%
  arrange(
    desc(hbr_rate)
  )



x_for_hbr_dum <- to_dummy(x_for_hbr)

x_for_hbr_dum_train = x_for_hbr_dum[1:train_length, ] 
x_for_hbr_dum_te = x_for_hbr_dum[(train_length + 1):nrow(x), ] 
x_for_hbr_tr = x_for_hbr_dum_train[ind_sample, ]
x_for_hbr_va = x_for_hbr_dum_train[-ind_sample, ]


# 
# x_merged_hbr_tr = cbind(dtm_merged_tr, x_for_hbr_tr)
# x_merged_hbr_va = cbind(dtm_merged_va, x_for_hbr_va)
# x_merged_hbr_te = cbind(dtm_merged_te, x_for_hbr_dum_te)



# xgb hbr x training --------------------
md_x_xgb <- xgboost(
  data = x_for_hbr_tr,
  label = ifelse(hbr_tr == 1, 1, 0),
  max.depth = 7,
  eta = 0.06,
  nrounds = 600,
  objective = "binary:logistic",
  eval_metric = "auc",
  verbose = T,
  weight = ifelse(hbr_tr == 1, 2, 1),
  print_every_n = 100,
  nthread = 12
)
xgb_x_pred <- predict(md_x_xgb, x_for_hbr_va)
get_auc(xgb_x_pred, hbr_va)
# 0.8040814 \\ 0.814598 \\ 0.8900042 \\ 0.8935132 \\ 0.8940697
# 0.8950717 (shampoo) \\ 0.8955801 (neighborhood cutoff quantile 0.75)
# 

# saveRDS(
#   md_x_xgb,
#   file = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Models\xgb_hbr_point893.rdata)"
# )




# xgm x tuning -------------
df_vs_xgb <- vector_search(
  vec_param1 = (0:5 * 0.1) + 1,
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  trainer = \(x,y,p) {
    md <- xgboost(
      data = x,
      label = ifelse(y == 1, 1, 0),
      max.depth = 7,
      eta = 0.2,
      nrounds = 200,
      objective = "binary:logistic",
      eval_metric = "auc",
      verbose = T,
      weight = ifelse(y == 1, p, 1),
      print_every_n = 100,
      nthread = 12,
      base_score = ifelse(y == 1, 1, 0) %>% mean()
    )
    return(md)
  },
  predictor = \(md,x){
    return(predict(md, x))
  },
  measurer = \(y1,y2){
    return(
      get_auc(
        y1,y2
      )
    )
  },
  n_per_round = 1
)
df_vs_xgb


# xgb hbr merged training --------------------
# x_for_hbr$bathroom_pp %>% summary()
# x_for_hbr$sqft_price %>% summary()

# setting base_score improves model significantly
md_xgb <- xgboost(
  data = x_merged_hbr_tr,
  label = ifelse(hbr_tr == 1, 1, 0),
  max.depth = 7,
  eta = 0.2, 
  nrounds = 200,
  objective = "binary:logistic",
  eval_metric = "auc", 
  verbose = T,
  weight = ifelse(hbr_tr == 1, 1.1, 1),
  print_every_n = 100,
  nthread = 12,
  subsample = 1,
  base_score = ifelse(hbr_tr == 1, 1, 0) %>% mean()
)
xgb_pred <- predict(md_xgb, x_merged_hbr_va)
get_auc(xgb_pred, hbr_va)
# nround=600 | norund=500 | subsample=0.618 | base_score=p_rate
# 0.8609318 \\ 0.8604589 \\ 0.8606085      \\ 0.8610364   \\ 0.85922
# 0.8634284 \\ 0.8653248 \\ 0.868463  \\ 0.8694929 \\ 0.8716606
# 0.868860  \ 0.872371 \\ 0.8721934 \\ 0.8708768 \\ 0.872461
# 0.8727355 \\ 0.8733932 \\ 0.8773783 \\ 0.8780786 \\ 0.8791728
# 0.8798415 (instant bookable) \\ 0.8805404 (business travel)
# 0.8822878 (market) \\ 0.8831319(security deposit)
# 0.8832139 (license) \\ 0.8846236 (maximum nights)
# 0.8893832 (host acc rate & host resp rate)
# 0.8920743 (longtiude latitude original)


# saveRDS(
#   md_xgb,
#   file = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Models\xgb_hbr_point889.rdata)"
# )
# md_loaded = readRDS(
#   file = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Models\xgb_hbr_point889.rdata)"
# )
# md_loaded$feature_names
xgb_pred <- predict(md_loaded, x_merged_hbr_va)
get_auc(xgb_pred, hbr_va)


# ranger x hbr training --------------------
md_ranger_hbr <- ranger(
  x = x_for_hbr_tr, y = hbr_tr,
  max.depth = 30,
  num.trees = 1000,
  importance = 'impurity',
  probability = T,
  class.weights = c(1, 1.5),
  num.threads = 12
)
ranger_hbr_pred <- predict(md_ranger_hbr, x_for_hbr_va)$predictions[,2]
get_auc(ranger_hbr_pred, hbr_va)
# 0.8789127


# ranger x prs training --------------------
md_ranger_prs <- ranger(
  x = x_for_hbr_tr, y = prs_tr,
  max.depth = 30,
  num.trees = 1000,
  importance = 'impurity',
  probability = T,
  class.weights = c(1, 1.5),
  num.threads = 12
)
ranger_prs_pred <- predict(md_ranger_prs, x_for_hbr_va)$predictions[,2]
get_auc(ranger_prs_pred, prs_va)
# 0.6761264 \\ 0.6634543 \\ 0.6581618 \\ 0.7300763 \\ 0.770795 \\ 0.7711039
get_cutoff_dataframe(ranger_prs_pred, prs_va, step = 0.01) %>%
plot_cutoff_dataframe()


# ridge merged training -------------
md_ridge_hbr<- glmnet(
  x = x_merged_hbr_tr, y = hbr_tr,
  alpha = 1,
  family = 'binomial',
  lambda = 10^-8,
  weights = ifelse(hbr_tr == 1, 1.6, 1),
  parallel= T
)
pred_ridge_hbr <- predict(md_ridge_hbr, x_merged_hbr_va, type = 'response')
get_auc(pred_ridge_hbr, hbr_va)
# .8449566  \\ 0.8473909 \\ 0.8604988