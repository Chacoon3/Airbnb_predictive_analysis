# library and sourcing -------------
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
options(scipen = 12)
set.seed(1)


# global triggers
first_time_index = T
global_reindex = F


# read ------------------------------
folder_dir = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data)"
x <- get_cleaned(folder_dir, F)


df_ws <- read.csv(
  r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\walk_score.csv)",
  colClasses = c('zipcode' = 'character') 
)


df_live_score <- read.csv(
  r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\live_score.csv)",
  colClasses = c('zipcode' = 'character') 
) %>%
  mutate(
    live_score = live_score %>% replace_na(median(live_score)),
    house_score = house_score %>% replace_na(median(house_score)),
    neighbor_score = neighbor_score %>% replace_na(median(neighbor_score)),
    trans_score = trans_score %>% replace_na(median(trans_score)),
    env_score = env_score %>% replace_na(median(env_score)),
  )


df_crime_rate <- read.csv(
  r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\crime rate by zipcode.csv)",
  colClasses = c('zipcode' = 'character', 'risk' = 'character') 
) %>%
  mutate(
    risk = parse_number(risk)
  ) %>%
  select(!X)


df_zipcode_area <- read.csv(
  r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\zipcode_area.csv)",
  colClasses = c('zipcode' = 'character')
) %>%
  mutate(
    zipcode = substr(zipcode,1,5),
    area_land_km = area_land_meters / (1000^2),
    area_water_km = area_water_meters / (1000^2)
  ) %>%
  select(
    !c('area_land_meters', 'area_water_meters')
  )


# df_zipcode_pop <- read.csv(
#   r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\zipcode_population_2010.csv)",
#   colClasses = c('zipcode' = 'character', 'gender' = 'factor')
# ) %>% 
#   group_by(zipcode) %>%
#   mutate(
#     population = mean(population)
#   )
# df_zipcode_pop = df_zipcode_pop %>%
#   select(zipcode, population) %>%
#   unique()



# df_station <- read.csv(
#   r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\station.csv)",
#   colClasses = c('osm_id' = 'character', 'lon' = 'numeric', 'lat' = 'numeric')
# )


x_train_raw <- read.csv(r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\airbnb_train_x_2023.csv)",
  colClasses = c('zipcode' = 'character')
)


x_test_raw <- read.csv(r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Data\airbnb_test_x_2023.csv)",
  colClasses = c('zipcode' = 'character')
)


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


# df_zipcode_pop %>% group_by(zipcode) %>%
#   summarise(
#     inst = n(),
#     uniq = unique(population) %>% length()
#   )
# summary(merge_test$population)
# merge_test = left_join(
#   x = x[train_length:nrow(x), ], y = df_zipcode_pop,
#   by = 'zipcode'
# )
# merge_test$population %>% summary()

# sum(merge_test$is_business_travel_ready == x_test_raw$is_business_travel_ready,
#     na.rm = T)
# x_test_raw$is_business_travel_ready %>% is.na() %>% sum()
# nrow(x_test_raw)
# names(merge_test)


# scratch ---------------
# 
# x_for_hbr$live_score %>% boxplot()
# df_crime_rate %>% 
#   select(
#     overall_crime_rate, violent_crime_rate, poverty_crime_rate, other_crime_rate
#   ) %>%
#   boxplot()
# 
# df_test_join = left_join(
#   x,
#   df_crime_rate %>% select(overall_crime_rate, violent_crime_rate),
#   by = 'zipcode'
# )
# df_test_join$zipcode %>% as.factor() %>% summary()
# df_test_join$overall_crime_rate %>% summary()
# obj_test = count_station(
#   vec_lat = x_test_raw$latitude,
#   vec_lng = x_test_raw$longitude,
#   ref_lat = df_station$lat,
#   ref_lng = df_station$lon
# )

# ind_country_outlier <- which(x_train_raw$country != 'United States', arr.ind= T)
# y_train[ind_country_outlier, ]

# x_train_raw %>%
#   select(longitude, latitude) %>%
#   mutate(
#     longitude = scale(longitude),
#     latitude = scale(latitude)
#   ) %>%
#   plot()
# 
x_raw_view <- x_train_raw %>% data.frame(
  hbr = hbr
) %>%
  group_by(host_has_profile_pic %>% replace_na(F)) %>%
  summarise(
    inst_count = n(),
    p_rate = sum(hbr == 1) / inst_count,
    n_rate = sum(hbr!=1) / inst_count
  ) %>%
  arrange(
    desc(p_rate)
  )
# 
# (x_test_raw$first_review %>%
#   as.Date() %>% 
#   lubridate::year() == 2009) %>% sum()


# feature engineering ------------
# df_ws$bike_score %>% boxplot()
feature_eng <- function(df, price_log = T) {
  verifications = df$host_verifications %>% as.character() %>% tolower()
  amenities = df$amenities %>% as.character() %>% tolower()

  df <- df %>%
    # value correction and addition ---------
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
      inst_count = n()
    ) %>%
    ungroup() %>%
    mutate(
      neighbourhood =
        ifelse(inst_count < quantile(inst_count, 0.75), 'Other',
               neighbourhood) %>%
        as.factor()
    ) %>%
    select(!inst_count) %>%
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
    city_name,
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
    host_identity_verified,
    host_has_profile_pic,
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
    monthly_price,
    zipcode # select zipcode so as to merge walk score dataset
  ) %>%
    # factor mutation ---------------
  mutate(
    # availability_30 = case_when(
    #   availability_30 >= 28 ~ 'monthly',
    #   availability_30 >= 21 ~ '3 weeks',
    #   availability_30 >= 14 ~ '2 weeks',
    #   availability_30 >= 7 ~ 'a week',
    #   availability_30 >= 1 ~ 'within a week',
    #   T ~ 'always'
    # ) %>% as.factor(),
    
    
    # availability_60 = case_when(
    #   availability_60 == 0 ~ '0',
    #   availability_60 >= 45 ~ '>=45',
    #   availability_60 >= 7 ~ '[7, 45)',
    #   T ~ '[1,7)'
    # ) %>% as.factor(),
    
    
    # availability_365 = case_when(
    #   availability_365 == 0 ~ '0',
    #   availability_365 >= 364 ~ '>=364',
    #   availability_365 >= quantile(availability_365, 0.75) ~ '>=3rd qu.',
    #   availability_365 >= quantile(availability_365, 0.5) ~ '>=2nd qu.', 
    #   availability_365 >= quantile(availability_365, 0.25) ~ '>=1st qu.', 
    #   T ~ '[1,1st qu.)'
    # ) %>% as.factor(),
    
    # availability_outlier = 
    #   (availability_30 == 0 | availability_30 == 30) * 1+
    #   (availability_60 == 0 | availability_60 == 60) * 1+
    #   (availability_365 == 0 | availability_365 == 365) * 1,
    
    bathroom_na = is.na(bathrooms),
    bathrooms = ifelse(is.na(bathrooms), 0, bathrooms),
    bathroom_pp = case_when(
      bathrooms == 0 ~ '0',
      (accommodates / bathrooms) <= 1 ~ '(0,1]',
      (accommodates / bathrooms) <= 3 ~ '(1,3]',
      T ~ '(3,inf)'
    ) %>% as.factor(),
    
    bedroom_na = is.na(bedrooms),
    bedrooms = ifelse(is.na(bedrooms), 0, bedrooms),
    bedroom_pp = case_when(
      bedrooms == 0 ~ '0',
      (accommodates / bedrooms) <= 1 ~ '(0,1]',
      (accommodates / bedrooms) <= 2 ~ '(1,2]',
      (accommodates / bedrooms) <= 3 ~ '(2,3]',
      T ~ '>3'
    ) %>% as.factor(),
    
    beds_na = is.na(beds),
    beds = ifelse(is.na(beds), 0, beds),
    
    bed_pp = case_when(
      beds == 0 ~ '0',
      accommodates / beds <= 1 ~ '(0,1]',
      accommodates / beds <= 2 ~ '(1,2]',
      T ~ '[2, inf)'
    ) %>% as.factor(),
    bed_type = bed_type == 'Real Bed',
    
    
    city_name = city_name %>% as.factor(),
    cleaning_fee_na = is.na(cleaning_fee),
    
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
    
    
    # minimum_nights = case_when(
    #   minimum_nights >= 365 ~ 'Year',
    #   minimum_nights >= 93 ~ 'Season',
    #   minimum_nights >= 30 ~ 'Month',
    #   minimum_nights >= 7 ~ 'Week',
    #   minimum_nights >= 2 ~ 'Days',
    #   TRUE ~ 'No'
    # ) %>% as.factor(),
    # 
    # 
    # maximum_nights = case_when(
    #   maximum_nights <= 7 ~ 'weekly',
    #   maximum_nights <= 30 ~ 'monthly',
    #   maximum_nights <= 365 ~ 'yearly',
    #   T ~ 'more than a year'
    # ) %>% as.factor(),
    
    
    # country = ifelse(country == 'United States', 'US', 'Other') %>%
    #   as.factor(),
    
    
    # cleaning_fee = case_when(
    #   cleaning_fee >= quantile(cleaning_fee, 0.75) ~ '>=3rd qu.',
    #   cleaning_fee >= quantile(cleaning_fee, 0.5) ~ '>=2nd qu.',
    #   cleaning_fee >= quantile(cleaning_fee, 0.25) ~ '>=1st qu.',
    #   cleaning_fee > 0 ~ '>0',
    #   T ~ '0'
    # ) %>% as.factor(),
    
    
    first_review = first_review %>% 
      as.Date() %>% 
      lubridate::year() %>% as.numeric(),
    # first_review = case_when(
    #   first_review <= 2013 ~ '<=2013',
    #   first_review == 2015 | first_review == 2014 ~ '2014-2015',
    #   T ~ '>2015'
    # ) %>% as.factor(),
    
    
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
    
    # price_outlier = price >= 1000
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
      pattern = 'air conditioning|air conditioner|conditioner|conditioning'
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
    computer_or_laptop = amenities %>% grepl(
        pattern = 'computer|laptop',ignore.case = T
    ),
    crib = amenities %>% grepl(
      pattern = 'crib', ignore.case = T
    ),
    # doorman = amenities %>% grepl(
    #   pattern = 'doorman', ignore.case = T
    # ),
    extinguisher = amenities %>% grepl(
      pattern = 'extinguisher'
    ),
    first_aid = amenities %>% grepl(
      pattern = 'first aid'
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
    # tv = amenities %>% grepl(
    #   pattern = 'tv',
    # ),
    washer = amenities %>% grepl(
      pattern = 'washer'
    ),
  ) %>%
    # numerical mutation -------------
  mutate(
    amenities_count = x$amenities %>% str_count(','),
    amenities_count = ifelse(
      is.na(amenities_count), 0, amenities_count
    ),
    
    cleaning_fee = ifelse(
      is.na(cleaning_fee), 0, cleaning_fee
    ),

    desc_word_count = x$description %>% str_count("\\w+"),
    desc_word_count = ifelse(
      is.na(desc_word_count), 0, desc_word_count
    ),
    
    # hostabout_word_count = x$host_about %>% str_count("\\w+"),
    # hostabout_word_count = ifelse(
    #   is.na(hostabout_word_count), 0, hostabout_word_count
    # ),
    # 
    # neighborhood_overview_word_count = x$neighborhood_overview %>%
    #   str_count("\\w+"),
    # neighborhood_overview_word_count = ifelse(
    #   is.na(neighborhood_overview_word_count), 0, neighborhood_overview_word_count
    # ),
    # 
    summary_word_count = x$summary %>% str_count("\\w+"),
    summary_word_count = ifelse(
      is.na(summary_word_count), 0, summary_word_count
    ),
    
    # space_word_count = x$space %>%
    #   str_count(pattern = '\\w+'),
    # space_word_count = ifelse(
    #   is.na(space_word_count), 0, space_word_count
    # ),
    
    # why translate av 30? cuz it demonstrates negative corr. with p_rate.
    # yet the case of 0 is an outlier so we translate that case.
    availability_30 = ifelse(
      availability_30 == 0, 29, availability_30
    ),
    # availability_60 = ifelse(availability_60 < 1, 1, availability_60) %>% log(),
    
    
    minimum_nights = ifelse(is.na(minimum_nights), 0, minimum_nights),
  
    maximum_nights = ifelse(
        is.na(maximum_nights), get_mode(maximum_nights), maximum_nights
      ),
    
    night_range = maximum_nights - minimum_nights,
    # price = ifelse(
    #   price > quantile(price, 0.75, na.rm = T), 
    #   quantile(price, 0.75, na.rm = T), price),
    # price = ifelse(price <= 1, median(price, na.rm = T), price),
    price_pp = price / accommodates,
    price_pp_ind = price_pp > median(price_pp),
    
    no_review_period = first_review - host_since,
    
    
    weekly_price = ifelse(weekly_price <= 1, 7 * price, weekly_price),
    monthly_price = ifelse(monthly_price <= 1, 30 * price, monthly_price)
  ) %>%
    # natural language missing value flags  ----------------
    mutate(
      # amenities_na = is.na(df$amenities),
      access_na = is.na(df$access),
      desc_na = is.na(df$description),
      host_about_na = is.na(df$host_about),
      # host_name_na = is.na(df$host_name),
      house_rules_na = is.na(df$house_rules),
      interaction_na = is.na(df$interaction),
      name_na = is.na(df$name),
      neighborhood_overview_na = is.na(df$neighborhood_overview),
      notes_na = is.na(df$notes),
      space_na = is.na(df$space),
      # street_na = is.na(df$street),
      summary_na = is.na(df$summary),
      transit_na = is.na(df$transit),
      # private = grepl(pattern = 'private', x = summary, ignore.case = T)
      # transit_uber = grepl(pattern = 'uber',transit)
      # manhattan = grepl(pattern = 'manhattan',transit),
      # downtown = grepl(pattern = 'downtown', transit),
      # station = grepl(pattern = 'station', transit),
      # walk = grepl(pattern = 'walk', transit)
      # free = grepl(pattern = 'free', transit)
    )
  
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


x_for_hbr_fe <- feature_eng(x, price_log = F)

# ncol(x_for_hbr_fe)
# names(x_for_hbr_fe)
# 
# summary(x_for_hbr_fe$space_word_count)
# merging external data --------------
merge_external <- function(x, df_ws, df_zipcode_area, df_live_score) {
  
  res <- left_join(
    x = x, y = df_ws,
    by = 'zipcode'
  )  %>%
    mutate(
      walk_score =
        ifelse(is.na(walk_score), median(walk_score, na.rm = T), walk_score),
      bike_score =
        ifelse(is.na(bike_score), median(bike_score, na.rm = T), bike_score)
    ) %>%
    left_join(
      y = df_zipcode_area %>% select(c('zipcode', 'area_land_km')),
      by = 'zipcode'
    ) %>%
    mutate(
      area_land_km = ifelse(
        is.na(area_land_km),
        median(area_land_km, na.rm = T), 
        area_land_km
      )
    ) %>%
    group_by(zipcode) %>%
    mutate(
      inst = n(),
      list_density = ifelse(
        is.na(area_land_km) | inst < 10,
        0,
        inst / area_land_km
      )
    ) %>%
    ungroup() %>% 
    # left_join(
    #     df_crime %>% select(zipcode, overall_crime_rate, violent_crime_rate),
    #     by = 'zipcode'
    # ) %>%
    # left_join(
    #   df_live_score %>% select(zipcode, live_score),
    #   by = 'zipcode'
    # ) %>%
    # mutate(
    #   live_score = replace_na(live_score, replace = median(live_score, na.rm = T))
    # ) %>%
    select(!c('zipcode', 'inst'))
  
  
  # station_count = count_station(
  #   vec_lat = res$latitude,
  #   vec_lng = res$longitude,
  #   ref_lat = df_station$lat,
  #   ref_lng = df_station$lon
  # )
  # res[, 'station_count'] = station_count
  
  
  return(
    res
  )
}


x_for_hbr  <- merge_external(
  x_for_hbr_fe,
      df_ws = df_ws,
      df_zipcode_area = df_zipcode_area,
      df_live_score = df_live_score
    )
  

# x_for_hbr$host_since[(train_length + 1):nrow(x_for_hbr) ] %>% summary()

# quantile(x_for_hbr$price, 0.995)
# x_for_hbr[1:train_length, ] %>% data.frame(
#   hbr = hbr
# ) %>%
#   group_by(host_name_na) %>%
#   summarise(
#     inst = n(),
#     p_count = sum(hbr == 1),
#     p_rate = p_count / inst,
#     n_count = sum(hbr != 1),
#     n_rate = n_count / inst
#     # avg_price = mean(price),
#     # median_price = median(price),
#     # min_price = min(price)
#     # max_price = max(price)
#   ) %>%
#   select(!c('p_count', 'n_count'))
# 
# 
# obj_view = x_for_hbr[1:train_length, ] %>% data.frame(
#   hbr = hbr
# ) %>%
#   group_by(list_density) %>%
#   summarise(
#     inst = n(),
#     p_count = sum(hbr == 1),
#     p_rate = p_count / inst,
#     n_count = sum(hbr != 1),
#     n_rate = n_count / inst
#   ) %>%
#   select(!c('p_count', 'n_count')) %>%
#   arrange(desc(p_rate))
# 
# cor.test(
#   x_for_hbr$area_land_km, x_for_hbr$list_density
# )
# 0.3397728 coorelation coeff significant for walk score
# 0.3173795 coorelation coeff significant for bike score


# sum(x$country != 'United States')
# sum(x_train_raw$country!= 'United States')

# to dummy and add cluster flag ------------------------
x_for_hbr_dum <- to_dummy(x_for_hbr)


x_for_hbr_dum = cbind(
  x_for_hbr_dum,
  get_cluster_label(
    x_for_hbr_dum + 0, preprocess = 'scale'
  )
)


x_for_hbr_dum_te = x_for_hbr_dum[(train_length + 1):nrow(x), ] 
x_for_hbr_dum_train = x_for_hbr_dum[1:train_length, ] 


# remove country outlier ----------------
# note that there is no country outlier in testing set

ind_outliers = which(
  x[1:train_length, ]$country != 'United States')
x_for_hbr_dum_train <- x_for_hbr_dum_train[-ind_outliers, ]
hbr_filtered = hbr[-ind_outliers]
prs_filtered = prs[-ind_outliers]

# splitting -------------
if (first_time_index) {
  first_time_index = F
  ind_sample <- sample(1:length(hbr_filtered), 0.7 * length(hbr_filtered))
} else if (global_reindex) {
  ind_sample <- sample(1:length(hbr_filtered), 0.7 * length(hbr_filtered))
}


x_for_hbr_tr = x_for_hbr_dum_train[ind_sample, ]
x_for_hbr_va = x_for_hbr_dum_train[-ind_sample, ]

hbr_tr = hbr_filtered[ind_sample]
hbr_va = hbr_filtered[-ind_sample]
prs_tr = prs_filtered[ind_sample]
prs_va = prs_filtered[-ind_sample]


x_for_hbr %>% select(
  maximum_nights, minimum_nights
) %>%
  boxplot()
summary(x_for_hbr$host_has_profile_pic)
summary(x_for_hbr$host_identity_verified)
# (x_for_hbr$maximum_nights > (365 * 3)) %>% sum()
# nrow(x_for_hbr) == (nrow(x_train_raw) + nrow(x_test_raw))

# over sampling ---------------
# oversample_size = sum(hbr_tr == 1) * 0.3
# ind_oversample = sample(which(hbr_tr == 1, arr.ind = T, ), replace = T, oversample_size)
# x_for_hbr_tr = rbind(
#   x_for_hbr_tr,
#   x_for_hbr_tr[ind_oversample, ]
# )
# hbr_tr = c(
#   hbr_tr,
#   hbr_tr[ind_oversample]
# )



# training hbr xgb -----------------
# colnames(x_for_hbr_tr)
md_x_xgb <- xgboost(
  data = x_for_hbr_tr,
  label = ifelse(hbr_tr == 1, 1, 0),
  max.depth = 7,
  eta = 0.09,
  nrounds = 500,
  objective = "binary:logistic",
  verbose = T,
  print_every_n = 100,
  nthread = 12,
  eval_metric = "auc"
)
xgb_x_pred <- predict(md_x_xgb, x_for_hbr_va)
get_auc(xgb_x_pred, hbr_va)
plot_roc_ggplot(xgb_x_pred, hbr_va)
# 0.7 ratio
# 0.8917648 \\ 0.8946285 (added top dest) \\ 0.8897774 (add price outlier)
# 0.8963029 \\ 0.8965139 \\ 0.9021844 \\ 0.901308 \\ 0.9017588 \\ 0.9021038

# get_vip_dataframe(md_x_xgb, x_for_hbr_va)

# vip xgb --------------
vip(md_x_xgb, 50)

# error analysis ---------------
df_res_pred <- 
  data.frame(
    x_for_hbr[1:train_length, ][-ind_outliers, ][-ind_sample, ],
    prob = xgb_x_pred,
    label = hbr_va
  )

write.csv(
  df_res_pred,
  file = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis\Models\prediction_err_0514.csv)"
)

xgb_x_pred %>% hist()


# complexity analysis xgb hbr ------------
# max depth 7 appears optimal
# nround appears optimal around 500
# eta 0.09 appears optimal
# better no class weight
# better no base score

plot_complexity_curve(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  vec_param1 = c(7,8), # optimal 
  trainer = \(x,y,p1) {
    return(
      xgboost(
        data = x,
        label = ifelse(y == 1, 1, 0),
        eta = 0.09,
        max.depth = 7,
        nrounds = 500,
        objective = "binary:logistic",
        eval_metric = "auc",
        verbose = F,
        weight = ifelse(y == 1, 1, 1),
        print_every_n = 100,
        nthread = 12
      )
    )
  },
  predictor = predict,
  measurer = get_auc,
  n_per_round = 3
)



# tuning xgb  -----------------
# optimal nrounds = 325, eta = 0.11, max.depth = 8 
#       param1 param2 param3 measurement
# 1    0.10      8    325   0.9033545
df_cs <- cube_search(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  vec_param1 = c(0.1, 0.11, 0.12), 
  vec_param2 = 8, 
  vec_param3 = c(275, 300, 325, 350), 
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
        weight = ifelse(y == 1, 1.5, 1),
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


# cross validating xgb -------------------
ncol(x_for_hbr_tr)
cv_res <- cross_val(
  trainer = \(x,y) {
    return( xgboost(
      data = x,
      label = ifelse(y == 1, 1, 0),
      max.depth = 7,
      eta = 0.09,
      nrounds = 500,
      objective = "binary:logistic",
      verbose = T,
      weight = ifelse(y == 1, 1, 1),
      print_every_n = 100,
      nthread = 12,
      eval_metric = "auc"
      # base_score = mean(ifelse(y == 1, 1, 0))
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
cv_res %>% mean()
cv_res %>% min()
cv_res %>% max()

# cross valid log  ------------
# with top dest     avg 0.8927897 max 0.8941699 
# without top dest  avg 0.8928667 max 0.895 
# without tv        avg 0.8926228 max 0.896694
# without price outlier flag  avg 0.8919701 max 0.8967024
# with continuous cleaning fee avg 0.8939847 max 0.8963233
# with continuous availability 30 avg 0.8940891 max 0.8960364
# with 0 translated av30  avg. 0.8940892 max. 0.8968122
# with continous av60     avg. 0.8943849 max.0.896302
# with continuous av365   avg. 0.8948406 max. 0.896302
# with availability outlier avg. 0.8942957  | max.0.8961886 (appears useless)
# with original bed_type avg. 0.8943832 | max. 0.8965065
# with continuous bathroom pp avg. 0.8943314 | max 0.898603
# with numeric first review year avg. 0.896005 | max. 0.8978125
# removed first review year 2008 avg. 0.8958425 |  max. 0.8973087
# 05/13 initial         avg. 0.8960354 |  max. 0.8989604
# 05/13 initial         avg. 0.8953351 max. 0.8981013
# added computer|laptop avg. 0.8957504 | max. 0.8995607
# added no transit      avg. 0.8962241 | max. 0.8973485
# walk added                        avg. 0.8961877 | max. 0.8985124
# ref                               avg. 0.8961758 | max. 0.8988523
# walk score dataset merged         avg. 0.8965719 | max. 0.8989053
# zipcode area and #list/area added avg. 0.8966663 | max. 0.8999599
# zipcode population added          avg. 0.8968125 | max. 0.8988109
# no_review_period added            avg. 0.8968125 | max. 0.8988109
# ref                               avg. 0.8961758 | max. 0.8988523
# pop and no review period          avg. 0.8970172 | max. 0.9008017
# ref                               avg. 0.8971002 | max. 0.9014378
# na imputation with df_ws and df_area avg. 0.8967548 | max. 0.8976834
# imputation with df_ws and df_area avg. 0.8970283 | max. 0.9016073
# with zipcode population           avg. 0.8970103 | max. 0.8997069
# with station count                avg. 0.8968159 | max. 0.8989788
# with access_na                    avg. 0.8971553 | max. 0.8982766
# with na flag of all nlp columns   avg. 0.8971048 | max. 0.8981779
# with factor city name             avg. 0.8979282 | max. 0.8999678
# (note that city name has larger granularity than city)


# cross valid log part 2 -----------------------
# ref                               avg. 0.8991295 | max. 0.9012278
# unbounded numerical min max night avg. 0.9026869 | max. 0.9050356
# 3 year bounded max night          avg. 0.9020988 | max. 0.9049463
# mode imputed unbound max night and night range added
#                                   avg. 0.9024655 | max. 0.9050503
# night range removed               avg. 0.9026036 | max. 0.9040455
# beds removed fe func not changed  avg. 0.9023428 | max. 0.9050916
# beds restored & numeric deposit   avg. 0.9023201 | max. 0.903464
# beds, bedrooms, bathrooms, claning fee na flagged ->
#                                   avg. 0.9027178 | max. 0.9069918
# amenities count added             avg. 0.9029674 | max. 0.9054379
# ref                               avg. 0.902069  | max. 0.9042029
# word count desc and summary add   avg. 0.9027984  | max. 0.9047979
# word count desc and summary add   avg. 0.9021937  | max. 0.9034472
# cleaning fee na imputed           avg. 0.9021937  | max. 0.9034472
# crib added                        avg. 0.9023794  | max. 0.9037946


# cross valid log part 3 --------------
# ref                               avg. 0.9025307 | max. 0.904336
# single xgb amenities_na removed   avg. 0.9026497 | max. 0.9056097
# single xgb with bin cluster       avg. 0.9030928 | max. 0.9067885
# single xgb with live score        avg. 0.9025392 | max. 0.9061712
# single xgb without live score     avg. 0.9025405 | max. 0.9060061
# single xgb tuned                  avg. 0.9030541 | max. 0.9049797
# single xgb tuned no base score    avg. 0.9038776 | max. 0.9070992
# ensemble xgb                      avg. 0.9055914 | max. 0.9086199
# ensemble xgb latest               avg. 0.905661  | max. 0.9075042
# ensemble xgb cv min pred          avg. 0.9013268 | max. 0.9034421
# single xgb host pic and host veri avg. 0.9034269 | max. 0.9052684
# single xgb host veri only         avg. 0.9032971 | max. 0.905124
# single xgb no host veri           avg. 0.9029503 | max. 0.9048014



# training ridge ---------------------
md_hbr_ridge <- glmnet(
  x = x_for_hbr_tr,
  y = hbr_tr,
  parallel= T,
  lambda = 1e-7,
  family = 'binomial',
  alpha  = 1,
  weights = ifelse(hbr_tr == 1, 1, 1)
)
pred_ridge_hbr = predict(md_hbr_ridge, x_for_hbr_va)
get_auc(pred_ridge_hbr, hbr_va)
# 0.8661152
length(pred_ridge_hbr) == length(xgb_x_pred)
pred_avg = (pred_ridge_hbr + xgb_x_pred) / 2
get_auc(pred_avg, hbr_va)


# complexity analysis ridge hbr ------------
# optimal lambda 1e-7
# optimal class weight 1
plot_complexity_curve(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  vec_param1 = 1:5 * 0.2, 
  trainer = \(x,y,p1) {
    return(
      glmnet(
        x = x,
        y = y,
        parallel= T,
        lambda = 1e-7,
        family = 'binomial',
        alpha  = 1,
        weights = ifelse(y == 1, 1, 1)
      )
    )
  },
  predictor = predict,
  measurer = get_auc,
  n_per_round = 2
)




# training ranger hbr ----------------
# class weights helps
ncol(x_for_hbr_tr)
# pn_rate = sum(hbr_tr == 1) / sum(hbr_tr != 1)
md_ranger_hbr <- ranger(
  x = x_for_hbr_tr, y = hbr_tr,
  max.depth = 9,
  mtry = ncol(x_for_hbr_tr),
  num.trees = 500,
  min.bucket = 10,
  importance = 'impurity',
  probability = T,
  class.weights = c(1, 1.2),
  num.threads = 12
)
ranger_hbr_pred <- predict(md_ranger_hbr, x_for_hbr_va)$predictions[,2]
get_auc(ranger_hbr_pred, hbr_va)
# 0.8573872 \\ 0.8707894 depth from 10 to 15
# 0.8661867 min.bucket 10 cls weight n > p
# 0.8672256 cls weight n < p
# 0.8696877 (increase depth and ntree)
# 0.8526587


# complexity analysis ranger hbr ------------
# optimal n tree appears around 500
# optimal max.depth appears to be 26
# case weight appears to have no significant use
plot_complexity_curve(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  vec_param1 = 5:8 * 100, 
  trainer = \(x,y,p1) {
    return(
      ranger(
        x = x, y = y,
        max.depth = 26,
        # mtry = ncol(x),
        num.trees = p1,
        importance = 'impurity',
        probability = T,
        num.threads = 12
      )
    )
  },
  predictor = \(m,x) {
    return(
      predict(m, x)$predictions[,2]
    )
  },
  measurer = get_auc,
  n_per_round = 2
)


# linear combination of models ---------------
# optimal w for ridge is 0.04
vec_auc = c()
for (w in 1:99 / 100) {
  obj_pred = w * pred_ridge_hbr + (1 - w) * xgb_x_pred
  obj_pred = get_auc(obj_pred, hbr_va)
  vec_auc = c(vec_auc, obj_pred)
}
which.max(vec_auc)
max(vec_auc)



# nlp analysis 2




# training naive bayes ---------------
# md_nb_hbr <- naiveBayes(
#     x = x_for_hbr_tr, y = hbr_tr
#   )
# pred_nb_hbr = predict(md_nb_hbr, x_for_hbr_va, type = 'raw')
# get_auc(pred_nb_hbr[,1], hbr_va)

# final prediction -----------
md_final_hbr_xgb <- xgboost(
  data = rbind(x_for_hbr_tr, x_for_hbr_va),
  label = ifelse(c(hbr_tr, hbr_va) == 1, 1, 0),
  max.depth = 7,
  eta = 0.09,
  nrounds = 500,
  objective = "binary:logistic",
  eval_metric = "auc",
  verbose = T,
  print_every_n = 100,
  nthread = 14
)
xgb_final_hbr_pred <- predict(md_final_hbr_xgb, x_for_hbr_dum_te)
write.table(xgb_final_hbr_pred, "high_booking_rate_group5_compare.csv", row.names = FALSE)


# training prs xgb -----------------
# colnames(x_for_hbr_tr)
md_xgb_prs <- xgboost(
  data = x_for_hbr_tr,
  label = ifelse(prs_tr == 1, 1, 0),
  max.depth = 9,
  eta = 0.11,
  nrounds = 325,
  objective = "binary:logistic",
  verbose = F,
  weight = ifelse(prs_tr == 1, 1.5, 1),
  print_every_n = 100,
  nthread = 12,
  eval_metric = "auc",
  base_score = mean(ifelse(prs_tr == 1, 1, 0))
)
pred_prs_xgb <- predict(md_xgb_prs, x_for_hbr_va)
get_auc(pred_prs_xgb, prs_va)
# 0.7 ratio
# 0.7999422 \\ 0.7970059
get_cutoff_dataframe(pred_prs_xgb, prs_va, step = 0.001, max_fpr = 0.095) %>%
  plot_cutoff_dataframe()



# training prs ranger ----------------
# class weights helps
# pn_rate = sum(prs_tr == 1) / sum(prs_tr != 1)
md_ranger_prs <- ranger(
  x = x_for_hbr_tr, y = prs_tr,
  max.depth = 26,
  num.trees = 800,
  min.bucket = 10,
  importance = 'impurity',
  probability = T,
  class.weights = c(1, 1.5),
  num.threads = 12
)
pred_ranger_prs <- predict(
  md_ranger_prs, x_for_hbr_va
  )$predictions[,2]
get_auc(pred_ranger_prs, prs_va)


get_cutoff_dataframe(pred_ranger_prs, prs_va, step = .001) %>%
  plot_cutoff_dataframe()


# complexity analysis ranger prs ------------
plot_complexity_curve(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(prs_tr, prs_va),
  vec_param1 = 0:5 * 200 + 1000, # optimal 
  trainer = \(x,y,p1) {
    return(
      ranger(
        x = x, y = y,
        max.depth = 18,
        num.trees = p1,
        min.bucket = 10,
        importance = 'impurity',
        probability = T,
        class.weights = c(1, 1 / pn_rate),
        num.threads = 12
      )
    )
  },
  predictor = \(y1,y2) {
    return(
      predict(y1, y2)$predictions[,2]
    )
  },
  measurer = get_auc,
  n_per_round = 3
)


# amenities unigram analysis -------------
dtm_am <- get_dtm(
  text_col = x$amenities, 
  custom_stop_words = 'will',
  ngram = c(1L,1L)
)

dtm_am_train = dtm_am[1:train_length, ]
dtm_am_te = dtm_am[(train_length + 1): nrow(dtm_am_train), ]
dtm_am_train_filtered = dtm_am_train[-ind_outliers,]
dtm_am_tr = dtm_am_train[-ind_outliers,][ind_sample, ]
dtm_am_va = dtm_am_train[-ind_outliers,][-ind_sample, ]

df_am = data.frame(
  am = x$amenities[1:train_length],
  hbr = hbr,
  prs = prs
  
) %>%
  mutate(
    laptop = grepl(pattern = 'laptop', x = am, ignore.case = T),
    computer = grepl(pattern = 'computer', x = am, ignore.case = T),
    laptop_or_computer = grepl(pattern = 'computer|laptop', x = am, ignore.case = T),
  )


# computer appears useless but laptop is slightly higher in positive instances
df_am %>%
  group_by(prs) %>%
  summarise(
    inst = n(),
    laptop_rate = mean(laptop),
    computer_rate = mean(computer),
    laptop_or_computer = mean(laptop_or_computer),
  )


df_am %>%
  group_by(hbr) %>%
  summarise(
    inst = n(),
    laptop_rate = mean(laptop),
    computer_rate = mean(computer),
  )
  


# amenities ridge -----------
md_am_ridge = glmnet(
  x = dtm_am_tr, y = hbr_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 1e-8,
  weights = ifelse(hbr_tr == 1, 1.6, 1)
)
predict(md_am_ridge, dtm_am_va, type = 'response') %>%
  get_auc(hbr_va)


get_vip_dataframe(md_am_ridge, dtm_am_tr)
vip(md_am_ridge, 50)


# amenities xgb -----------
md_am_xgb <- xgboost(
  data = dtm_am_tr,
  label = ifelse(hbr_tr == 1, 1, 0),
  eval_metric = 'auc',
  nrounds = 150,
  eta = 0.11,
  max.depth = 8,
  print_every_n = 100,
  objective = "binary:logistic",
  nthread = 12,
  weight = ifelse(hbr_tr == 1, 1.6, 1)
)
predict(md_am_xgb, dtm_am_va) %>%
  get_auc(hbr_va)
vip(md_am_xgb, 50)


# transit unigram analysis -------------
dtm_transit <- get_dtm(
  text_col = x$transit, 
  custom_stop_words = 'will',
  ngram = c(1L,1L)
)

dtm_transit_train = dtm_transit[1:train_length, ]
dtm_transit_te = dtm_transit[(train_length + 1): nrow(dtm_transit_train), ]
dtm_transit_train_filtered = dtm_transit_train[-ind_outliers,]
dtm_transit_tr = dtm_transit_train[-ind_outliers,][ind_sample, ]
dtm_transit_va = dtm_transit_train[-ind_outliers,][-ind_sample, ]

 data.frame(
   transit = x$transit[1:train_length],
  hbr = hbr, prs = prs
) %>%
  mutate(
    na = is.na(transit),
    manhattan = grepl(pattern = 'manhattan',transit),
    downtown = grepl(pattern = 'downtown', transit),
    station = grepl(pattern = 'station', transit),
    walk = grepl(pattern = 'walk', transit),
    free = grepl(pattern = 'free', transit)
  ) %>%
  group_by(hbr) %>%
  summarise(
    inst = n(),
    na = mean(na),
    manhattan = mean(manhattan),
    downtown = mean(downtown),
    station = mean(station),
    walk = mean(walk),
    free = mean(free)
  )

# transit ridge -----------
# md_transit_ridge = glmnet(
#   x = dtm_transit_tr, y = hbr_tr,
#   family = 'binomial',
#   alpha = 1,
#   lambda = 1e-7,
#   weights = ifelse(hbr_tr == 1, 1.6, 1)
# )
# predict(md_transit_ridge, dtm_transit_va, type = 'response') %>%
#   get_auc(hbr_va)
# 
# 
# get_vip_dataframe(md_transit_ridge, dtm_transit_tr)
# vip(md_transit_ridge, 50)


# transit xgb -----------
md_transit_xgb <- xgboost(
  data = dtm_transit_tr,
  label = ifelse(hbr_tr == 1, 1, 0),
  eval_metric = 'auc',
  nrounds = 150,
  eta = 0.11,
  max.depth = 8,
  print_every_n = 100,
  objective = "binary:logistic",
  nthread = 12,
  weight = ifelse(hbr_tr == 1, 1.6, 1)
)
predict(md_transit_xgb, dtm_transit_va) %>%
  get_auc(hbr_va)
vip(md_transit_xgb, 50)



# summary bigram analysis -------------
dtm_sum <- get_dtm(
  text_col = x$summary, 
  custom_stop_words = 'will',
  ngram = c(1L,2L)
)

dtm_sum_train = dtm_sum[1:train_length, ]
dtm_sum_te = dtm_sum[(train_length + 1): nrow(dtm_sum_train), ]
dtm_sum_train_filtered = dtm_sum_train[-ind_outliers,]
dtm_sum_tr = dtm_sum_train[-ind_outliers,][ind_sample, ]
dtm_sum_va = dtm_sum_train[-ind_outliers,][-ind_sample, ]


df_sum = data.frame(
  summary = x$summary[1:train_length],
  hbr = hbr,
  prs = prs

) %>%
  mutate(
    private = grepl(pattern = 'private', x = summary, ignore.case = T),
    free = grepl(pattern = 'free', x = summary, ignore.case = T)
  ) %>%
  group_by(hbr) %>%
  summarise(
    count = n(),
    private = mean(private),
    free = mean(free)
  )

# summary ridge -----------
md_am_ridge = glmnet(
  x = dtm_am_tr, y = hbr_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 1e-8,
  weights = ifelse(hbr_tr == 1, 1.6, 1)
)
predict(md_am_ridge, dtm_am_va, type = 'response') %>%
  get_auc(hbr_va)


get_vip_dataframe(md_am_ridge, dtm_am_tr)
vip(md_am_ridge, 50)


# summary xgb -----------
md_sum_xgb <- xgboost(
  data = dtm_sum_tr,
  label = ifelse(hbr_tr == 1, 1, 0),
  eval_metric = 'auc',
  nrounds = 150,
  eta = 0.11,
  max.depth = 8,
  print_every_n = 100,
  objective = "binary:logistic",
  nthread = 12,
  weight = ifelse(hbr_tr == 1, 1.6, 1)
)
predict(md_sum_xgb, dtm_sum_va) %>%
  get_auc(hbr_va)
vip(md_sum_xgb, 50)



# manual ensemble -------------------

# list trainer declaration -----------
list_trainer = list(
  # xgb ---------------------
  'xgb' = \(x,y) {
    return(
      xgboost(
        data = x,
        label = ifelse(y == 1, 1, 0),
        max.depth = 11,
        eta = 0.11,
        nrounds = 325,
        objective = "binary:logistic",
        verbose = T,
        weight = ifelse(y == 1, 1.5, 1),
        print_every_n = 100,
        nthread = 12,
        eval_metric = "auc",
        base_score = mean(ifelse(y == 1, 1, 0))
      )
    )
  },
  
  # ranger -------------------
  'ranger' = \(x,y) {
    return(
      ranger(
        x = x, y = y,
        max.depth = 26,
        num.trees = 500,
        importance = 'impurity',
        probability = T,
        num.threads = 12
      )
    )
  }
  
  # ridge ---------------
  # 'ridge' = \(x,y) {
  #   return(
  #     glmnet(
  #       x = x,
  #       y = y,
  #       parallel= T,
  #       lambda = 1e-7,
  #       family = 'binomial',
  #       alpha  = 1,
  #     )
  #   )
  # }
)



# list predictor declaration ----------
list_predictor = list(
  'xgb' = predict,
  'ranger' = \(m,x) {
    return(
      predict(m, x)$predictions[,2]
    )
  }
  # 'ridge' = predict
)

# train ensemble of different models ----------
md_ensemble_hbr = joint_train(
  x = x_for_hbr_tr, y = hbr_tr,
  vec_model_names = c('xgb', 'ranger'),
  list_trainer = list_trainer
)

joint_pred_prob = joint_predict(
  x = x_for_hbr_va,
  list_model = md_ensemble_hbr,
  list_predictor = list_predict,
  pred_aggregator = 'mean' # mean better than min or max
)
get_auc(joint_pred_prob, hbr_va)





# manual heterogenous ensemble cross val -----------------
me_cv = cross_val(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  
  trainer = \(x,y) {
    return(
      joint_train(
        x, y, 
        list_trainer = list_trainer
      )
    )
  },
  
  predictor = \(m,x) {
    return(
      joint_predict(
        x = x,
        list_model = m,
        list_predictor = list_predictor,
        pred_aggregator = 'mean'
      )
    )
  },
  
  measurer = get_auc
)
me_cv %>% mean()
me_cv %>% max()
# avg. 0.8783107 | max. 0.8804193


# training xgb ensemble ------------
md_xgb_esb_hbr = single_ensemble(
  x = x_for_hbr_tr, 
  y = hbr_tr,
  ensemble_size = 10,
  trainer = \(x,y) {
    return(
      xgboost(
        data = x,
        label = ifelse(y == 1, 1, 0),
        max.depth = 7,
        eta = 0.09,
        nrounds = 500,
        objective = "binary:logistic",
        verbose = T,
        print_every_n = 500,
        nthread = 12,
        eval_metric = "auc"
      )
    )
  }
)


pred_xgb_esb_hbr = joint_predict(
  x_for_hbr_va,
  list_model = md_xgb_esb_hbr,
  list_predictor = list(predict)
)


get_auc(pred_xgb_esb_hbr, hbr_va)
# 0.9046163 \\ 0.9056033 \\ 0.9058013


# complexity analysis xgb ensemble ---------------
plot_complexity_curve(
  n_per_round = 2,
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  vec_param1 = c(5, 10),
  trainer = \(x,y, p) {
    return(
      single_ensemble(
        x, y, 
        verbose = F,
        ensemble_size = p,
        trainer = \(x,y) {
          return(
            xgboost(
              data = x,
              label = ifelse(y == 1, 1, 0),
              max.depth = 8,
              eta = 0.11,
              nrounds = 250,
              objective = "binary:logistic",
              verbose = T,
              weight = ifelse(y == 1, 1.5, 1),
              print_every_n = 10000,
              nthread = 12,
              eval_metric = "auc",
              base_score = mean(ifelse(y == 1, 1, 0))
            )
          )
        }
          
      )
    )
  },
  
  predictor = \(m,x) {
    return(
      joint_predict(
        x = x,
        list_model = m,
        list_predictor = list(predict)
      )
    )
  },
  
  measurer = get_auc
)


# tuning ensemble xgb --------------
mat_res = matrix_search(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  vec_param1 = 0:4 * 2 + 10,
  vec_param2 = c(0.618, 0.8),
  trainer = \(x, y, p1, p2) {
    return(
      single_ensemble(
        x= x, y= y,
        ensemble_size = p1,
        sample_prop = p2,
        trainer = \(x,y) {
          return(
            xgboost(
              data = x,
              label = ifelse(y == 1, 1, 0),
              max.depth = 8,
              eta = 0.11,
              nrounds = 250,
              objective = "binary:logistic",
              verbose = T,
              weight = ifelse(y == 1, 1.5, 1),
              print_every_n = 100,
              nthread = 12,
              eval_metric = "auc",
              base_score = mean(ifelse(y == 1, 1, 0))
            )
          )
        }
      )
    )
  },
  predictor = \(m,x) {
    joint_predict(list_model = m, x = x, list_predictor = list(predict))
  },
  measurer = get_auc
)

# cross valid xgb ensemble ---------------
cv_xgb_esb = cross_val(
  x = rbind(x_for_hbr_tr, x_for_hbr_va),
  y = c(hbr_tr, hbr_va),
  fold_count = 3,
  trainer = \(x,y) {
    return(
      single_ensemble(
        x = x, y = y,
        verbose = F,
        ensemble_size = 10,
        trainer = \(x,y) {
          return(
            xgboost(
              data = x,
              label = ifelse(y == 1, 1, 0),
              max.depth = 7,
              eta = 0.09,
              nrounds = 500,
              objective = "binary:logistic",
              verbose = F,
              nthread = 12,
              eval_metric = "auc"
            )
          )
        }
      )
    )
  },
  predictor = \(m,x) {
    return(
      joint_predict(
        m, x,
        list_predictor = list(predict),
        pred_aggregator = 'mean',
      )
    )
  },
  measurer = get_auc
)
cv_xgb_esb %>% mean()
cv_xgb_esb %>% min()
cv_xgb_esb %>% max()


# final xgb ensemble ------------
md_final_xgb_esb_hbr = single_ensemble(
  x = rbind(x_for_hbr_tr, x_for_hbr_va), 
  y = c(hbr_tr, hbr_va),
  ensemble_size = 10,
  trainer = \(x,y) {
    return(
      xgboost(
        data = x,
        label = ifelse(y == 1, 1, 0),
        max.depth = 7,
        eta = 0.09,
        nrounds = 500,
        objective = "binary:logistic",
        verbose = T,
        print_every_n = 500,
        nthread = 12,
        eval_metric = "auc"
      )
    )
  }
)


pred_final_xgb_esb_hbr = joint_predict(
  x_for_hbr_dum_te,
  list_model = md_final_xgb_esb_hbr,
  list_predictor = list(predict)
)


write.table(
    pred_final_xgb_esb_hbr, 
    "high_booking_rate_group5.csv",
    row.names = FALSE
  )


