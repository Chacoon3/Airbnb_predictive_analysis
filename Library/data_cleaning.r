source('Library\\utils.r')

library(tidyverse)
library(Metrics)
library(text2vec)
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

# function declarations 


# sentiment analysis block 
private_get_itoken <- function(text_col, tokenizer) {
  if (is.null(text_col)) {
    stop("column must not be null!")
  }
  
  return(
    itoken(
      text_col, 
      preprocessor = tolower,
      tokenizer = tokenizer,
      progressbar = FALSE
    )
  )
}


private_get_dfm <- function(text_col) {
  
  get_tokenizer <- function(v) {
    v %>%
      removeNumbers %>% #remove all numbers
      removePunctuation %>% #remove all punctuation
      removeWords(tm::stopwords(kind="en")) %>% #remove stopwords
      stemDocument %>%
      word_tokenizer 
  }
  
  obj_itoken <- text_col %>%
    # replace_na(replace = 'Missing')
    private_get_itoken(tokenizer = get_tokenizer)
  
  obj_vectorizer <- obj_itoken %>%
    create_vocabulary() %>%
    prune_vocabulary(vocab_term_max = 500) %>%
    vocab_vectorizer()
  
  dtm <- create_dtm(obj_itoken, obj_vectorizer)
  dfm <- as.dfm(dtm)
  return(dfm)
}


private_get_sent <- function(dfm) {
  bing_sent <- get_sentiments("bing")
  bing_negative <- bing_sent %>%
    filter(sentiment == 'negative')
  bing_positive <- bing_sent %>%
    filter(sentiment == 'positive')
  
  dict_snmt <- dictionary(list(negative = bing_negative$word, positive = bing_positive$word))
  count_of_snmt <- dfm_lookup(dfm, dict_snmt, valuetype = 'fixed')
  snmt <- convert(count_of_snmt, to = "data.frame") %>%
    mutate(sent_score = as.factor(
      case_when(
        positive > negative ~ 'Positive',
        positive == negative ~ 'Neutral',
        positive < negative ~ 'Negative'
      )
    )) %>%
    select(
      sent_score
    )
  
  rm(bing_sent, bing_negative, bing_positive, dict_snmt, count_of_snmt)
  
  return(snmt)
}


get_sent_score <- function(text_col) {
  res <- text_col %>%
    private_get_dfm() %>%
    private_get_sent()
  
  return(res$sent_score)
}
# end of sentiment analysis block


# get the shape of a dataframe
get_shape <- function(dataframe) {
  return(c(nrow(dataframe), ncol(dataframe)))
}


# get column-wise sum of na instances
count_col_na <- function(df) {
  res <- df %>%
    apply(2, \(col) {sum(is.na(col))})
  return(res)
}


get_accuracy <- function(y_pred, y_valid) {
  if (length(y_pred) != length(y_valid)) {
    stop("error: prediction and valid column has different numbers of rows!")
  }
  return(
    sum(y_pred == y_valid) / length(y_pred)
  )
}


# data cleaning sub routines 


dc_preprocess <- function(x_train, x_test, y_train) {
  print('data cleaning completed: 10% ...')
  # merge train and test
  data_all <- rbind(x_train, x_test) %>%
    data.frame(
      rbind(y_train, data.frame(
              perfect_rating_score = rep('placeholder',nrow(x_test)),
              high_booking_rate = rep('placeholder',nrow(x_test))
            )
        )
    ) %>%
  # drop rows having no target values
    filter(
      !is.na(high_booking_rate) &
      !is.na(perfect_rating_score)
    )
  return(data_all)
}


dc_sean <- function(dataframe) {
  print('data cleaning completed: 90% ...')
  
  # feature engineering 
  dataframe <- dataframe %>%
    mutate(
      access_snmt = get_sent_score(dataframe$access) %>% as.factor(),
      desc_snmt = get_sent_score(dataframe$description) %>% as.factor(),
      host_about_snmt = get_sent_score(dataframe$host_about) %>% as.factor(),
      house_rules_snmt = get_sent_score(dataframe$house_rules) %>% as.factor(),
      interaction_snmt = get_sent_score(dataframe$interaction) %>% as.factor(),
      neighborhood_snmt = get_sent_score(dataframe$neighborhood) %>% as.factor(),
      notes_snmt = get_sent_score(dataframe$notes) %>% as.factor(),
      summary_snmt = get_sent_score(dataframe$summary) %>% as.factor()
      
      
      # access_snmt = as.factor(dataframe$access_snmt),
      # desc_snmt = as.factor(dataframe$desc_snmt),
      # host_about_snmt = as.factor(dataframe$host_about_snmt),
      # house_rules_snmt = as.factor(dataframe$house_rules_snmt),
      # interaction_snmt = as.factor(dataframe$interaction_snmt),
      # neighborhood_snmt = as.factor(dataframe$neighborhood_snmt),
      # notes_snmt = as.factor(dataframe$notes_snmt),
      # summary_snmt = as.factor(dataframe$summary_snmt)
    ) %>%
    mutate(
      host_listings_count =  # 2023-4-4 fixed
        ifelse(is.na(dataframe$host_listings_count), median(dataframe$host_listings_count, na.rm = TRUE), dataframe$host_listings_count),
      host_since = # 2023-4-4 fixed
        ifelse(is.na(dataframe$host_since), get_mode(dataframe$host_since), dataframe$host_since),
      host_total_listings_count = # 2023-4-4 fixed
        ifelse(is.na(dataframe$host_total_listings_count), 
               median(dataframe$host_total_listings_count, na.rm = TRUE), dataframe$host_total_listings_count),
      host_response_time = # fixed 2023-4-5
        ifelse(is.na(dataframe$host_response_time), 
               get_mode(dataframe$host_response_time), dataframe$host_response_time),
      host_since = ifelse(
        is.na(dataframe$host_since), median(dataframe$host_since, na.rm = TRUE), dataframe$host_since
      ),
      city = as.factor(dataframe$city) # fixed 2023-4-6
    )

  return(dataframe)
}


dc_xiaoze <- function(dataframe) {
  print('data cleaning completed: 40% ...')
  
  #edit NA columns
  #H
  median_bathrooms <- median(dataframe$bathrooms, na.rm = TRUE)
  dataframe$bathrooms[is.na(dataframe$bathrooms)] <- median_bathrooms
  #J
  median_bedrooms <- median(dataframe$bedrooms, na.rm = TRUE)
  dataframe$bedrooms[is.na(dataframe$bedrooms)] <- median_bedrooms
  #K
  median_beds <- median(dataframe$beds, na.rm = TRUE)
  #dataframe$beds[is.na(dataframe$bathrooms)] <- median_beds
  dataframe$beds[is.na(dataframe$beds)] <- median_beds # 2023-4-4 fixed
  #M
  dataframe$city[is.na(dataframe$city)] <- "MISSING"
  #O
  dataframe$cleaning_fee <- parse_number(dataframe$cleaning_fee)
  dataframe$cleaning_fee <- ifelse(is.na(dataframe$cleaning_fee),0,dataframe$cleaning_fee)
  
  
  #I:Create a new factor: bed_category is "bed" if the bed_type is Real Bed and "other" otherwise
  dataframe$bed_category <- ifelse(dataframe$bed_type == "Real Bed", "Bed", "other")
  dataframe$bed_type <- as.factor(dataframe$bed_type)
  #L:For cancellation_policy, group "strict" and "super_strict_30" into "strict"
  dataframe$cancellation_policy <- ifelse(dataframe$cancellation_policy %in% c("strict","super_strict_30"),"strict",dataframe$cancellation_policy)
  dataframe$cancellation_policy <- as.factor(dataframe$cancellation_policy)
  
  return(dataframe)
}


dc_jingruo <- function(df) {
  print('data cleaning completed: 25% ...')
  
  #remove items relevant to translation error
  df$amenities <- gsub("translation missing: en\\.hosting_amenity_\\d+", "",df$amenities) 
  
  # get the name and frequency of each item
  item_counts <- table(str_extract_all(df$amenities, regex("[[:alnum:] ()'’‘-]+", ignore_case = T)) %>% unlist())
  #transform the table into a dataframe
  item_df <- data.frame(
    items = names(item_counts),
    frequency = as.numeric(item_counts),
    stringsAsFactors = FALSE
  )
  item_df
  #merge duplicated items 
  # Cable TV; Smart TV; TV --> TV
  # Doorman; Doorperson; Doorman Entry --> Doorperson
  # WiFi, internets, Wireless Internet
  # travel crib, crib -->crib
  # Hot tub, bath tub --> bath tub
  #make new rows sum up the frequency of duplicates
  new_row <- data.frame(items=c("TV","Internet"), frequency = c(sum(item_df[item_df$item %in% c("TV", "Smart TV","Cable TV"), "frequency"]),sum(item_df[item_df$item %in% c("Wifi", "Internets", "Wireless Internet"), "frequency"])))
  #drop the previous rows
  item_df <-item_df[-which(item_df$items %in% c("TV", "Smart TV","Cable TV", "Wifi", "Internet", "Wireless Internet")),]
  #add new rows
  item_df <- rbind(item_df, new_row)
  item_df <- item_df[order(item_df$frequency, decreasing = TRUE),] #sort the dataframe by frequency in descending order
  item_df
  
  #unify the names of TV and internet
  df$amenities <- gsub("Smart TV|Cable TV","TV",df$amenities)
  df$amenities <-gsub("Wifi|Wireless Internet","Internet",df$amenities)
  
  #Subset the top 10 items and build corresponding dummy variables
  subset_items <- item_df$items[1:10] 
  for (i in subset_items){
    df[,i] <- str_detect(df$amenities, i)
    is.na(df[,i]) <- 0 
  }
  
  #count how many items are included in each row
  df$amentities_number <- sapply(str_split(df$amenities, pattern = ","),length)
  
  #drop the original amemtities row
  df$amenities <- NULL
  
  #get the name and frequency of each verification
  verifications <- df$host_verifications %>% 
    strsplit(",") %>%
    str_extract_all(regex("[[A-Za-z] \\-_]+",ignore_case = T)) %>% 
    unlist() %>% 
    unique()
  verifications
  #remove wrong results
  verifications <- verifications[-which(verifications %in% c("c", " "))]
  #create dummy variables
  for (i in verifications){
    df[,i] <- str_detect(df$host_verifications, i)
  }
  #delete original column
  df$df <- NULL
  
  return(df)
}


dc_johannah <- function(dataframe) {
  print('data cleaning completed: 55% ...')
  
  attach(dataframe)
  res <- dataframe %>%
    mutate(
      has_interaction= ifelse(is.na(interaction), 0, 1),
      is_business_travel_ready= ifelse(is.na(is_business_travel_ready), FALSE, is_business_travel_ready),
      jurisdiction_names=as.factor(jurisdiction_names),
      market = ifelse(is.na(market), 'Missing', market),
      market = as.factor(market),
      monthly_price = parse_number(monthly_price, na = c("", "NA")),
      monthly_price = ifelse(is.na(monthly_price),0,monthly_price), 
      neighbourhood=ifelse(is.na(neighbourhood), "Missing", neighbourhood),
      neighbourhood=as.factor(neighbourhood),
      is_note=ifelse(is.na(notes), 0,1),
      price = parse_number(price, na = c("", "NA")),
      property_type=ifelse(is.na(property_type), "Other",property_type),
      property_type=as.factor(property_type),
      property_category = ifelse(property_type %in% c('Apartment','Serviced apartment','Loft'), 'apartment', property_type),
      property_category = ifelse(property_type %in% c('Bed & Breakfast', 'Boutique hotel', 'Hostel'), 'hotel', property_category),
      property_category = ifelse(property_type %in% c('Townhouse', 'Condominium'), 'condo', property_category),
      property_category = ifelse( property_type %in% c('Bungalow', 'House'), 'house',property_category),
      property_category = ifelse( property_category %in% c('apartment','hotel','condo', 'house'), property_category, 'other'),
      property_category = as.factor(property_category),
      # fixed 2023-4-6
      #experiences_offered= ifelse(is.na(experiences_offered), 'none', experiences_offered),
      #experiences_offered=as.factor(experiences_offered),
      extra_people= parse_number(extra_people, na = c("", "NA")),
      extra_people=ifelse(is.na(extra_people), 0, extra_people),
      has_about=ifelse(is.na(host_about), 0,1),
      host_acceptance_rate = parse_number(host_acceptance_rate, na = c("", "NA")),
      host_acceptance_rate = # 2023-4-5 fixed
        ifelse(
          is.na(host_acceptance_rate), median(host_acceptance_rate, na.rm = TRUE), host_acceptance_rate
        ),
      host_acceptance= case_when(
        host_acceptance_rate == 100 ~ "ALL", 
        host_acceptance_rate < 100 ~ "SOME", 
        TRUE ~ "MISSING"), # fixed 2023-4-5
      host_acceptance=as.factor(host_acceptance),
      host_has_profile_pic = ifelse(is.na(host_has_profile_pic), FALSE, host_has_profile_pic),
      host_identity_verified =ifelse(is.na(host_identity_verified), FALSE, host_identity_verified),
      host_is_superhost=ifelse(is.na(host_is_superhost), FALSE, host_is_superhost),
      has_host_name= ifelse(is.na(host_name), 0,1),
      host_location=ifelse(is.na(host_location), "Missing", host_location),
      host_location=as.factor(host_location),
      host_neighbourhood=ifelse(is.na(host_neighbourhood), "Missing", host_neighbourhood),
      host_neighbourhood=as.factor(host_neighbourhood),
      host_response_rate = parse_number(host_response_rate, na = c("", "NA")),
      host_response_rate = # 2023-4-5 fixed
        ifelse(
          is.na(host_response_rate), median(host_response_rate, na.rm = TRUE), host_response_rate
        ),
      host_response=case_when(
        host_response_rate == 100 ~ "ALL", 
        host_response_rate < 100 ~ "SOME", 
        TRUE ~ "MISSING"), # fixed 2023-4-5
      host_response = as.factor(host_response),
      has_house_rules = ifelse(is.na(house_rules), 0, 1)
    )
  detach(dataframe)
  return(res)
}


dc_quinn <- function(dataframe) {
  print('data cleaning completed: 70% ...')
  
  
  attach(dataframe)
  res <- dataframe %>%
    mutate(
      state = ifelse(is.na(state), 'MISSING', state), # fixed 2023-4-6
      smart_location = ifelse(is.na(smart_location), 'MISSING', smart_location), # fixed 2023-4-6
      security_deposit=parse_number(security_deposit), #convert dollar to number
      weekly_price=parse_number(weekly_price), #convert dollar to number
      room_type=as.factor(room_type), #category variable
      smart_location=as.factor(smart_location), #category variable - may be automatically generated
      state=as.factor(state), #category variable
      #continuous variables replaced with mean, discrete with median
      security_deposit = ifelse(is.na(security_deposit), mean(security_deposit, na.rm = TRUE), security_deposit), #replace na's with mean
      square_feet = ifelse(is.na(square_feet), median(square_feet, na.rm = TRUE), square_feet), #replace na's with median
      weekly_price = ifelse(is.na(weekly_price), mean(weekly_price, na.rm = TRUE), weekly_price), #replace na's with mean
      zipcode=ifelse(is.na(zipcode),"MISSING",zipcode),
      zipcode=as.factor(zipcode),
    )
  detach(dataframe)
  return(res)
}


get_cleaned <- function(folder_dir) {
  print('initializing data cleaning ...')
  
  wd <- getwd()
  setwd(folder_dir)
  x_train <- read.csv('airbnb_train_x_2023.csv')
  x_test <- read.csv('airbnb_test_x_2023.csv')
  y_train <- read.csv('airbnb_train_y_2023.csv')
  setwd(wd)
  
  
  row_train <- nrow(x_train)
  row_test <- nrow(x_test)
  df <- dc_preprocess(x_train, x_test, y_train) %>%
    dc_jingruo() %>%
    dc_xiaoze() %>%
    dc_johannah() %>%
    dc_quinn() %>%
    dc_sean()
  
  row_total <- nrow(df)
  assert_row_count = row_total == row_train + row_test
  if (!assert_row_count) {
    stop("Error: Row count of output dataframe different from the sum of train and test dataframe!")
  }
  
  colwise_na_count = count_col_na(df)
  if (any(colwise_na_count)) {
    warning(c(
      "Missing values in output dataframe! Columns are:\n",
      paste(names(df)[which(colwise_na_count > 0)], collapse = ', ')
    )
    )
  }
  
  x <- df %>%
    select(!c(high_booking_rate, perfect_rating_score))
  
  
  print('data cleaning completed!')
  return(x)
}


export_cleaned <- function(folder_dir) {
  x_all <- get_cleaned(folder_dir)
  
  wd <- getwd()
  setwd(folder_dir)
  y_train <- read.csv('airbnb_train_y_2023.csv')
  row_train <- nrow(y_train)
  x_train_clean <- x_all[1:row_train, ]
  x_test_clean <- x_all[(row_train + 1): nrow(x_all), ]
  
  write.csv(x_train_clean, file = paste(
    folder_dir, 'x_train_clean.csv', sep = '\\'
  ))
  
  write.csv(x_test_clean, file = paste(
    folder_dir, 'x_test_clean.csv', sep = '\\'
  ))
  
  setwd(wd)
  
  print('export completed!')
}


# garbage comments below just don't read'em 
# kept just in case some day I need'em again


# sample code for function call
# t = r"(C:\Users\Chaconne\Documents\学业\UMD\Courses\758T Predictive\785T_Pred_Assignment\GA\Airbnb_predictive_analysis\Data)"
# export_cleaned(t, t)
# 
# x_tr_cl <- read.csv('x_train_clean.csv')
# x_te_cl <- read.csv('x_test_clean.csv')
# nrow(x_tr_cl) == nrow(x_train)
# nrow(x_te_cl) == nrow(x_test)


# data cleaning sample code
# data_all <- data_cleaning(r"(C:\Users\Chaconne\Documents\学业\UMD\Courses\758T Predictive\785T_Pred_Assignment\GA\Airbnb_predictive_analysis\Data)")


# 2. Check if there are still abnormal values across the columns
# colwise_na_count <- count_col_na(data_all)
# any(colwise_na_count) # TURE if there are missing values in any columns
# data_all %>%
#   select(c(high_booking_rate, perfect_rating_score)) %>%
#   apply(2, FUN = \(col) {sum(is.na(col))})


# y_train <- df_train %>%
#   select(c(perfect_rating_score, high_booking_rate))
# df_train <- df_train %>%
#   select(!c(perfect_rating_score, high_booking_rate))
# 
# 
# row_test = sum(ifelse(data_all$perfect_rating_score == 'placeholder',1 ,0))
# row_total = nrow(data_all)
# row_train = row_total - row_test
# 
# # all the training features
# x_train = data_all[1:row_train, ] %>%
#   select(!c(high_booking_rate, perfect_rating_score))
# # all the testing features
# x_test <- data_all[(row_train + 1):row_total, ] %>%
#   select(!c(high_booking_rate, perfect_rating_score))
# # high booking rate
# y_train_hbr <- data_all[1:row_train, "high_booking_rate"] %>%
#   as.factor()
# # prefect rating score
# y_train_prs = data_all[1:row_train, "perfect_rating_score"] %>%
#   as.factor()
# # check if all features are clean
# x_train %>% 
#   apply(2, FUN = \(col) {sum(is.na(col))})


# testing
# get_shape(x_train)
# get_shape(x_test)
# length(y_train_hbr)
# length(y_train_prs)
# any(is.na(y_train_hbr))
# any(is.na(y_train_prs))
# any(count_col_na(x_train))

# x_train$property_category
# x_train$bed_category
# x_test$property_category
# x_test$bed_category
# length(x_test$accommodates)
# the cleaned objects are: x_train, x_test, y_train_hbr, and y_train_prs
# build your models based on these four variables.
# you can do any transformations on the features as long as you consider it to
# be helpful to improve performance
