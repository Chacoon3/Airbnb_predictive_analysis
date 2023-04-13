library(tidyverse)
library(Metrics)
library(text2vec)
library(SentimentAnalysis)
library(caret)
# library(tree)
library(class)
library(readr)
# library(e1071)
# library(naivebayes)
# library(ROCR)
# library(glmnet)


# function declarations 

# sample given number of rows from the dataframe
get_df_sample <- function(dataframe, sample_size) {
  # sample given number of rows from the original dataframe
  indice = 1:nrow(dataframe)
  sampled_indice <- sample(indice, sample_size, replace = FALSE)
  sampled_df <- dataframe[sampled_indice, ]
  
  return(sampled_df)
}


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
  # create columns indicating sentiment 
  # named after the original columns
  
  # first, select text columns which contains contents that may contain sentiment.
  # then map each of the text to a number that ranges from -1 to 1, indicating its sentiment
  
  # regular expression based approaches were not applied owing to a lack of target patterns.
  
  # local func: maps a column of text to a column of sentiment measure
  
  
  # feature engineering 
  dataframe <- dataframe %>%
    mutate(
      access_snmt = analyzeSentiment(dataframe$access)$SentimentQDAP,
      access_snmt = replace_na(access_snmt, 0),

      desc_snmt = analyzeSentiment(dataframe$description)$SentimentQDAP,
      desc_snmt = replace_na(desc_snmt, 0),

      host_about_snmt = analyzeSentiment(dataframe$host_about)$SentimentQDAP,
      host_about_snmt = replace_na(host_about_snmt, 0),

      house_rules_snmt = analyzeSentiment(dataframe$rules)$SentimentQDAP,
      house_rules_snmt = replace_na(house_rules_snmt, 0),

      interaction_snmt = analyzeSentiment(dataframe$interaction)$SentimentQDAP,
      interaction_snmt = replace_na(interaction_snmt, 0),

      neighborhood_snmt = analyzeSentiment(dataframe$neighborhood)$SentimentQDAP,
      neighborhood_snmt = replace_na(neighborhood_snmt, 0),

      notes_snmt = analyzeSentiment(dataframe$notes)$SentimentQDAP,
      notes_snmt = replace_na(notes_snmt, 0),

      summary_snmt = analyzeSentiment(dataframe$summary)$SentimentQDAP,
      summary_snmt = replace_na(summary_snmt, 0),
    ) %>%
    mutate(
      host_listings_count =  # 2023-4-4 fixed
        ifelse(is.na(dataframe$host_listings_count), median(dataframe$host_listings_count, na.rm = TRUE), dataframe$host_listings_count),
      host_since = # 2023-4-4 fixed
        ifelse(is.na(dataframe$host_since), mode(dataframe$host_since), dataframe$host_since),
      host_total_listings_count = # 2023-4-4 fixed
        ifelse(is.na(dataframe$host_total_listings_count), 
               median(dataframe$host_total_listings_count, na.rm = TRUE), dataframe$host_total_listings_count),
      host_response_time = # fixed 2023-4-5
        ifelse(is.na(dataframe$host_response_time), 
               mode(dataframe$host_response_time), dataframe$host_response_time),
      host_since = ifelse(
        is.na(dataframe$host_since), median(dataframe$host_since, na.rm = TRUE), dataframe$host_since
      ),
      city = as.factor(city) # fixed 2023-4-6
    ) %>%
    # column dropping, some are dropped temporarily, some are permanently
    select(!c(
      experiences_offered, # should drop cuz it is monotonous
      access,
      description,
      host_about,
      host_name,
      house_rules,
      interaction,
      name,
      neighborhood_overview,
      notes,
      space,
      street,
      summary,
      transit,
      jurisdiction_names,
      license
    ))
  
  return(dataframe)
}


dc_xiaoze <- function(dataframe) {
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
  wd <- getwd()
  setwd(folder_dir)
  y_train <- read.csv('airbnb_train_y_2023.csv')
  x_train <- read.csv('airbnb_train_x_2023.csv')
  x_test <- read.csv('airbnb_test_x_2023.csv')
  setwd(wd)
  
  row_train <- nrow(x_train)
  row_test <- nrow(x_test)
  
  data_all <- dc_preprocess(x_train, x_test, y_train) %>%
    dc_jingruo() %>%
    dc_xiaoze() %>%
    dc_johannah() %>%
    dc_quinn() %>%
    dc_sean()
  
  row_total = nrow(data_all)
  
  assert_row_count = row_total == row_train + row_test
  if (!assert_row_count) {
    stop("Error: Row count of output dataframe different from the sum of train and test dataframe!")
  }

  colwise_na_count = count_col_na(data_all)
  if (any(colwise_na_count)) {
      warning(paste(
        "Warning: Missing values in output dataframe! Columns are:",
        names(data_all)[which(colwise_na_count > 0)]
      )
    )
  }
  
  return(data_all)
} 


export_cleaned <- function(folder_dir) {
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
  
  row_total = nrow(data_all)
  assert_row_count = row_total == row_train + row_test
  if (!assert_row_count) {
    stop("Error: Row count of output dataframe different from the sum of train and test dataframe!")
  }
  
  colwise_na_count = count_col_na(data_all)
  if (any(colwise_na_count)) {
    warning(paste(
      "Warning: Missing values in output dataframe! Columns are:",
      names(data_all)[which(colwise_na_count > 0)]
      )
    )
  }
  
  x <- data_all %>%
    select(!c(high_booking_rate, perfect_rating_score))
  x_train_clean <- x[1:row_train, ]
  x_test_clean <- x[(row_train + 1): row_total, ]
  
  write.csv(x_train_clean, file = paste(
    folder_dir, 'x_train_clean.csv', sep = '\\'
  ))
  
  write.csv(x_test_clean, file = paste(
    folder_dir, 'x_test_clean.csv', sep = '\\'
  ))
}

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