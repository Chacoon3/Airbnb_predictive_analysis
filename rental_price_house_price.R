#load packages
#install.packages('httr')
#install.packages('jsonlite')
library(httr)
library(jsonlite)

#import data
source('Library//data_cleaning.r')
source('Library//utils.r')
library(xgboost)
library(ranger)
library(lubridate)

# replace the string with the directory of the project folder "Data"
# the original datasets MUST be placed under the Data folder for the following script to run correctly.
folder_dir = r"(/Users/nuomihan/Desktop/758T group project/airbnb/Data)"



# This line needs only to be run once, which exports two csv files, one for the training X's, the other for the testing X's. Once the two files were created you only need to run the read.csv statements under this line.
# export_cleaned(folder_dir)


# read
# x_train <- read.csv('Data\\x_train_clean.csv')
# x_test <- read.csv('Data\\x_test_clean.csv')
x <- get_cleaned(folder_dir, FALSE)

############get the average price of listings################
#get zipcode and state in training set
address = data.frame(x$zipcode,x$state)

#some zipcodes have 9 digit, remove the last 4 digit
address$x.zipcode = substr(address$x.zipcode,1,5)
address$x.zipcode = as.character(address$x.zipcode)

#get unique zipcodes
address = address[!duplicated(address$x.zipcode),]

#check the number of unique zipcode
nrow(address)

# Set the API endpoint and API key
endpoint <- "https://api.mashvisor.com/v1.1/client/trends/listing-price"
api_key <- " "

# Build the request headers
headers <- c("x-api-key" = api_key)

# Make an API request for each neighborhood
for (i in 1:nrow(address)) {
  # Build the request payload with the current neighborhood
  payload <- list(
    state = address[i,'x.state'],
    zip_code = address[i,'x.zipcode'])

  # Make the API request for the current neighborhood
  response <- GET(endpoint, add_headers(headers), query = payload)
  # Check if the request was successful
  if (status_code(response) == 200) {
    # Parse the response JSON
    data <- httr::content(response, as= "text") %>% fromJSON()
    # Add the listing's price to the dataframe
    address[i, "avg_price"] <- ifelse(is.null(data$content$avg_price) , NA,data$content$avg_price)
    address[i, "avg_price_per_sqft"] <- ifelse(is.null(data$content$avg_price_per_sqft) , NA,data$content$avg_price_per_sqft)
    address[i, "median_price"] <- ifelse(is.null(data$content$median_price) , NA,data$content$median_price)
    address[i, "median_price_per_sqft"] <- ifelse(is.null(data$content$median_price_per_sqft) , NA,data$content$median_price_per_sqft)
  } 
}
address

#get market summary of short term rental

endpoint_market_summary = "https://api.mashvisor.com/v1.1/client/airbnb-property/market-summary"
for (i in 1:nrow(address)) {
  # Build the request payload with the current neighborhood
  payload <- list(
    state = address[i,'x.state'],
    zip_code = address[i,'x.zipcode'])
  
  # Make the API request for the current neighborhood
  response <- GET(endpoint_market_summary, add_headers(headers), query = payload)
  # Check if the request was successful
  if (status_code(response) == 200) {
    # Parse the response JSON
    data <- httr::content(response, as= "text") %>% fromJSON()
    # Add the listing's price to the dataframe
    address[i, "listings_count"] <- ifelse(is.null(data$content$listings_count) , NA,data$content$listings_count)
    address[i, "average_occupancy"] <- ifelse(is.null(data$content$occupancy_histogram$average_occupancy) , NA,data$content$occupancy_histogram$average_occupancy)
    address[i, "average_night_price"] <- ifelse(is.null(data$content$night_price_histogram$average_price) , NA,data$content$night_price_histogram$average_price)
    address[i, "average_rental_income"] <- ifelse(is.null(data$content$rental_income_histogram$average_rental_income) , NA,data$content$rental_income_histogram$average_rental_income)
  } 
}

#export data
write_csv(address,"rental_price_house_price.csv")

