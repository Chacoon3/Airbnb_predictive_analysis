#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
#install.packages('tidyverse')
library(tidyverse)
options(scipen=999)
source('/cloud/project/Library/data_cleaning.r')
get_cleaned('/cloud/project/Data')
install.packages('quanteda')
data1 <-read_csv('x_train_clean.csv')
summary(data1)
data2<-data1 %>%
  mutate(
    bed_type=as.factor(bed_type),
    cancellation_policy=as.factor(cancellation_policy),
    city=as.factor(city),
    city_name=as.factor(city_name),
    country=as.factor(country),
    country_code=as.factor(country_code),
    host_neighbourhood=as.factor(host_neighbourhood),
    host_response_time=as.factor(host_response_time),
    market=as.factor(market),
    neighbourhood=as.factor(neighbourhood),
    property_type=as.factor(property_type),
    room_type=as.factor(room_type),
    smart_location=as.factor(smart_location),
    state=as.factor(state),
    zipcode=as.factor(zipcode),
    property_category=as.factor(property_category),
    host_acceptance=as.factor(host_acceptance),
    host_response=as.factor(host_response)
  )%>%
  select(-host_verifications)
summary(data2)
airbnb<-data2%>%
  mutate(time_since_first_review= difftime('2023-01-01',first_review, unit='days'),
         time_since_first_review=as.duration(time_since_first_review)/dyears(x=1),
         time_host_since= difftime('2023-01-01',host_since, unit='days'),
         time_host_since=as.duration(time_host_since)/dyears(x=1),
         host_local= ifelse(as.character(host_neighbourhood)==as.character(neighbourhood), TRUE, FALSE)
         )

summary(airbnb)