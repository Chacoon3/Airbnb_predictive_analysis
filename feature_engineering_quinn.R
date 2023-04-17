# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file.
source('Library\\data_cleaning.r')
source('Library\\utils.r')

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



### QUINN'S FEATURE ENGINEERING VARIABLES:
#access
#accommodates
#amenities
#availability_30
#availability_365
#availability_60
#availability_90
#bathrooms
#bed_type
#bedrooms
#beds
#cancellation_policy
#city
#city_name



###EXPLORING DATA, LIGHT CLEANING, SUBSETTING RELEVANT COLUMNS


summary(x_test)

colnames(x_test)

x_test_small_1 <- x_test %>%
  select(2:14) %>% 
  mutate(bed_type = as.factor(bed_type),
         cancellation_policy = as.factor(cancellation_policy),
         city = as.factor(city),
         city_name = as.factor(city_name))

x_test_small_2 <- x_test %>%
  select(70:80)

x_test_small <- cbind(x_test_small_1,x_test_small_2)

summary(x_test_small)


###FEATURE ENGINEERING

head(x_test_small,5)

x_test_engineered <- x_test_small %>%
  mutate(bed_bath_ratio = bedrooms/bathrooms, #ratio of bedrooms to bathrooms
         people_per_bed = accommodates/beds, #people the property accommodates per bed
         availability_365_level = as.factor( #bins availability for next 365 days into high medium and low
           case_when(
             availability_365>200 ~ "HIGH",
             availability_365>100 ~ "MEDIUM",
             TRUE ~ "LOW"
           )
         )
         )

hist(x_test_small$availability_365)

head(x_test_engineered,5)
