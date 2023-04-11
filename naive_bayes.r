# This line references the data_cleaning.r script so that you can call functions that are written in that r file. 
# Libraries called in the referenced file will automatically be included in this file.
source('data_cleaning.r')

# replace the string with the folder directory where you store the original data files.
folder_dir = r"(C:\Users\Chaconne\Documents\学业\UMD\Courses\758T Predictive\785T_Pred_Assignment\GA\Airbnb_predictive_analysis\Data)"

# This line needs only to be run once, which exports two csv files, one for training X's, the other for testing X's. Once the two files were created you only need to run the read.csv statements following this line.
export_cleaned(folder_dir)


# read
x_train <- read.csv('x_train_clean.csv')
x_test <- read.csv('x_test_clean.csv')
y_train <- read.csv('airbnb_train_y_2023.csv')
hbr <- y_train$high_booking_rate
prs <- y_train$perfect_rating_score

# test
nrow(x_train) == nrow(y_train)
nrow(x_test) == 12205
(length(hbr) == length(prs)) && (length(hbr) == nrow(y_train))



# codes start here ----------------------------