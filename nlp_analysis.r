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


folder_dir = r"(C:\Users\Chaconne\Documents\学业\Projects\Airbnb_predictive_analysis_copy\Data)"
x <- get_cleaned(folder_dir, F)
wd <- getwd()
setwd(folder_dir)
y_train <- read.csv('airbnb_train_y_2023.csv')
y_train <- y_train %>%
  mutate(
    high_booking_rate = 
      ifelse(high_booking_rate== 'YES', 1, 0) %>% as.factor(),
    perfect_rating_score = 
      ifelse(perfect_rating_score== 'YES', 1, 0) %>% as.factor(),
  )
setwd(wd)


get_dtm <- function(
    text_col, 
    ngram = c(1L, 2L), 
    doc_prop_min = 0,
    doc_prop_max = 1,
    tf_idf = T
    ) {
  
  # inner function
  replace_punctuations <- function(text_col) {
    textcol <- text_col %>% gsub(
      pattern = r"(\{|\}|")",
      replacement = ''
    ) %>%
      gsub(
        pattern = ',',
        replacement = ' '
      )
  }
  
  
  itoken_data = itoken(
    text_col,
    progressbar = F,
    tokenizer = \(v) {
      v %>%
        tolower %>% # to lower
        removeNumbers %>% #remove all numbers
        replace_punctuations %>% #remove all punctuation
        removePunctuation %>%
        removeWords(tm::stopwords(kind="en")) %>% #remove stopwords
        stemDocument %>% # stemming 
        word_tokenizer 
    }
  )
  
  vocab_data = itoken_data %>%
    create_vocabulary(
      ngram = ngram
    ) %>%
    prune_vocabulary(
      doc_proportion_min = 0.05,
      doc_proportion_max = 0.8
    )
  
  obj_vectorizer <- vocab_vectorizer(vocab_data)
  dtm_data = create_dtm(itoken_data, obj_vectorizer)
  
  if (tf_idf) {
    tfidf = TfIdf$new()
    dtm_data = mlapi::fit_transform(x = dtm_data, model = tfidf)
  }
  
  return(dtm_data)
}


train_length = nrow(y_train)
dtm <- get_dtm(x$amenities, ngram = c(1L,2L))
dtm_train = dtm[1:train_length, ]
dtm_te = dtm[(train_length + 1): nrow(dtm), ]


ind_sample = sample(1:train_length, size = 0.75 * train_length)
dtm_tr = dtm_train[ind_sample, ]
dtm_va = dtm_train[-ind_sample, ]
y_tr = y_train$high_booking_rate[ind_sample]
y_va = y_train$high_booking_rate[-ind_sample]


dtm_access <- get_dtm(x$access)
dtm_access_train = dtm_access[1:train_length, ]
dtm_access_te = dtm_access[(train_length + 1): nrow(dtm), ]


ind_sample = sample(1:train_length, size = 0.75 * train_length)
dtm_access_tr = dtm_access_train[ind_sample, ]
dtm_access_va = dtm_access_train[-ind_sample, ]
y_tr = y_train$high_booking_rate[ind_sample]
y_va = y_train$high_booking_rate[-ind_sample]



# logistic amenities ---------------------
md <- glmnet(
  x = dtm_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_va, type = 'response')
get_auc(pred, y_va)
vip(md, 35)


# logistic access ----------------------
lambda_search <- vec_search(
  vec_param = seq(10^-20, 10&-10, length.out = 100),
  x = dtm_access_train, y = y_train$high_booking_rate,
  trainer = \(x, y, param) {
    md <- glmnet(
      x = x, y = y,
      family = 'binomial',
      alpha = 1,
      lambda = param
    )
    return(md)
  },
  predictor = \(md, x) {
    pred <- predict(md, newx = x, type = 'response')
    return(pred)
  },
  measurer = \(y1, y2){
    return(
      get_auc(y1, y2)
    )
  } 
)
lambda_search[order(lambda_search$measurement, decreasing = T), ]


md <- glmnet(
  x = dtm_access_tr, y = y_tr,
  family = 'binomial',
  alpha = 1,
  lambda = 10^-7
)

pred <- predict(md, newx = dtm_access_va, type = 'response')
get_auc(pred, y_va)
vip(md, 35)



vec_auc <- cross_val(
  x = dtm_train,
  y = y_train$high_booking_rate,
  trainer = \(x_set, y_set) {
    model <- glmnet(
      x = x_set, y = y_set,
      family = 'binomial',
      alpha = 1,
      lambda = 10^-7
    )
    return(model)
  },
  predictor = \(model, x) {
    pred <- predict(model, newx = x, type = 'response')
    return(pred)
  },
  measurer = \(y1, y2) {
    return(
      get_auc(y1, y2)
    )
  }
)
vec_auc


vec_doc_min_prop
for (ind in 1:length(vec_doc_min_prop)) {
  
}
