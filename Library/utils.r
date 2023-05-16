library(ROCR)


get_baseline_accuracy <- function(y, n = 0, p = 1) {
  sum_n = sum(y == n)
  sum_p = sum(y == p)
  total = length(y)
  if (sum_n > sum_p) {
    return(sum_n / total)
  }
  else{
    return(sum_p / total)
  }
}


# plot the roc graph
plot_roc <- function(pred_y_prob, valid_y_factor)  {
  pred_obj = prediction(pred_y_prob, valid_y_factor)
  roc_full <- performance(pred_obj, "tpr", "fpr")
  auc <- performance(pred_obj, measure = "auc")@y.values[[1]]
  
  plot(roc_full, col = "red", lwd = 1, main = 'ROC', 
       sub = paste('AUC:', round(auc, 4)))
  abline(v = 0.1)
}


# returns the auc
get_auc <- function(pred_y_prob, valid_y_factor) {
  pred_obj = prediction(pred_y_prob, valid_y_factor)
  auc <- performance(pred_obj, measure = 'auc')@y.values[[1]]
  return(auc)
}


get_tpr <- function(pred_y_prob, valid_y_factor, cutoff = 0.5, p = 1, n = 0) {
  total_p = sum(valid_y_factor == p)
  pred = ifelse(pred_y_prob >= cutoff, p, n)
  pred_tp = sum(pred == valid_y_factor & pred == p)
  return(pred_tp / total_p)
}


get_fpr <- function(pred_y_prob, valid_y_factor, cutoff = 0.5, p = 1, n = 0) {
  total_n = sum(valid_y_factor == n)
  pred = ifelse(pred_y_prob >= cutoff, p, n)
  pred_fp = sum(pred != valid_y_factor & pred == p)
  return(pred_fp / total_n)
}


plot_roc_ggplot <- function(pred_y_prob, valid_y, step_length = 0.01, p = 1, n = 0) {
  
  total_p = sum(valid_y == p)
  total_n = sum(valid_y == n)
  vec_tpr = c()
  vec_fpr = c()
  cutoff = 0 + step_length
  while (cutoff <= 1) {
      pred = ifelse(pred_y_prob >= cutoff, p, n)
      
      pred_tp = sum(pred == valid_y & pred == p)
      pred_fp = sum(pred != valid_y & pred == p)
      
      tpr = pred_tp / total_p
      fpr = pred_fp / total_n
      
      vec_tpr = c(vec_tpr, tpr)
      vec_fpr = c(vec_fpr, fpr)
      
      cutoff = cutoff + step_length
  }
  
  data <- data.frame(
    tpr = vec_tpr,
    fpr = vec_fpr
  )
  
  return(
    ggplot(data = data, aes(x = fpr, y = tpr)) +geom_line()
  )
}


get_learning_curve_data <- function(
      x, y, 
      trainer, predictor, measurer, 
      vec_train_ratio = 1:18 /20, heldout_ratio = 0.2
    ) {
  
  x_row = nrow(x)
  indice_use = sample(1:x_row, (1 - heldout_ratio) * x_row)
  x_use = x[indice_use, ]
  x_ho = x[-indice_use, ]
  y_use = y[indice_use]
  y_ho = y[-indice_use]
  
  x_use_row = nrow(x_use)
  vec_sample_size = c()
  vec_meas = c()
  for (ind in 1:length(vec_train_ratio)) {
    # sampling
    ratio = vec_train_ratio[ind]
    sample_size = ratio * x_tr_row
    indice_tr = sample(1:x_tr_row, sample_size)
    x_tr = x_use[indice_tr, ]
    y_tr = y_use[indice_tr]
    vec_sample_size = c(vec_sample_size, sample_size)
    
    # train -> predict -> measure
    model = trainer(x_tr, y_tr)
    pred = predictor(model, x_ho)
    score = measurer(pred, y_ho)
    
    # store measurement
    vec_meas = c(vec_meas, score)
  }
  
  data_for_plot = data.frame(
    sample_size = vec_sample_size,
    performance = vec_meas
  )
  
  return(data_for_plot)
}


plot_learning_curve <- function(
      x, y, 
      trainer, predictor, measurer, 
      vec_train_ratio = 1:18 /20, heldout_ratio = 0.2
) {
  data = get_learning_curve_data(
      x,y, trainer, predictor, measurer,
      vec_train_ratio, heldout_ratio
    )
  
  return(
    ggplot2::ggplot(
        data = data,
        aes(x = data[1, ], y = data[2, ])
      ) + geom_line()
  )
}


get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}


# returns a data frame that provides insights regarding cutoff selection
get_cutoff_dataframe <- 
  function(y_pred_prob, y_valid_factor, 
           level = c(0, 1), max_fpr = 0.1, step = 0.001) {
    
    if (length(y_pred_prob) != length(y_valid_factor)) {
      stop('prediction and validation have different lengths.')
    }
    
    p = level[2]
    n = level[1]
    
    get_fpr <- function(y_pred, y_valid) {
      count_fp = sum((y_valid == n) & (y_pred == p))
      count_tn = sum(y_valid == n)
      return(count_fp / count_tn)
    }
    
    
    get_tpr <- function(y_pred, y_valid) {
      count_tp = sum((y_valid == p) & (y_valid == y_pred))
      count_p = sum(y_valid == p)
      return(count_tp / count_p)
    }
    
    

    cutoff = 1
    cutoff_bound <- NULL
    vec_tpr = c()
    vec_fpr = c()
    vec_cutoff = c()
    best_valid_tpr = 0
    while (cutoff >= 0 ) {
      # print(cutoff)
      y_pred = ifelse(y_pred_prob >= cutoff, level[2], level[1])
      tpr = get_tpr(y_pred, y_valid_factor)
      fpr = get_fpr(y_pred, y_valid_factor)
      
      
      vec_cutoff= c(vec_cutoff, cutoff)
      vec_tpr = c(vec_tpr, tpr)
      vec_fpr = c(vec_fpr, fpr)
      
      
      if (fpr >= max_fpr && is.null(cutoff_bound)) {
        cutoff_bound = cutoff
        best_valid_tpr = max(vec_tpr)
      }
      
      cutoff = cutoff - step
    }
    
    
    df <- data.frame(
      cutoff = c(vec_cutoff, vec_cutoff),
      cutoff_bound = rep(cutoff_bound, length(vec_cutoff) * 2),
      fpr_cons = rep(max_fpr, length(vec_cutoff) * 2),
      tpr_best = rep(best_valid_tpr, length(vec_cutoff) * 2),
      metric = c(vec_tpr, vec_fpr),
      type = c(rep('tpr', length(vec_tpr)), rep('fpr', length(vec_fpr)))
    )
    
    return(df)
  }


# plota a cutoff data frame, i.e. trp and fpr over cutoff
plot_cutoff_dataframe <- function(df) {
  return(
    ggplot(data = df, aes(x = cutoff, y = metric, color = type)) +
      ggtitle( 
        paste(
          'cutoff over metrics -- fpr max = ',
          df$fpr_cons[1]
        )
      ) +
      geom_line() +
      geom_vline(xintercept = df$cutoff_bound) +
      labs(x = paste(
        'cutoff boundary: ', 
        base::round(df$cutoff_bound[1], 4),
        '   best valid tpr: ',
        base::round(df$tpr_best, 4)
        ),
      ) +
      ylim(0, 1)
  )
}


iterate_on <- function(on, action, verbose = TRUE) {
  vec_res <- c()
  if (verbose) {
    for (element in on) {
      res <- action(element)
      vec_res <- c(vec_res, res)
      paste(
        element, ' completed'
      ) %>%
        print()
    }
  }
  else{
    for (element in on) {
      res <- action(element)
      vec_res <- c(vec_res, res)
    }
  }
  
  return(vec_res)
}


# find columns that are monotonously valued
find_monotonous <- function(df) {
  col_count = ncol(df)
  vec_mono_col = c()
  for (ind in 1:col_count) {
    col <- df[, ind]
    
    if (length(unique(col)) == 1) {
      vec_mono_col = c(vec_mono_col, ind)
    }
  }
  
  return(
    names(df)[vec_mono_col]
  )
}


# cross validation 
cross_val <- function(
      trainer, predictor, measurer, 
      x, y, fold_count = 5,
      verbose = T
    ) {
  if (nrow(x) != length(y)) {
    stop(
      "input x and y should be of the same number of rows!"
    )
  }

  size = nrow(x)
  folds <- cut(
    1:size %>% sample(size = size), 
    breaks = fold_count, 
    labels = F
  )
  
  vec_measure = rep(0, fold_count)
  for (ind in 1:fold_count) {
    indice_tr = which(folds != ind, arr.ind = T)
    x_tr <- x[indice_tr, ]
    y_tr <- y[indice_tr]
    
    x_va <- x[-indice_tr, ]
    y_va <- y[-indice_tr]
    
    
    model <- trainer(x_tr, y_tr)
    y_pred <- predictor(model, x_va)
    m <- measurer(y_pred, y_va)
    vec_measure[ind] = m
    
    if (verbose) {
      print(
        paste('fold ', ind, ' completed', sep = '')
      )
    }
  }
  
  return(vec_measure)
}


compare_feature <- function(
    x_def, x_feat, y, trainer,
    predictor, measurer, n = 3, train_ratio = 0.7,
    verbose = T) {

  vec_measure_def = rep(0, n)
  vec_measure_feat = rep(0, n)
  x_row = nrow(x_def)
  counter = 0
  for (ind in 1:n) {
    indice = sample(1:x_row, size = train_ratio * x_row)
    x_tr_def = x_def[indice, ]
    x_va_def = x_def[-indice, ]
    y_tr = y[indice]
    y_va = y[-indice]
    x_tr_feat = x_feat[indice, ]
    x_va_feat = x_feat[-indice, ]
    
    perf_def = trainer(x_tr_def, y_tr) %>%
      predictor(x_va_def) %>%
      measurer(y_va)
    perf_feat = trainer(x_tr_feat, y_tr) %>%
      predictor(x_va_feat) %>%
      measurer(y_va)
    
    vec_measure_def[ind] = perf_def
    vec_measure_feat[ind] = perf_feat
    
    counter = counter + 1
    if (verbose) {
      print(counter)
    }
  }
  
  res = data.frame(
    n = 1:n,
    default_performance = vec_measure_def,
    new_feature_performance = vec_measure_feat
  )
  res[nrow(res) + 1, ] = c('average', mean(vec_measure_def), mean(vec_measure_feat))
  res[nrow(res) + 1, ] = c('best', max(vec_measure_def), max(vec_measure_feat))
  res[nrow(res) + 1, ] = c('min', min(vec_measure_def), min(vec_measure_feat))
  res[nrow(res) + 1, ] = c('variance', var(vec_measure_def), var(vec_measure_feat))
  
  return(res)
}


compare_feature_2 <- function(
    x_matrix, feat_vector, y, trainer,
    predictor, measurer, n = 3, train_ratio = 0.7,
    verbose = T) {
  
  vec_measure_def = rep(0, n)
  vec_measure_feat = rep(0, n)
  x_feat = cbind(x_matrix, feat_vector)
  x_row = nrow(x_matrix)
  counter = 0
  for (ind in 1:n) {
    indice = sample(1:x_row, size = train_ratio * x_row)
    x_tr_def = x_matrix[indice, ]
    x_va_def = x_matrix[-indice, ]
    y_tr = y[indice]
    y_va = y[-indice]
    x_tr_feat = x_feat[indice, ]
    x_va_feat = x_feat[-indice, ]
    
    perf_def = trainer(x_tr_def, y_tr) %>%
      predictor(x_va_def) %>%
      measurer(y_va)
    perf_feat = trainer(x_tr_feat, y_tr) %>%
      predictor(x_va_feat) %>%
      measurer(y_va)
    
    vec_measure_def[ind] = perf_def
    vec_measure_feat[ind] = perf_feat
    
    counter = counter + 1
    if (verbose) {
      print(counter)
    }
  }
  
  res = data.frame(
    n = 1:n,
    default_performance = vec_measure_def,
    new_feature_performance = vec_measure_feat
  )
  res[nrow(res) + 1, ] = c('average', mean(vec_measure_def), mean(vec_measure_feat))
  res[nrow(res) + 1, ] = c('best', max(vec_measure_def), max(vec_measure_feat))
  res[nrow(res) + 1, ] = c('min', min(vec_measure_def), min(vec_measure_feat))
  res[nrow(res) + 1, ] = c('variance', var(vec_measure_def), var(vec_measure_feat))
  
  return(res)
}


check_npr <- function(n_per_round) {
  if (n_per_round > 5) {
    n_per_round = 5
    warning('n_per_round maximum is 5')
  }
  
  if (n_per_round < 1) {
    n_per_round = 1
    warning('n_per_round minimum is 1')
  }
  
  return(n_per_round)
}


# hyperparameter tuning on a vector
vector_search <- function(
    vec_param1, x, y,
    trainer, predictor, measurer,
    verbose = T,
    n_per_round = 2
    ) {
  
  
    n_per_round = check_npr(n_per_round)
    x_row = nrow(x)
    res = data.frame(param1 = 0, measurement = 0)
    index_folds <- cut(
      1:x_row %>% sample(size = x_row), 
      breaks=5, 
      labels=FALSE
    )
    
  for (ind in 1:length(vec_param1)) {
    param1 = vec_param1[ind]
    
    vec_measure = rep(0, n_per_round)
    for (n in 1:n_per_round) {
      indice_tr <- which(index_folds != n,arr.ind=TRUE)
      x_tr = x[indice_tr, ]
      x_va = x[-indice_tr, ]
      y_tr = y[indice_tr]
      y_va = y[-indice_tr]
      
      model = trainer(x_tr, y_tr, param1)
      pred = predictor(model, x_va)
      meas = measurer(pred, y_va)   
      
      vec_measure[n] = meas
    }

    res[(nrow(res) + 1), ] = c(param1, mean(vec_measure))

    if (verbose) {
      print(
        paste('round ', ind, ' completed')
      )
    }
  }
    
    return(res[2:nrow(res), ] %>% arrange(desc(measurement)))
}


# hyperparameter tuning on a matrix by the Discartes products of two vectors
matrix_search <- function(
      vec_param1, vec_param2, x, y,
      trainer, predictor, measurer,
      verbose = T,
      n_per_round = 2
) {
  

  n_per_round = check_npr(n_per_round)
  counter = 0
  x_row = nrow(x)
  res = data.frame(param1 = 0, param2 = 0, measurement = 0)
  
  index_folds <- cut(
      1:x_row %>% sample(size = x_row), 
      breaks=5, 
      labels=FALSE
    )

  for (ind1 in 1:length(vec_param1)) {
    for (ind2 in 1:length(vec_param2)) {
      param1 = vec_param1[ind1]
      param2 = vec_param2[ind2]
      
      vec_measure = rep(0, n_per_round)
      for (n in 1:n_per_round) {
        indice_tr <- which(index_folds != n,arr.ind=TRUE)
        x_tr = x[indice_tr, ]
        x_va = x[-indice_tr, ]
        y_tr = y[indice_tr]
        y_va = y[-indice_tr]
        
        model = trainer(x_tr, y_tr, param1, param2)
        pred = predictor(model, x_va)
        meas = measurer(pred, y_va)
        
        vec_measure[n] = meas
      }
      
      res[(nrow(res) + 1), ] = c(param1, param2, mean(vec_measure))
      counter = counter + 1
      if (verbose) {
        print(
          paste('round ', counter, ' completed')
        )
      }
    }
  }
  
  return(res[2:nrow(res), ] %>% arrange(desc(measurement)))
}


# hyperparameter tuning on a cube by the Discartes products of three vectors
cube_search <- function(
      x, y, 
      vec_param1, vec_param2, vec_param3,
      trainer, predictor, measurer,
      verbose = T,
      n_per_round = 2
) {
  
  n_per_round = check_npr(n_per_round)
  counter = 0
  x_row = nrow(x)
  res = data.frame(param1 = 0, param2 = 0, param3 = 0, measurement = 0)

  index_folds <- cut(
    1:x_row %>% sample(size = x_row), 
    breaks=5, 
    labels=FALSE
  )
  
  for (i1 in 1:length(vec_param1)) {
    for (i2 in 1:length(vec_param2)) {
      for (i3 in 1:length(vec_param3)) {
        # parameter setting
        p1 = vec_param1[i1]
        p2 = vec_param2[i2]
        p3 = vec_param3[i3]
        
        
        vec_measure = rep(0, n_per_round)
        for (n in 1:n_per_round) {
          # splitting
          indice_tr <- which(index_folds != n,arr.ind=TRUE)
          x_tr = x[indice_tr, ]
          x_va = x[-indice_tr, ]
          y_tr = y[indice_tr]
          y_va = y[-indice_tr]
          
          # training
          model = trainer(x_tr, y_tr, p1, p2, p3)
          pred = predictor(model, x_va)
          meas = measurer(pred, y_va)
        
          
          vec_measure[n] = meas
        }

        res[nrow(res) + 1, ] = c(p1, p2, p3, mean(vec_measure))
        
        counter = counter + 1
        if (verbose) {
          print(
            paste('round ', counter, ' completed')
          )
        }
      }
    }
  }
  
  return(res[2:nrow(res), ] %>% arrange(desc(measurement)))
}


get_dtm <- function(
    text_col, 
    ngram = c(1L, 2L), 
    doc_prop_min = 0,
    doc_prop_max = 1,
    custom_stop_words = NULL,
    tf_idf = T,
    binary = F
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
  
  
  if (is.null(custom_stop_words)) {
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
  } else{
    itoken_data = itoken(
      text_col,
      progressbar = F,
      tokenizer = \(v) {
        v %>%
          tolower %>% # to lower
          removeNumbers %>% #remove all numbers
          replace_punctuations %>% #remove all punctuation
          removePunctuation %>%
          removeWords(custom_stop_words) %>% # word vector
          removeWords(tm::stopwords(kind="en")) %>% #remove stopwords
          stemDocument %>% # stemming 
          word_tokenizer 
      }
    )
  }

  
  vocab_data = itoken_data %>%
    create_vocabulary(
      ngram = ngram
    ) %>%
    text2vec::prune_vocabulary(
      doc_proportion_min = doc_prop_min,
      doc_proportion_max = doc_prop_max
    )
  
  obj_vectorizer <- vocab_vectorizer(vocab_data)
  dtm_data = create_dtm(itoken_data, obj_vectorizer)
  
  if (tf_idf) {
    tfidf = TfIdf$new()
    dtm_data = mlapi::fit_transform(x = dtm_data, model = tfidf)
  }
  
  if (binary) {
    dtm_data = dtm_data > 0 + 0
  }
  
  return(dtm_data)
}


subset_dtm <- function(dtm, vec_col_name) {
  res <- dtm[, which(colnames(dtm) %in% vec_col_name)] 
  return(res)
}


get_final_dtm <- function(x, binary = T) {
  
  print('generating final dtm')
  
  # amenities pruned dtm 
  dtm_am_pruned <- get_dtm(
    x$amenities, tf_idf = T,
    doc_prop_min = 0.01, doc_prop_max = 0.7, binary = binary
  )

  print('20% completed')
  
  # description dtm
  dtm_desc <- get_dtm(
    x$description, tf_idf = T, 
    doc_prop_min = 0.01, doc_prop_max = 0.7, binary = binary
  )

  print('40% completed')
  
  # house_rules dtm
  # dtm_hr <- get_dtm(x$house_rules, doc_prop_min = 0.01, doc_prop_max = 0.7)

  
  # interaction dtm
  dtm_itrt <- get_dtm(
    x$interaction, doc_prop_min = 0.01, doc_prop_max = 0.7, binary = binary)
  print('60% completed')


  # transit dtm
  dtm_transit <- get_dtm(
    x$transit, doc_prop_min = 0.01, doc_prop_max = 0.7, binary = binary
    )
  print('80% completed')
  
  
  # summary dtm
  dtm_summary <- get_dtm(x$summary, doc_prop_min = 0.01, doc_prop_max = 0.7,
                         binary = binary)
  
  # host verification dtm
  # dtm_hv <- get_dtm(x$host_verifications, doc_prop_min = 0.01, doc_prop_max = 0.7)
  # dtm_hv_train = dtm_hv[1:train_length, ]
  # dtm_hv_te = dtm_hv[(train_length + 1): nrow(dtm_hv), ]
  
  
  dtm_final = cbind(
    dtm_am_pruned, dtm_desc, dtm_itrt, dtm_transit, dtm_summary
  )
  
  print('100% completed')
  
  return(dtm_final)
}


plot_tree <- function(dataframe, formula) {
  md_tree <- rpart(
    formula,
    data = dataframe
  )
  
  plot(md_tree)
  text(md_tree)
}


get_vip_dataframe <- function(md, x) {
  plt_vip = vip::vip(md, ncol(x))
  if (any(colnames(plt_vip$data) == 'Sign')) {
    res <- plt_vip$data %>%
      as.data.frame() %>%
      arrange(
        desc(Sign), desc(Importance)
      )
  } else {
    res <- plt_vip$data %>%
      as.data.frame() %>%
      arrange(
        desc(Importance)
      )
  }

  return(res)
}


# returns a character vector storing the feature names whose importance metric
# is not below the given quantile
get_important_feature <- function(md, x, quantile_threshold = 0.75) {
  
  df_vip <- get_vip_dataframe(md, x)
  
  res <- df_vip %>%
    arrange(desc(Importance)) %>%
    filter(Importance >= quantile(Importance, quantile_threshold))

  return(res$Variable)
}



add_new_feature <- function(
    x_main, new_feat, feat_name  
) {
  x_main = data.frame(
    x_main,
    feat_name = new_feat
  )
  colnames(x_main)[length(colnames(x_main))] = feat_name
  
  return(x_main)
}


# count how many geo coordinates are within the raius of the input coordinate
count_station <- function(
    vec_lat, vec_lng,
    ref_lat, ref_lng,
    radius_km = 2,
    verbose = T
) {
  
  start = 1
  end = 0
  res = c()
  while (end < length(vec_lat)) {
    end = min(end + 10000, length(vec_lat))
    
    sub_res = geodist(
      x = data.frame(
        latitude = vec_lat[start:end],
        longitude = vec_lng[start:end]
      ),
      y = data.frame(
        latitude = ref_lat,
        longitude = ref_lng
      ),
      quiet = T
    ) %>%
      apply(
        MARGIN = 1,
        FUN = \(r) {
          return(
            # distance calculated are in metres by default
            sum(r <= radius_km * 1000)
          )
        }
      )
    
    res = c(res, sub_res)
    
    start = start + 10000
    
    if (verbose) {
      print(
        paste(
          round(start * 100 / length(vec_lat), 2), '% completed', sep = ''
        )
      )
    }
  }
  
  return(res)
}


# train several models using the input x and y and returns them as a list
joint_train <- function(
    x, y,
    list_trainer,
    vec_model_names = NULL # names of each model, if you want to specify
) {
  
  list_model = list()
  if (is.null(vec_model_names)) {
    for (ind in 1:length(list_trainer)) {
      md = list_trainer[[ind]](x,y)
      list_model[[ind]] = md
    }
  }
  else {
    for (ind in 1:length(list_trainer)) {
      md = list_trainer[[ind]](x,y)
      list_model[[vec_model_names[ind]]] = md
    }
  }


  return(list_model)
}
# 
# 


# joint prediction using lists of models and predicting functiosn
joint_predict <- function(
    list_model,
    x,
    list_predictor,
    # can be either a string specifying an function or a function
    pred_aggregator = NULL, 
    cut_off = NULL
) {
  
  
  predictor_len = length(list_predictor)
  matrix_pred = matrix(nrow = nrow(x), ncol = length(list_model))
  for (ind in 1:length(list_model)) {
    pred = list_predictor[[min(ind, predictor_len)]](list_model[[ind]], x)
    # print('pred')
    # print(pred)
    matrix_pred[, ind] = pred
  }
  
  
  if (is.null(pred_aggregator)) {
    pred_aggregator = 'mean'
  }
  
  res = apply(X = matrix_pred, MARGIN = 1, FUN = pred_aggregator)
  
  return(res)
}


get_cluster_label <- function(x, preprocess = NULL, n_center = 2, n_start = 10) {
  
  if (! is.null(preprocess)) {
    x = x %>%
      apply(MARGIN = 2, FUN = preprocess)
  }

  
  md_km = kmeans(x,centers=n_center,nstart=n_start)
  
  return(md_km$cluster)
}




# returns a sample of the given dataframe using over sampling
# over_sample <- function(y, over_p = T, p = 1, n = 0, p_prop = 0.3) {
#   
#   if (p_prop > 1 || p_prop < 0) {
#     stop('y_prop must be within 0 and 1')
#   }
#   
#   if (over_p) {
#     index = which(y == p, arr.ind = T)
#   }
#   else {
#     index = which(y != p, arr.ind = T)
#   }
#   
#   extra_index = sample(index, size = p_prop * length(index))
#   return(extra_index)
# }


# returns the index of the first few wrong rankings
# get_wrong_ranking_index <- function(y_pred_prob, y_va, first_few = 1000, export = F, file = NULL, p = 1, n = 0) {
#   
#   df <- data.frame(
#     index = 1:length(y_pred_prob),
#     score = y_pred_prob,
#     label = y_va
#   ) %>%
#     arrange(
#       desc(score)
#     )
#   
#   last_true_p = max(which(df$label == p, arr.ind = T))
#   
#   current = 1
#   index_wrong_p = c()
#   while (first_few > 0 && current <= last_true_p) {
#     if (df$label[current] == n) {
#       index_wrong_p = c(index_wrong_p, current)
#       first_few = first_few - 1
#     }
#     
#     current = current + 1
#   }
#   
#   return(
#     index_wrong_p
#   )
# }


plot_complexity_curve <- function(
      x,y, vec_param1, 
      trainer, predictor, measurer,
      verbose = T, n_per_round = 3,
      param_name = NULL
    ) {
  
  
  n_per_round = check_npr(n_per_round)
  x_row = nrow(x)
  res = data.frame(param1 = 0, measurement_tr = 0, measurement_va = 0)
  index_folds <- cut(
    1:x_row %>% sample(size = x_row), 
    breaks=5, 
    labels=FALSE
  )
  
  for (ind in 1:length(vec_param1)) {
    param1 = vec_param1[ind]
    
    vec_measure_tr = rep(0, n_per_round)
    vec_measure_va = rep(0, n_per_round)
    for (n in 1:n_per_round) {
      indice_tr <- which(index_folds != n,arr.ind=TRUE)
      x_tr = x[indice_tr, ]
      x_va = x[-indice_tr, ]
      y_tr = y[indice_tr]
      y_va = y[-indice_tr]
      
      
      model = trainer(x_tr, y_tr, param1)
      pred_tr = predictor(model, x_tr)
      meas_tr = measurer(pred_tr, y_tr)   
      pred_va = predictor(model, x_va)
      meas_va = measurer(pred_va, y_va)   
      
      
      vec_measure_tr[n] = meas_tr
      vec_measure_va[n] = meas_va
    }
    
    res[(nrow(res) + 1), ] = c(param1, mean(vec_measure_tr), mean(vec_measure_va))
    
    if (verbose) {
      print(
        paste('round ', ind, ' completed')
      )
    }
  }
  
  res = res[2:nrow(res), ] # remove the first row which is a place holder
  optimal_param = vec_param1[which.max(res$measurement_va)]
  
  if (is.null(param_name)) {
    param_name = 'param'
  }

  
  return(
      ggplot() + 
      labs(
        title = "Complexity Graph", subtitle = param_name, 
        caption = paste('optimal param: ', optimal_param, sep = '')
      ) +
      geom_line(
        data = data.frame(param = res$param1, measure = res$measurement_tr), 
        aes(x = param, y= measure, color = 'train')
      ) +
      geom_line(
        data = data.frame(param = res$param1, measure = res$measurement_va),
        aes(x = param, y =measure, color = 'valid')
      ) + 
      geom_vline(
        xintercept = optimal_param
      )
  )
}


# create boosting by training one model several times on samples
single_ensemble <- function(
    x, y,
    trainer,
    verbose = T,
    ensemble_size  = 10,
    sample_prop = 0.618
) {
  
  list_boost = list()
  x_row = nrow(x)
  for (ind in 1:ensemble_size) {
    ind_sample = sample(1:x_row, size = sample_prop * x_row)
    md = trainer(x[ind_sample, ], y[ind_sample])
    list_boost[[ind]] = md
    
    if (verbose) {
      print(
        paste(
          'weak learner ', ind, ' completed', sep = ''
        )
      )
    }
  }
  
  return(list_boost)
}
