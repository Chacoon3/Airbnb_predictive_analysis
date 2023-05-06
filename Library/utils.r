library(ROCR)


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


plot_roc_ggplot <- function(pred_y_prob, valid_y_factor, step_length = 0.01, p = 1, n = 0) {
  
  total_p = sum(valid_y_factor == p)
  total_n = sum(valid_y_factor == n)
  vec_tpr = c()
  vec_fpr = c()
  cutoff = 0 + step_length
  while (cutoff <= 1) {
      pred = ifelse(pred_y_prob >= cutoff, p, n)
      
      pred_tp = sum(pred == valid_y_factor & pred == p)
      pred_fp = sum(pred != valid_y_factor & pred == p)
      
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


get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}


# returns a data frame that provides insights regarding cutoff selection
get_cutoff_dataframe <- 
  function(y_pred_prob, y_valid_factor, level, max_fpr = 0.1, step = 0.001) {
    
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
    
    
    tpr = 0
    fpr = 0
    cutoff = 1
    cutoff_bound <- 2
    vec_tpr = c(tpr)
    vec_fpr = c(fpr)
    vec_cutoff = c(cutoff)
    best_valid_tpr = 0
    while (cutoff >= 0 ) {
      cutoff = cutoff - step
      # print(cutoff)
      y_pred = ifelse(y_pred_prob >= cutoff, level[2], level[1])
      tpr = get_tpr(y_pred, y_valid_factor)
      fpr = get_fpr(y_pred, y_valid_factor)
      
      
      vec_cutoff= c(vec_cutoff, cutoff)
      vec_tpr = c(vec_tpr, tpr)
      vec_fpr = c(vec_fpr, fpr)
      
      if (fpr > max_fpr && cutoff_bound > 1) {
        cutoff_bound = cutoff
        best_valid_tpr = tpr
      }
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


# plota a cutoff data frame, i.e. cutoff over trp and fpr
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


cross_val <- function(trainer, predictor, measurer, x, y, fold_count = 5) {

  size = nrow(x)
  folds <- cut(
    1:size %>% sample(size = size), 
    breaks = fold_count, 
    labels = FALSE
  )
  
  vec_measure = rep(0, fold_count)
  for (ind in 1:fold_count) {
    indice_tr = which(folds != ind, arr.ind = TRUE)
    x_tr <- x[indice_tr, ]
    y_tr <- y[indice_tr]
    x_va <- x[-indice_tr, ]
    y_va <- y[-indice_tr]
    
    
    model <- trainer(x_tr, y_tr)
    y_pred <- predictor(model, x_va)
    m <- measurer(y_pred, y_va)
    vec_measure[ind] = m
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


grid_search_xgb <- function(
    x_tr, y_tr, x_va, y_va, 
    vec_tree_depth, 
    vec_nround,
    vec_eta_set,
    report_progress = T
){
  
  #an empty dataframe to store auc
  auc_df = data.frame(depth = c(0),
                      nround = c(0),
                      eta_set=c(0),
                      auc = c(0))
  
  counter = 0
  #nested loops to tune these three parameters
  for(i in c(1:length(vec_tree_depth))){
    for(j in c(1:length(vec_nround))){
      for(k in c(1:length(vec_eta_set))){
        thisdepth <- vec_tree_depth[i]
        thisnrounds <- vec_nround[j]
        thiseta <- vec_eta_set[k]
        
        inner_bst <- xgboost(
          data = x_tr,
          label = y_tr,
          max.depth = thisdepth,
          eta = thiseta, 
          nrounds = thisnrounds,
          objective = "binary:logistic",
          eval_metric = "auc", 
          verbose = F
        )
        
        inner_bst_pred <- predict(inner_bst, x_va)
        auc_valid <- get_auc(inner_bst_pred, y_va)
        auc_df[nrow(auc_df)+1,] <- c(thisdepth,thisnrounds,thiseta,auc_valid)
        
      }
      counter = counter + 1
      if (report_progress) {
        print(counter)
      }
    }
  }
  
  return(auc_df)
}


# hyperparameter tuning with a vector of hyperpamameter
vec_search <- function(
    vec_param, x, y,
    trainer, predictor, measurer,
    verbose = T,
    train_ratio = 0.75,
    cv_folds = 0) {
  
  
  x_row = nrow(x)
  vec_measure = c()
  for (ind in 1:length(vec_param)) {
    
    if (cv_folds <= 0) {
      indice_tr = sample(1:x_row, train_ratio * x_row)
      x_tr = x[indice_tr, ]
      x_va = x[-indice_tr, ]
      y_tr = y[indice_tr]
      y_va = y[-indice_tr]
      
      param = vec_param[ind]
      model = trainer(x_tr, y_tr, param)
      pred = predictor(model, x_va)
      meas = measurer(pred, y_va)
      
      vec_measure = c(vec_measure, meas)
    }
    else {
      vec_m = cross_val(
        trainer, predictor, measurer,
        x, y, cv_folds
      )
      
      m = mean(vec_m)
      vec_measure = c(vec_measure, m)
    }
    
    if (verbose) {
      print(
        paste('round ', ind, ' completed')
      )
    }
  }
  data = data.frame(
    param = vec_param,
    measurement = vec_measure
  )
  return(data)
}


# hyperparameter tuning with two vectors of hyperpamameter
mat_search <- function(
      vec_param1, vec_param2, x, y,
      trainer, predictor, measurer,
      verbose = T,
      train_ratio = 0.75
) {
  
  x_row = nrow(x)
  vec_measure = c()
  res = data.frame(param1 = c(), param2 = c(), measurement = c())
  
  for (ind in 1:length(vec_param1)) {
    param1 = vec_param1[ind]
    
    for (ind2 in 1:length(vec_param2)) {
      param2 = vec_param2[ind]
      
      indice_tr = sample(1:x_row, train_ratio * x_row)
      x_tr = x[indice_tr, ]
      x_va = x[-indice_tr, ]
      y_tr = y[indice_tr]
      y_va = y[-indice_tr]
      
      model = trainer(x_tr, y_tr, param1, param2)
      pred = predictor(model, x_va)
      meas = measurer(pred, y_va)
      
      
      res[nrow(res) + 1, ] = c(param1, param2, meas)
      
      if (verbose) {
        print(
          paste('round ', ind, ' completed')
        )
      }
    }
  }
  
  return(res)
}

