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


compare_feature <- function(feat_mutator, x, y, model) {
  x_new_feats <- x %>% feat_mutator()
  
  
}