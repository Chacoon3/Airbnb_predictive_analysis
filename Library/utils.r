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


get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}


get_cutoff_dataframe <- 
  function(y_pred_prob, y_valid_factor, level, max_fpr = 0.1, step = 0.005) {
    
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
      }
    }
    
    
    df <- data.frame(
      cutoff = c(vec_cutoff, vec_cutoff),
      cutoff_bound = rep(cutoff_bound, length(vec_cutoff) * 2),
      metric = c(vec_tpr, vec_fpr),
      type = c(rep('tpr', length(vec_tpr)), rep('fpr', length(vec_fpr)))
    )
    
    return(df)
  }


plot_cutoff_dataframe <- function(df) {
  return(
    ggplot(data = df, aes(x = cutoff, y = metric, color = type)) +
      ggtitle('cutoff over metrics') +
      geom_line() +
      geom_vline(xintercept = df$cutoff_bound) +
      ylim(0, 1)
  )
}
