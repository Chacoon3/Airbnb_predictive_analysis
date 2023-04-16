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
