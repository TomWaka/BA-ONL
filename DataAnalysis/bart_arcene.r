library(data.table)
library(dbarts)

load_data <- function(data_path, labels_path, sep=' ', header=FALSE) {
  data <- fread(data_path, sep=sep, header=header)
  labels <- fread(labels_path, sep=sep, header=header)[, lapply(.SD, function(x) ifelse(x == -1, 0, 1)), .SDcols=1]
  return(list(data, labels))
}

train_data <- load_data(train_data_path, train_labels_path)
valid_data <- load_data(valid_data_path, valid_labels_path)

# データの結合
X <- rbindlist(list(train_data[[1]], valid_data[[1]]))
y <- rbindlist(list(train_data[[2]], valid_data[[2]]))

# クロスバリデーションでの評価
accuracies <- c()
aucs <- c()
ras <- c()
rbs <- c()

for(i in 1:5) {
  test_index <- ((i-1)*40+1):(i*40)
  train_index <- setdiff(1:nrow(X), test_index)
  
  X_train <- X[train_index, ]
  X_test <- X[test_index, ]
  y_train <- y[train_index, , drop=FALSE]
  y_test <- y[test_index, , drop=FALSE]
  
  # bart
  bart_model <- bart(X_train, as.numeric(y_train$V1), keeptrees = TRUE)
  samples <- predict(bart_model, newdata = X_test, type = "ppd")

  posterior_lower <- apply(samples, 2, function(x) quantile(x, probs = 0.025))
  posterior_upper <- apply(samples, 2, function(x) quantile(x, probs = 0.975))
  y_prob <- apply(samples, 2, function(x) mean(x))
  y_pred <- ifelse(y_prob > 0.5, 1, 0)
  
  acc <- mean(y_pred == y_test$V1)
  auc_value <- pROC::auc(pROC::roc(y_test$V1, y_prob))

  RA <- sum((abs(y_test - y_pred) > 0) & (posterior_upper > 0.5 & posterior_lower < 0.5)) / sum(abs(y_test - y_pred) > 0)
  RB <- sum((abs(y_test - y_pred) == 0) & !(posterior_upper > 0.5 & posterior_lower < 0.5)) / sum(!(posterior_upper > 0.5 & posterior_lower < 0.5))

  accuracies <- c(accuracies, acc)
  aucs <- c(aucs, auc_value)
  ras <- c(ras, RA)
  rbs <- c(rbs, RB)
}