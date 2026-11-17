suppressPackageStartupMessages(library(data.table))

set.seed(20261116)
train <- fread("data/train.csv")
test <- fread("data/test.csv")
setorder(train, breath_id, time_step, id)
setorder(test, breath_id, time_step, id)

# Hold out complete breaths so adjacent rows cannot leak across the split.
breaths <- unique(train$breath_id)
valid_breaths <- sample(breaths, ceiling(length(breaths) * 0.10))
train[, is_valid := breath_id %in% valid_breaths]

make_features <- function(d) {
  d[, step := seq_len(.N) - 1L, by = breath_id]
  d[, dt := time_step - shift(time_step, fill = first(time_step)), by = breath_id]
  d[, cum_u := cumsum(u_in), by = breath_id]
  d[, cum_vol := cumsum(u_in * dt), by = breath_id]
  for (k in 1:3) d[, paste0("lag", k) := shift(u_in, k, fill = 0), by = breath_id]
  # Polynomial terms and lung-condition indicators let the linear model capture
  # the main nonlinear time, valve, and cumulative-volume effects.
  d[, `:=`(t2 = time_step^2, t3 = time_step^3,
           u2 = u_in^2, log_cum = log1p(cum_vol), cum2 = cum_vol^2,
           ut = u_in * time_step, cut = cum_vol * time_step,
           uout_t = u_out * time_step)]
  d[, `:=`(r20 = as.numeric(R == 20), r50 = as.numeric(R == 50),
           c20 = as.numeric(C == 20), c50 = as.numeric(C == 50))]
  d[, `:=`(rt = R * time_step, ct = C * time_step,
           rcum = R * cum_vol, ccum = C * cum_vol)]
  d
}

train <- make_features(train)
test <- make_features(test)
features <- c("R", "C", "time_step", "t2", "t3", "u_in", "u2", "u_out",
              "step", "lag1", "lag2", "lag3", "cum_u", "cum_vol", "log_cum",
              "cum2", "ut", "cut", "uout_t", "r20", "r50", "c20", "c50",
              "rt", "ct", "rcum", "ccum")

# Accumulate normal equations in chunks to avoid allocating a many-million-row
# dense feature matrix. A small ridge term stabilizes correlated polynomial terms.
p <- length(features) + 1L
gram <- matrix(0, p, p)
rhs <- numeric(p)
chunk <- 200000L
fit_rows <- which(!train$is_valid)
for (start in seq.int(1L, length(fit_rows), by = chunk)) {
  ii <- fit_rows[start:min(start + chunk - 1L, length(fit_rows))]
  x <- cbind(1, as.matrix(train[ii, ..features]))
  y <- train$pressure[ii]
  gram <- gram + crossprod(x)
  rhs <- rhs + crossprod(x, y)
}
diag(gram) <- diag(gram) + pmax(diag(gram), 1) * 1e-8
beta <- solve(gram, rhs)

predict_chunks <- function(d) {
  out <- numeric(nrow(d))
  for (start in seq.int(1L, nrow(d), by = chunk)) {
    ii <- start:min(start + chunk - 1L, nrow(d))
    x <- cbind(1, as.matrix(d[ii, ..features]))
    out[ii] <- drop(x %*% beta)
  }
  out
}
valid_rows <- which(train$is_valid & train$u_out == 0)
valid_pred <- predict_chunks(train[valid_rows])
mae <- mean(abs(valid_pred - train$pressure[valid_rows]))
cat(sprintf("Validation breaths: %d; scored rows (u_out == 0): %d; MAE: %.6f\n",
            length(valid_breaths), length(valid_rows), mae))

# Refit the same baseline on every labeled breath for test predictions.
gram[,] <- 0
rhs[] <- 0
fit_rows <- seq_len(nrow(train))
for (start in seq.int(1L, length(fit_rows), by = chunk)) {
  ii <- fit_rows[start:min(start + chunk - 1L, length(fit_rows))]
  x <- cbind(1, as.matrix(train[ii, ..features]))
  y <- train$pressure[ii]
  gram <- gram + crossprod(x)
  rhs <- rhs + crossprod(x, y)
}
diag(gram) <- diag(gram) + pmax(diag(gram), 1) * 1e-8
beta <- solve(gram, rhs)
test_pred <- predict_chunks(test)
submission <- data.table(id = test$id, pressure = test_pred)
setorder(submission, id)
fwrite(submission, "submission.csv")
sample <- fread("data/sample_submission.csv")
stopifnot(identical(names(submission), names(sample)), nrow(submission) == nrow(sample),
          identical(submission$id, sample$id), all(is.finite(submission$pressure)))
cat(sprintf("Submission format OK: %d rows, columns %s\n",
            nrow(submission), paste(names(submission), collapse = ",")))
