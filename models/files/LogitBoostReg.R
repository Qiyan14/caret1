#' LogitBoostReg Model Definition
#'
#' Boosted Logistic Regularization Regression using a modified LogitBoost algorithm.
#'
#' @export
LogitBoostReg <- list(
  label = "Boosted Logistic Regularization Regression (LogitBoostReg)",
  library = "TestingTools",
  loop = function(grid) {            
    ## Get the largest value of nIter to fit the "full" model
    loop <- grid[which.max(grid$nIter), , drop = FALSE]
    
    submodels <- grid[-which.max(grid$nIter), , drop = FALSE]
    
    ## Encapsulate submodels in a list
    submodels <- list(submodels)
    list(loop = loop, submodels = submodels)
  },
  type = "Classification",
  parameters = data.frame(
    parameter = 'nIter',
    class = 'numeric',
    label = '# Boosting Iterations'
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out <- data.frame(nIter = 1 + ((1:len) * 10))
    } else {
      out <- data.frame(nIter = unique(sample(1:100, size = len, replace = TRUE)))
    }
    out
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    ## Use your new LogitBoostReg function from TestingTools
    TestingTools::LogitBoostReg(as.matrix(x), y, nIter = param$nIter, ...)
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    ## Predict class labels using the new LogitBoostReg predictor
    out <- TestingTools::predict.LogitBoostReg(modelFit, newdata, type = "class")
    if (!is.null(submodels)) {
      tmp <- out
      out <- vector(mode = "list", length = nrow(submodels) + 1)
      out[[1]] <- tmp
      
      for (j in seq(along = submodels$nIter)) {
        out[[j + 1]] <- TestingTools::predict.LogitBoostReg(modelFit,
                                newdata,
                                nIter = submodels$nIter[j])
      }
    }
    out
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    out <- TestingTools::predict.LogitBoostReg(modelFit, newdata, type = "raw")
    ## Normalize probabilities
    out <- t(apply(out, 1, function(x) x / sum(x)))
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = nrow(submodels) + 1)
      tmp[[1]] <- out
      
      for (j in seq(along = submodels$nIter)) {
        tmpProb <- TestingTools::predict.LogitBoostReg(modelFit,
                          newdata,
                          type = "raw",
                          nIter = submodels$nIter[j])
        tmpProb <- t(apply(tmpProb, 1, function(x) x / sum(x)))
        tmp[[j + 1]] <- as.data.frame(tmpProb[, modelFit$obsLevels, drop = FALSE],
                                      stringsAsFactors = TRUE)
      }
      out <- tmp
    }
    out
  },
  predictors = function(x, ...) {                    
    if (!is.null(x$xNames)) {
      out <- unique(x$xNames[x$Stump[, "feature"]])
    } else {
      out <- NA
    }
    out
  },
  levels = function(x) x$obsLevels,
  tags = c("Ensemble Model", "Boosting", "Implicit Feature Selection",
           "Tree-Based Model", "Logistic Regression"),
  sort = function(x) x[order(x[, 1]), ]
)
