% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/train.default.R
\name{train}
\alias{train}
\alias{train.default}
\alias{train.formula}
\alias{train.recipe}
\title{Fit Predictive Models over Different Tuning Parameters}
\usage{
train(x, ...)

\method{train}{default}(
  x,
  y,
  method = "rf",
  preProcess = NULL,
  ...,
  weights = NULL,
  metric = ifelse(is.factor(y), "Accuracy", "RMSE"),
  maximize = ifelse(metric \%in\% c("RMSE", "logLoss", "MAE", "logLoss"), FALSE, TRUE),
  trControl = trainControl(),
  tuneGrid = NULL,
  tuneLength = ifelse(trControl$method == "none", 1, 3)
)

\method{train}{formula}(form, data, ..., weights, subset, na.action = na.fail, contrasts = NULL)

\method{train}{recipe}(
  x,
  data,
  method = "rf",
  ...,
  metric = ifelse(is.factor(y_dat), "Accuracy", "RMSE"),
  maximize = ifelse(metric \%in\% c("RMSE", "logLoss", "MAE"), FALSE, TRUE),
  trControl = trainControl(),
  tuneGrid = NULL,
  tuneLength = ifelse(trControl$method == "none", 1, 3)
)
}
\arguments{
\item{x}{For the default method, \code{x} is an object where
samples are in rows and features are in columns. This could be a
simple matrix, data frame or other type (e.g. sparse matrix) but
must have column names (see Details below). Preprocessing using
the \code{preProcess} argument only supports matrices or data
frames. When using the recipe method, \code{x} should be an
unprepared [recipes::recipe()] object that describes the model
terms (i.e. outcome, predictors, etc.) as well as any
pre-processing that should be done to the data. This is an
alternative approach to specifying the model. Note that, when
using the recipe method, any arguments passed to \code{preProcess}
will be ignored. See the links and example below for more details
using recipes.}

\item{\dots}{Arguments passed to the classification or
regression routine (such as
\code{\link[randomForest]{randomForest}}). Errors will occur if
values for tuning parameters are passed here.}

\item{y}{A numeric or factor vector containing the outcome for
each sample.}

\item{method}{A string specifying which classification or
regression model to use. Possible values are found using
\code{names(getModelInfo())}. See
\url{http://topepo.github.io/caret/train-models-by-tag.html}. A
list of functions can also be passed for a custom model
function. See
\url{http://topepo.github.io/caret/using-your-own-model-in-train.html}
for details.}

\item{preProcess}{A string vector that defines a pre-processing
of the predictor data. Current possibilities are "BoxCox",
"YeoJohnson", "expoTrans", "center", "scale", "range",
"knnImpute", "bagImpute", "medianImpute", "pca", "ica" and
"spatialSign". The default is no pre-processing. See
\code{\link{preProcess}} and \code{\link{trainControl}} on the
procedures and how to adjust them. Pre-processing code is only
designed to work when \code{x} is a simple matrix or data frame.}

\item{weights}{A numeric vector of case weights. This argument
will only affect models that allow case weights.}

\item{metric}{A string that specifies what summary metric will
be used to select the optimal model. By default, possible values
are "RMSE" and "Rsquared" for regression and "Accuracy" and
"Kappa" for classification. If custom performance metrics are
used (via the \code{summaryFunction} argument in
\code{\link{trainControl}}, the value of \code{metric} should
match one of the arguments. If it does not, a warning is issued
and the first metric given by the \code{summaryFunction} is
used. (NOTE: If given, this argument must be named.)}

\item{maximize}{A logical: should the metric be maximized or
minimized?}

\item{trControl}{A list of values that define how this function
acts. See \code{\link{trainControl}} and
\url{http://topepo.github.io/caret/using-your-own-model-in-train.html}.
(NOTE: If given, this argument must be named.)}

\item{tuneGrid}{A data frame with possible tuning values. The
columns are named the same as the tuning parameters. Use
\code{\link{getModelInfo}} to get a list of tuning parameters
for each model or see
\url{http://topepo.github.io/caret/available-models.html}.
(NOTE: If given, this argument must be named.)}

\item{tuneLength}{An integer denoting the amount of granularity
in the tuning parameter grid. By default, this argument is the
number of levels for each tuning parameters that should be
generated by \code{\link{train}}. If \code{\link{trainControl}}
has the option \code{search = "random"}, this is the maximum
number of tuning parameter combinations that will be generated
by the random search. (NOTE: If given, this argument must be
named.)}

\item{form}{A formula of the form \code{y ~ x1 + x2 + ...}}

\item{data}{Data frame from which variables specified in
\code{formula} or \code{recipe} are preferentially to be taken.}

\item{subset}{An index vector specifying the cases to be used
in the training sample. (NOTE: If given, this argument must be
named.)}

\item{na.action}{A function to specify the action to be taken
if NAs are found. The default action is for the procedure to
fail. An alternative is \code{na.omit}, which leads to rejection
of cases with missing values on any required variable. (NOTE: If
given, this argument must be named.)}

\item{contrasts}{A list of contrasts to be used for some or all
the factors appearing as variables in the model formula.}
}
\value{
A list is returned of class \code{train} containing:
 \item{method }{The chosen model.} \item{modelType }{An
 identifier of the model type.} \item{results }{A data frame the
 training error rate and values of the tuning parameters.}
 \item{bestTune }{A data frame with the final parameters.}
 \item{call}{The (matched) function call with dots expanded}
 \item{dots}{A list containing any ... values passed to the
 original call} \item{metric}{A string that specifies what
 summary metric will be used to select the optimal model.}
 \item{control}{The list of control parameters.} \item{preProcess
 }{Either \code{NULL} or an object of class
 \code{\link{preProcess}}} \item{finalModel}{A fit object using
 the best parameters} \item{trainingData}{A data frame}
 \item{resample}{A data frame with columns for each performance
 metric. Each row corresponds to each resample. If leave-one-out
 cross-validation or out-of-bag estimation methods are requested,
 this will be \code{NULL}. The \code{returnResamp} argument of
 \code{\link{trainControl}} controls how much of the resampled
 results are saved.} \item{perfNames}{A character vector of
 performance metrics that are produced by the summary function}
 \item{maximize}{A logical recycled from the function arguments.}
 \item{yLimits}{The range of the training set outcomes.}
 \item{times}{A list of execution times: \code{everything} is for
 the entire call to \code{train}, \code{final} for the final
 model fit and, optionally, \code{prediction} for the time to
 predict new samples (see \code{\link{trainControl}})}
}
\description{
This function sets up a grid of tuning parameters for a number
 of classification and regression routines, fits each model and
 calculates a resampling based performance measure.
}
\details{
\code{train} can be used to tune models by picking the
 complexity parameters that are associated with the optimal
 resampling statistics. For particular model, a grid of
 parameters (if any) is created and the model is trained on
 slightly different data for each candidate combination of tuning
 parameters. Across each data set, the performance of held-out
 samples is calculated and the mean and standard deviation is
 summarized for each combination. The combination with the
 optimal resampling statistic is chosen as the final model and
 the entire training set is used to fit a final model.

The predictors in \code{x} can be most any object as long as
 the underlying model fit function can deal with the object
 class. The function was designed to work with simple matrices
 and data frame inputs, so some functionality may not work (e.g.
 pre-processing). When using string kernels, the vector of
 character strings should be converted to a matrix with a single
 column.

More details on this function can be found at
 \url{http://topepo.github.io/caret/model-training-and-tuning.html}.

A variety of models are currently available and are enumerated
 by tag (i.e. their model characteristics) at
 \url{http://topepo.github.io/caret/train-models-by-tag.html}.

More details on using recipes can be found at
 \url{http://topepo.github.io/caret/using-recipes-with-train.html}.
 Note that case weights can be passed into \code{train} using a
 role of \code{"case weight"} for a single variable. Also, if
 there are non-predictor columns that should be used when
 determining the model's performance metrics, the role of
 \code{"performance var"} can be used with multiple columns and
 these will be made available during resampling to the
 \code{summaryFunction} function.
}
\examples{

\dontrun{

#######################################
## Classification Example

data(iris)
TrainData <- iris[,1:4]
TrainClasses <- iris[,5]

knnFit1 <- train(TrainData, TrainClasses,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "cv"))

knnFit2 <- train(TrainData, TrainClasses,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "boot"))


library(MASS)
nnetFit <- train(TrainData, TrainClasses,
                 method = "nnet",
                 preProcess = "range",
                 tuneLength = 2,
                 trace = FALSE,
                 maxit = 100)

#######################################
## Regression Example

library(mlbench)
data(BostonHousing)

lmFit <- train(medv ~ . + rm:lstat,
               data = BostonHousing,
               method = "lm")

library(rpart)
rpartFit <- train(medv ~ .,
                  data = BostonHousing,
                  method = "rpart",
                  tuneLength = 9)

#######################################
## Example with a custom metric

madSummary <- function (data,
                        lev = NULL,
                        model = NULL) {
  out <- mad(data$obs - data$pred,
             na.rm = TRUE)
  names(out) <- "MAD"
  out
}

robustControl <- trainControl(summaryFunction = madSummary)
marsGrid <- expand.grid(degree = 1, nprune = (1:10) * 2)

earthFit <- train(medv ~ .,
                  data = BostonHousing,
                  method = "earth",
                  tuneGrid = marsGrid,
                  metric = "MAD",
                  maximize = FALSE,
                  trControl = robustControl)


#######################################
## Example with a recipe

data(cox2)

cox2 <- cox2Descr
cox2$potency <- cox2IC50

library(recipes)

cox2_recipe <- recipe(potency ~ ., data = cox2) \%>\%
  ## Log the outcome
  step_log(potency, base = 10) \%>\%
  ## Remove sparse and unbalanced predictors
  step_nzv(all_predictors()) \%>\%
  ## Surface area predictors are highly correlated so
  ## conduct PCA just on these.
  step_pca(contains("VSA"), prefix = "surf_area_",
           threshold = .95) \%>\%
  ## Remove other highly correlated predictors
  step_corr(all_predictors(), -starts_with("surf_area_"),
            threshold = .90) \%>\%
  ## Center and scale all of the non-PCA predictors
  step_center(all_predictors(), -starts_with("surf_area_")) \%>\%
  step_scale(all_predictors(), -starts_with("surf_area_"))

set.seed(888)
cox2_lm <- train(cox2_recipe,
                 data = cox2,
                 method = "lm",
                 trControl = trainControl(method = "cv"))

#######################################
## Parallel Processing Example via multicore package

## library(doMC)
## registerDoMC(2)

## NOTE: don't run models form RWeka when using
### multicore. The session will crash.

## The code for train() does not change:
set.seed(1)
usingMC <-  train(medv ~ .,
                  data = BostonHousing,
                  method = "glmboost")

## or use:
## library(doMPI) or
## library(doParallel) or
## library(doSMP) and so on

}


}
\references{
\url{http://topepo.github.io/caret/}

Kuhn (2008), ``Building Predictive Models in R Using the caret''
(\doi{10.18637/jss.v028.i05})

\url{https://topepo.github.io/recipes/}
}
\seealso{
\code{\link{models}}, \code{\link{trainControl}},
 \code{\link{update.train}}, \code{\link{modelLookup}},
 \code{\link{createFolds}}, \code{\link[recipes]{recipe}}
}
\author{
Max Kuhn (the guts of \code{train.formula} were based
 on Ripley's \code{nnet.formula})
}
\keyword{models}
