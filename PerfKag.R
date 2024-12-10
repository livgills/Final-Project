library(tidymodels)
library(vroom)
library(themis)
library(keras)
library(stacks)
library(parsnip)
library(bonsai)

# Load the datasets
train <- vroom("./train.csv")
test <- vroom("./test.csv")

train$Cover_Type <- as.factor(train$Cover_Type)


# Define the recipe for preprocessing
train$Cover_Type <- as.factor(train$Cover_Type)


# Define the recipe for preprocessing
recipe <- recipe(Cover_Type ~ ., data = train) %>%
  update_role(Id, new_role = "ID") %>% 
  step_impute_median(contains("Soil_Type")) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())



rfmod <- rand_forest(min_n = 2, mtry = 17, trees = 2000) %>% 
  set_engine('ranger') %>%
  set_mode('classification')

# Build the Workflow
rfwf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rfmod)

# Fit the Workflow
final_wf <- rfwf %>%
  fit(data = train)

# Make Predictions from the Fitted Workflow
rf_preds <- final_wf %>%
  predict(new_data = test, type = 'class')

# Prepare the Predictions for Output
rf_output <- tibble(Id = test$Id, Cover_Type = rf_preds$.pred_class)

# Write the Predictions
vroom_write(rf_output, "RF_preds.csv", delim = ",")

##############################
boostmod <- boost_tree(trees = 1500, learn_rate = 0.1, tree_depth = 15) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

# Build the Workflow
boostwf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(boostmod)

# Fit the Workflow
final_wf <- boostwf %>%
  fit(data = train)

# Make Predictions from the Fitted Workflow
boost_preds <- final_wf %>%
  predict(new_data = test, type = "class")

# Prepare the Predictions for Output
boost_output <- tibble(Id = test$Id, Cover_Type = boost_preds$.pred_class)

# Write the Predictions
vroom_write(boost_output, "Boosted_preds_new.csv", delim = ",")

##################

untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# Define the Folds for Cross-Validation
folds <- vfold_cv(train, v = 5, repeats = 1)

# -------- Build the Base Learners for the Stack --------

# RF Model
rf_mod <- rand_forest(mtry = 17,
                      min_n = 2,
                      trees=2000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# RF Workflow
rf_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_mod)

# Resample for Stack as a 'tunedModel'
rf_results_stack <- fit_resamples(rf_wf,
                                  resamples = folds,
                                  metrics = metric_set(roc_auc),
                                  control = tunedModel)
# ----------------------------

# XG Boosted Forest
XGboosted_model <- boost_tree(tree_depth=15,
                              trees=1500,
                              learn_rate=.1) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# XG Boost Workflow
XGboost_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(XGboosted_model)

# Resample for Stack as a 'tunedModel'
XGboosted_results_stack <- fit_resamples(XGboost_wf,
                                         resamples = folds,
                                         metrics = metric_set(accuracy,roc_auc),
                                         control = tunedModel)

 

# Setup the Stack and Add Candidates
my_stack <- stacks() %>%
  add_candidates(XGboosted_results_stack) %>% 
  add_candidates(rf_results_stack) #

# Blend Predictions and Fit Members to Create a Stacked Model
stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

# Make Predictions from the Stacked Model
stack_preds <- stack_mod %>%
  predict(new_data = test, type = "class")

# Prepare the Predictions for Output
stack_output <- tibble(Id = test$Id, Cover_Type = stack_preds$.pred_class)

# Write the Predictions
vroom_write(stack_output, "rrrrk.csv", delim = ",")
########

boost_mod <- boost_tree(tree_depth=tune(),
                        trees=tune(),
                        learn_rate=tune()) |>
  set_engine("lightgbm") |>
  set_mode("classification")

boost_wf <- workflow() |>
  add_recipe(recipe) |>
  add_model(boost_mod)

## Set up grid and tuning values
boost_tuning_params <- grid_regular(tree_depth(),
                                    trees(),
                                    learn_rate(),
                                    levels = 10)

##Split data for CV
boost_folds <- vfold_cv(train, v = 6, repeats = 1)

##Run the CV
boost_CV_results <- boost_wf |>
  tune_grid(resamples = boost_folds,
            grid = boost_tuning_params,
            metrics = metric_set(roc_auc, f_meas, sens, recall, 
                                 precision, accuracy))
#Find best tuning parameters
boost_best_tune <- boost_CV_results |>
  select_best(metric = "accuracy") 

##finalize the workflow and fit it
boost_final <- boost_wf |>
  finalize_workflow(boost_best_tune) |>
  fit(data = train)

##predict
boost_preds <- boost_final |>
  predict(new_data = test, type = "class")

boost_kaggle <- boost_preds|>
  bind_cols(test) |>
  select(Id, .pred_class) |>
  rename(Cover_Type = .pred_class)

##write out file
vroom_write(x=boost_kaggle, "./submission.csv", delim=",")


