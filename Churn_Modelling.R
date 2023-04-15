library(tidyverse)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(scorecard)
library(readr)
library(dplyr)
library(ggsci)
library(yardstick)



data <- read_csv('Churn_Modelling .csv')

data %>% glimpse()


data$Exited %>% table() %>% prop.table()


data <- data %>% subset(select = -Surname)

ivars <- data %>% iv(y = 'Exited') %>% 
  as_tibble() %>% 
  mutate(info_value = round(info_value,3)) %>% 
  arrange(desc(info_value))

ivars






## Dropping the cols with no information value

ivars %>% filter(info_value>0.02)

ivars <- ivars[[1]]


## Combining the target column and columns with iv greater than 0.02

data <- data %>% select(Exited, ivars)

data %>% dim()


bins <- data %>% woebin("Exited")

bins$Age %>% dim()
bins$Age %>% as.tibble() %>% view()
bins$Age %>% woebin_plot()


data_split <- data %>% split_df('Exited',ratio = 0.8,seed =  101)

data_split


train_woe <- data_split$train %>% woebin_ply(bins) 
test_woe <- data_split$test %>% woebin_ply(bins)


train_woe %>% dim()
test_woe %>% dim()


test_woe %>% view()


names <- train_woe %>% names() %>% gsub("_woe","",.)   


names(train_woe) <- names;
names(test_woe) <- names;

names


## Multicolliniearity

target <- 'Exited'

features <- train_woe %>% select(-Exited) %>% names()

features


f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))

glm <- glm(f, data = train_woe, family = "binomial")

glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

glm %>% summary()


while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,"variable"]
  afterVIF <- afterVIF$variable
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = train_woe, family = "binomial")
}


features <- glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable)


## Model

h2o.init()

train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
test_h2o <- test_woe %>% select(target,features) %>% as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)




## To Eliminate the features with p-value greater than 0.05
while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
}
model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3))

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)


## Variable importance

h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)



## Model Evaluation

pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict) ## Returns the probabilities for p0 and p1

## Finding threshold by max f1 score

model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')


## ROC curve and AUC score = 0.8223

eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = data_split$test$Exited %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")



## Confusion Matrix
model %>% h2o.performance(test_h2o) %>% h2o.precision()

model %>% 
  h2o.confusionMatrix(test_h2o) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))


