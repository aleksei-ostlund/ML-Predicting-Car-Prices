#load packages

library(tidyverse)
library(caret)

#import data file
cars_data <- read.table("imports-85.data", sep = ",")

#convert to tibble
cars_data <- as_tibble(cars_data)

#name columns according to description file
cars_data_cleaned <- rename(cars_data, 
                    "symboling" = "V1",
                    "normalized_losses" = "V2",
                    "make" = "V3",
                    "fuel_type" = "V4", 
                    "aspiration" = "V5",
                    "num_of_doors" = "V6",
                    "body_style" = "V7",
                    "drive_wheels" = "V8",
                    "engine_location" = "V9",
                    "wheel_base" = "V10",
                    "length" = "V11",
                    "width" = "V12",
                    "height" = "V13",
                    "curb_weight" = "V14",
                    "engine_type" = "V15",
                    "num_of_cylinders" = "V16",
                    "engine_size" = "V17",
                    "fuel_system" = "V18",
                    "bore" = "V19",
                    "stroke" = "V20",
                    "compression_ratio" = "V21",
                    "horsepower" = "V22",
                    "peak_rpm" = "V23",
                    "city_mpg" = "V24",
                    "highway_mpg" = "V25",
                    "price" = "V26"
)

#select quantitative columns
cars_numeric <- cars_data_cleaned %>%
  select(-make, -fuel_type, -aspiration, -body_style, -drive_wheels,
         -engine_location, -engine_type, -fuel_system)

#check unique values

print(unique(cars_numeric$num_of_doors))
print(unique(cars_numeric$num_of_cylinders))

#change written numbers to digits
cars_numeric <- cars_numeric%>%
  mutate(num_of_doors=  recode(cars_numeric$num_of_doors, 
                               "two" = 2, 
                               "four" = 4)) %>%
  mutate(num_of_cylinders = recode(cars_numeric$num_of_cylinders, 
                                   "four" = 4,
                                   "six" = 6,
                                   "five" = 5,
                                   "three" = 3,
                                   "twelve" = 12,
                                   "two" = 2,
                                   "eight" = 8
                                   ))

#remove NA rows
cars_numeric_clean <- cars_numeric %>%
  na_if("?") %>%
  na.omit()

#check column types
as_tibble(str(cars_numeric_clean))

#convert characters to numeric
cars_numeric_clean <- cars_numeric_clean %>%
  transform(num_of_doors = as.numeric(num_of_doors),
           normalized_losses = as.numeric(normalized_losses),
           bore = as.numeric(bore),
           stroke = as.numeric(stroke),
           horsepower = as.numeric(horsepower),
           peak_rpm = as.numeric(peak_rpm),
           price = as.numeric(price)
             )

#lattice plots

#pos correlation
featurePlot(cars_numeric_clean$horsepower, cars_numeric_clean$price)

#pos correlation
featurePlot(cars_numeric_clean$engine_size, cars_numeric_clean$price)

#pos correlation
featurePlot(cars_numeric_clean$curb_weight, cars_numeric_clean$price)

#neg correlation
featurePlot(cars_numeric_clean$city_mpg, cars_numeric_clean$price)

#partition data with 70% of data allocated to training
train_indices <- createDataPartition(y=cars_numeric_clean[["price"]],
                                     p = 0.7,
                                     list=FALSE)

train_listings <- cars_numeric_clean[train_indices,]
test_listings <- cars_numeric_clean[-train_indices,]
knn_grid <- expand.grid(k = 1:20)

five_fold_control <- trainControl(method = "cv", number = 5)

#set up k nearest neighbors models

cars_knn_model <- train(price ~ horsepower + engine_size + curb_weight + city_mpg,
                        data = train_listings,
                        method = "knn",
                        trControl = five_fold_control,
                        preProcess = c("center", "scale"),
                        tuneGrid = knn_grid
)

cars_knn_model_2 <- train(price ~ horsepower + engine_size + curb_weight,
                        data = train_listings,
                        method = "knn",
                        trControl = five_fold_control,
                        preProcess = c("center", "scale"),
                        tuneGrid = knn_grid
)

cars_knn_model_3 <- train(price ~ horsepower + engine_size,
                        data = train_listings,
                        method = "knn",
                        trControl = five_fold_control,
                        preProcess = c("center", "scale"),
                        tuneGrid = knn_grid
)

#model 1 predictions (RMSE 1905)
test_predictions <- predict(cars_knn_model, newdata = test_listings)
as_tibble(test_predictions)

test_listings <- test_listings %>%
  mutate(error = price - test_predictions)

test_listings <- test_listings %>%
  mutate(sq_error = (error^2))

rmse1 <- postResample(pred = test_predictions, obs = test_listings$price) 


#model 2 predictions (RMSE 2509)
test_predictions_2 <- predict(cars_knn_model_2, newdata = test_listings)
as_tibble(test_predictions)


rmse2 <- postResample(pred = test_predictions_2, obs = test_listings$price) 


#model 3 predictions (RMSE 2412)
test_predictions_3 <- predict(cars_knn_model_3, newdata = test_listings)
as_tibble(test_predictions)


rmse3 <- postResample(pred = test_predictions_3, obs = test_listings$price) %>%

#model 1 has lowest RMSE
model <- c(1,2,3)
rmse <- c(rmse1[[1]],rmse2[[1]],rmse3[[1]])
compared_rmse <- tibble(model,rmse) %>%
  arrange(rmse) %>%
  print()
