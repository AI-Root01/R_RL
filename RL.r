library(dplyr)
library(readr)
library(glmnet)
library(broom)

# Configuración inicial de rutas de archivos
train_folder <- "../Dataset/zonas_train"
test_folder <- "../Dataset/zona_test"

# Listas de archivos de entrenamiento y prueba
train_files <- paste0("zona_", c(...), "_entrenamiento.csv")

test_files <- paste0("zona_", c(...), "_prueba.csv")

# Características y targets a predecir
features_mapping <- list(
  Oxig_Disl_Max = c('...', '...', '...'),
  Oxig_Disl_Media = c('...', '...'),
  Oxig_Disl_Min = c('...', '...', '...')
)

# Modelos a implementar
models <- list(
  LMR = "lm",
  Lasso = "glmnet",
  Ridge = "glmnet",
  ElasticNet = "glmnet",
  `Multi-task Lasso` = "glmnet",
  `SAR-l1` = "glmnet",
  `SADL-I` = "glmnet",
  `SADL-2` = "glmnet",
  GLM = NULL  # Placeholder para GLM
)

# Función para cargar los datos de un archivo CSV
load_data <- function(file_path) {
  read_csv(file_path)
}

# Función para preparar los datos: separar características y target
prepare_data <- function(df, features, target_to_predict) {
  X <- df %>% select(all_of(features))
  y <- df[[target_to_predict]]
  list(X = X, y = y)
}

# Función para normalizar los datos
scale_data <- function(X_train, X_test) {
  scaler <- preProcess(X_train, method = c("center", "scale"))
  X_train_scaled <- predict(scaler, X_train)
  X_test_scaled <- predict(scaler, X_test)
  list(X_train_scaled = X_train_scaled, X_test_scaled = X_test_scaled)
}

# Lista para almacenar resultados de los modelos
all_results <- list()

# Evaluar cada modelo para cada zona
for (i in seq_along(train_files)) {
  train_file <- train_files[i]
  test_file <- test_files[i]
  
  # Cargar datos de entrenamiento y prueba
  train_data <- load_data(file.path(train_folder, train_file))
  test_data <- load_data(file.path(test_folder, test_file))

  cat("Evaluando zona:", strsplit(train_file, "_")[[1]][2], "\n")

  # Evaluar modelos para cada target
  for (target_col in names(features_mapping)) {
    features <- features_mapping[[target_col]]
    
    # Preparar los datos
    data_train <- prepare_data(train_data, features, target_col)
    data_test <- prepare_data(test_data, features, target_col)

    # Normalizar los datos
    scaled_data <- scale_data(data_train$X, data_test$X)
    X_train_scaled <- scaled_data$X_train_scaled
    X_test_scaled <- scaled_data$X_test_scaled
    
    # Evaluar cada modelo
    for (model_name in names(models)) {
      model <- models[[model_name]]
      
      if (model_name == "GLM") {
        # Modelo GLM para predicción de una única columna de target
        gl_model <- glm(y ~ ., data = as.data.frame(cbind(y = data_train$y, X_train_scaled)), family = gaussian())
        y_pred_train <- predict(gl_model, newdata = as.data.frame(cbind(y = data_train$y, X_train_scaled)))
        y_pred_test <- predict(gl_model, newdata = as.data.frame(cbind(y = data_test$y, X_test_scaled)))
        
        r2_train <- 1 - sum((data_train$y - y_pred_train) ^ 2) / sum((data_train$y - mean(data_train$y)) ^ 2)
        r2_test <- 1 - sum((data_test$y - y_pred_test) ^ 2) / sum((data_test$y - mean(data_test$y)) ^ 2)

        # Almacenar los resultados
        all_results[[length(all_results) + 1]] <- list(Zona = strsplit(train_file, "_")[[1]][2],
                                                         OD = target_col,
                                                         Model = "GLM",
                                                         R2_Train = r2_train,
                                                         R2_Test = r2_test)
        cat(sprintf("Modelo GLM - OD: %s, R2_Train: %.4f, R2_Test: %.4f\n", target_col, r2_train, r2_test))
      } else {
        # Otros modelos (Regresión, Lasso, Ridge, ElasticNet, etc.)
        if (model_name %in% c("Lasso", "Ridge", "ElasticNet", "Multi-task Lasso", "SAR-l1", "SADL-I", "SADL-2")) {
          alpha <- ifelse(model_name == "Lasso", 1, ifelse(model_name == "Ridge", 0, 0.5))
          glm_model <- cv.glmnet(as.matrix(X_train_scaled), data_train$y, alpha = alpha)
          y_pred_train <- predict(glm_model, s = "lambda.min", newx = as.matrix(X_train_scaled))
          y_pred_test <- predict(glm_model, s = "lambda.min", newx = as.matrix(X_test_scaled))

          r2_train <- 1 - sum((data_train$y - y_pred_train) ^ 2) / sum((data_train$y - mean(data_train$y)) ^ 2)
          r2_test <- 1 - sum((data_test$y - y_pred_test) ^ 2) / sum((data_test$y - mean(data_test$y)) ^ 2)

          # Almacenar los resultados
          all_results[[length(all_results) + 1]] <- list(Zona = strsplit(train_file, "_")[[1]][2],
                                                           OD = target_col,
                                                           Model = model_name,
                                                           R2_Train = r2_train,
                                                           R2_Test = r2_test)
          cat(sprintf("Modelo %s - OD: %s, R2_Train: %.4f, R2_Test: %.4f\n", model_name, target_col, r2_train, r2_test))
        }
      }
    }
  }
}

# Convertir resultados a DataFrame
results_df <- bind_rows(all_results)

# Reorganizar resultados para mejor presentación
organized_results <- results_df %>%
  pivot_wider(names_from = c(OD, Model), values_from = c(R2_Train, R2_Test))

# Guardar el archivo CSV con los resultados organizados
write_csv(organized_results, "Resultados_Organizados.csv")

cat("\nResultados R2 organizados por zona y modelo guardados '\n")
