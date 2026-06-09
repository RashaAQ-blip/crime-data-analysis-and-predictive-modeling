## =========================================================================
## Project: Crime Data Analysis & Predictive Modeling
## Author: Rasha Quadri
## Date: 2026-03-26
## Description: Complete data science workflow evaluating Los Angeles crime 
##              trends, demographic distributions, and binary classification 
##              modeling for law enforcement arrest outcomes using Logistic Regression.
## =========================================================================

## --- Load Required Libraries ---
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)


## 1. DATA LOADING & CLEANING ----------------------------------------------

# Note: Ensure the raw zipped data archive is available in your active directory
LA_crime <- read.csv(unzip("Crime_Data_from_2020_to_Present.csv.zip",
                           "Crime_Data_from_2020_to_Present.csv"))

# Sample 50,000 observations for computational and memory efficiency
set.seed(123)
LA_crime <- LA_crime[sample(nrow(LA_crime), 50000), ]

# Format Temporal Features
LA_crime$DATE.OCC <- as.Date(LA_crime$DATE.OCC, format = "%m/%d/%Y")
LA_crime$Year <- format(LA_crime$DATE.OCC, "%Y")

# Recode Empty Character Strings as Explicit Missing Values (NA)
LA_crime$Vict.Descent[LA_crime$Vict.Descent == ""] <- NA
LA_crime$Vict.Sex[LA_crime$Vict.Sex == ""] <- NA

# Convert Categorical Features to Structural R Factors
LA_crime$Vict.Sex <- as.factor(LA_crime$Vict.Sex)
LA_crime$Vict.Descent <- as.factor(LA_crime$Vict.Descent)
LA_crime$Status <- as.factor(LA_crime$Status)
LA_crime$AREA.NAME <- as.factor(LA_crime$AREA.NAME)
LA_crime$Crm.Cd.Desc <- as.factor(LA_crime$Crm.Cd.Desc)

# Audit Missing Value Positions Across Dataset
is.na(LA_crime)

# Isolate Specific Target Features and Predictors for Modeling Context
LA_crime_model <- LA_crime %>%
  select(Vict.Age, Vict.Sex, Vict.Descent, AREA.NAME, Crm.Cd.Desc, Status, Year)

# Handle Missing Observations via Listwise Deletion
LA_crime_model <- na.omit(LA_crime_model)


## 2. EXPLORATORY DATA ANALYSIS --------------------------------------------

### a) Longitudinal Crime Trend Trajectory (2020-2024)
crime_year <- LA_crime %>%
  count(Year)

ggplot(crime_year, aes(x = Year, y = n, group = 1)) +
  geom_line() +
  geom_point() +
  labs(title = "Crime Trend (2020-2024)",
       x = "Year",
       y = "Number of Crimes") +
  theme_minimal()

### b) Frequency Distribution Matrix of Victim Ethnicity Profiles
victim_eth <- LA_crime %>%
  count(Vict.Descent) %>%
  arrange(desc(n))

ggplot(victim_eth, aes(x = reorder(Vict.Descent, -n), y = n)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Crimes by Victim Ethnicity",
       x = "Ethnicity Code",
       y = "Count") +
  theme_minimal()


## 3. PREDICTIVE MODELING: ARREST OUTCOMES ---------------------------------

# Audit Case Status Class Distributions
table(LA_crime_model$Status)

# Feature Engineering: Construct Binary Target Factor for Arrest Outcomes
# (AA = Adult Arrest, JA = Juvenile Arrest)
LA_crime_model$Arrest <- ifelse(LA_crime_model$Status %in% c("AA", "JA"), 1, 0)
LA_crime_model$Arrest <- as.factor(LA_crime_model$Arrest)

# Reproducible Data Partitioning (80/20 Train/Test Split)
set.seed(123)
train_index <- sample(1:nrow(LA_crime_model), 0.8 * nrow(LA_crime_model))

train <- LA_crime_model[train_index, ]
test <- LA_crime_model[-train_index, ]

# Fit Main Effects Logistic Regression Framework
model <- glm(Arrest ~ Vict.Age + Vict.Sex + Vict.Descent + AREA.NAME,
             data = train,
             family = "binomial")

summary(model)

# Extract Probabilistic Prediction Vectors
prob <- predict(model, test, type = "response")

# Deploy Optimized Classification Decision Threshold (t = 0.1)
pred <- ifelse(prob > 0.1, 1, 0)
pred <- as.factor(pred)


## 4. MODEL EVALUATION & DIAGNOSTICS ---------------------------------------

# Generate Comprehensive Multi-Class Confusion Matrix Metrics
conf_mat <- confusionMatrix(pred, test$Arrest, positive = "1")
conf_mat

# Isolate Specific Multi-Threshold Diagnostic Ratios
precision <- conf_mat$byClass["Precision"]
recall <- conf_mat$byClass["Recall"]
f1 <- conf_mat$byClass["F1"]

cat("--- Discrete Threshold Matrix (t = 0.1) ---\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1, "\n")


### Resampling Infrastructure: 5-Fold Stratified Cross-Validation
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)

cv_model <- train(
  Arrest ~ Vict.Age + Vict.Sex + Vict.Descent + AREA.NAME,
  data = train,
  method = "glm",
  family = "binomial",
  trControl = train_control
)

print("--- Cross-Validation Summary Results ---")
cv_model


### Discriminatory Performance Profiling via ROC Curve Analysis
roc_curve <- roc(test$Arrest, prob)
plot(roc_curve, main = "ROC Curve - Arrest Predictive Framework", col = "blue")

# Extract Continuous Area Under the Curve (AUC)
auc(roc_curve)

