# Healthcare Cost Analysis in R
# Using tidymodels for ML workflow
# Author: Rakesh Budige

library(tidyverse)
library(tidymodels)
library(vip)  # Variable importance plots

# Load data
patients <- read_csv("data/sample/patient_data_full.csv")

# Feature engineering
patients <- patients %>%
  mutate(
    age_group = case_when(
      age < 45 ~ "18-44",
      age < 65 ~ "45-64",
      age < 75 ~ "65-74",
      TRUE ~ "75+"
    ),
    bmi_category = case_when(
      bmi < 18.5 ~ "Underweight",
      bmi < 25 ~ "Normal",
      bmi < 30 ~ "Overweight",
      TRUE ~ "Obese"
    ),
    comorbidity_score = diabetes + heart_disease
  )

# Visualizations
p <- ggplot(patients, aes(x = age, y = healthcare_cost)) +
  geom_point(aes(color = factor(comorbidity_score)), alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Age vs Healthcare Cost",
       color = "Comorbidities") +
  theme_minimal()

ggsave("output/cost_vs_age.png", p, width = 10, height = 6)

# Export for Power BI
write_csv(patients, "output/patient_data_for_powerbi.csv")
