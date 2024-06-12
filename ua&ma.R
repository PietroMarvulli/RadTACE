library(readr)
library(gtsummary)

data <- read.csv("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\clinical_data_II.csv")
data <- data %>% select(c(-ID,-TTP,-Death_1_StillAliveorLostToFU_0,-Censored_0_progressed_1))

# table_stats <-
#   data %>%
#     tbl_summary(by = target) %>%
#     add_p() %>%
#     bold_labels()
# table_stats

table_ua <- tbl_uvregression(data, method = glm, y = target) %>%
  add_global_p() %>%
  bold_p(t = 0.10) %>%
  bold_labels() %>%
  italicize_levels()
# table_ua

p_values <- table_ua$table_body$p.value
features <- table_ua$table_body$variable
selected_features <- list()
for (i in seq_along(p_values)) {
  if (p_values[i] <= 0.1) {
    selected_features <- append(selected_features, features[i])
  }
}
selected_features <- append(selected_features, "target")
data_ma <- data %>% select(all_of(unlist(selected_features)))
lr <- glm(target ~ ., data = data_ma)
lr %>%
  tbl_regression() %>%
    add_global_p() %>%
    bold_p(t = 0.10) %>%
    bold_labels() %>%
    italicize_levels()
