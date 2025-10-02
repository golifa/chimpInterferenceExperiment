# Load required libraries ----
library(tidyverse)
library(dplyr)
library(ggplot2)
library(readr)

# ============================================================================ #
# EXCLUSION CRITERIA SETTINGS 
# ============================================================================ #
DPRIME_SD_CUTOFF <- 2.0
PERFORMANCE_SD_CUTOFF <- 2.0 
RT_SD_CUTOFF <- 2.0 

print("=== EXCLUSION CRITERIA SETTINGS ===")
print(paste("D-prime SD cutoff:", DPRIME_SD_CUTOFF))
print(paste("Performance SD cutoff:", PERFORMANCE_SD_CUTOFF))
print(paste("RT SD cutoff:", RT_SD_CUTOFF))
print("========================================")

# READ FILES ----
data_directory <- "CHANGE TO PATH"
csv_files <- list.files(data_directory, pattern = "*_results.csv", full.names = TRUE)

print(paste("Found", length(csv_files), "data files"))
print("Files found:")
print(basename(csv_files))

all_data <- data.frame()
for(file in csv_files) {
  temp_data <- read_csv(file, show_col_types = FALSE)
  subject_id <- gsub("_results\\.csv$", "", basename(file))
  temp_data$actual_subject <- subject_id  
  temp_data$file_name <- basename(file)   
  all_data <- rbind(all_data, temp_data)
  print(paste("Loaded:", basename(file), "- Subject ID:", subject_id, "- Rows:", nrow(temp_data)))
}

print(paste("Total rows loaded:", nrow(all_data)))
print("Actual subjects from filenames:")
print(sort(unique(all_data$actual_subject)))
print(paste("Unique subjects:", length(unique(all_data$actual_subject))))

data <- all_data

# DATA PREPROCESSING ALL TRIALS  ----
clean_data <- data %>%
  filter(!is.na(response), !is.na(rt), !is.na(correct)) %>%
  mutate(
    accuracy = ifelse(correct == "CORRECT", 1, 0),
    monkey_alignment = ifelse(grepl("misaligned", block_condition), "misaligned", "aligned"),
    face_gender = ifelse(grepl("female", block_condition), "female", "male"),
    species = ifelse(grepl("human", category), "human", "monkey"),
    subject = actual_subject
  )

print(paste("Rows after basic filtering (ALL TRIALS):", nrow(clean_data)))
print("top_same distribution:")
print(table(clean_data$top_same, useNA = "ifany"))

# D-PRIME CALCULATION FUNCTIONS ----
calculate_dprime <- function(hit_rate, fa_rate) {
  hit_rate_adj <- pmax(pmin(hit_rate, 0.99), 0.01)
  fa_rate_adj <- pmax(pmin(fa_rate, 0.99), 0.01)
  dprime <- qnorm(hit_rate_adj) - qnorm(fa_rate_adj)
  return(dprime)
}

calculate_criterion <- function(hit_rate, fa_rate) {
  hit_rate_adj <- pmax(pmin(hit_rate, 0.99), 0.01)
  fa_rate_adj <- pmax(pmin(fa_rate, 0.99), 0.01)
  criterion <- -0.5 * (qnorm(hit_rate_adj) + qnorm(fa_rate_adj))
  return(criterion)
}

# D-PRIME CALCULATION ----
all_data_clean <- data %>%
  filter(!is.na(response), !is.na(rt), !is.na(correct)) %>%
  mutate(
    accuracy = ifelse(correct == "CORRECT", 1, 0),
    monkey_alignment = ifelse(grepl("misaligned", block_condition), "misaligned", "aligned"),
    face_gender = ifelse(grepl("female", block_condition), "female", "male"),
    species = ifelse(grepl("human", category), "human", "monkey"),
    subject = actual_subject
  )

dprime_data <- all_data_clean %>%
  group_by(subject, species, monkey_alignment) %>%
  summarise(
    hits = sum(accuracy == 1 & top_same == "SAME", na.rm = TRUE),
    same_trials = sum(top_same == "SAME", na.rm = TRUE),
    false_alarms = sum(accuracy == 0 & top_same == "DIFF", na.rm = TRUE),
    diff_trials = sum(top_same == "DIFF", na.rm = TRUE),
    hit_rate = ifelse(same_trials > 0, hits / same_trials, NA),
    fa_rate = ifelse(diff_trials > 0, false_alarms / diff_trials, NA),
    .groups = 'drop'
  ) %>%
  filter(!is.na(hit_rate) & !is.na(fa_rate)) %>%
  mutate(
    dprime = calculate_dprime(hit_rate, fa_rate),
    criterion = calculate_criterion(hit_rate, fa_rate)
  )

# PERFORMANCE CALCULATION - ALL TRIALS (CORRECT + INCORRECT) ----
performance_data <- clean_data %>%
  group_by(subject, species, monkey_alignment) %>%
  summarise(
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    n_trials = n(),
    .groups = 'drop'
  )

# RT CALCULATION - ALL TRIALS ----
rt_data <- clean_data %>%
  filter(rt > 0, rt < 3) %>%  # Basic RT filtering, no correctness filter
  group_by(subject, species, monkey_alignment) %>%
  summarise(mean_rt = mean(rt, na.rm = TRUE), .groups = 'drop')

# EXCLUSION CRITERIA ----
print("=== APPLYING EXCLUSION CRITERIA ===")

# Calculate overall d-prime statistics for exclusion
overall_dprime_stats <- dprime_data %>%
  summarise(
    mean_dprime = mean(dprime, na.rm = TRUE),
    sd_dprime = sd(dprime, na.rm = TRUE),
    cutoff_dprime = mean_dprime - DPRIME_SD_CUTOFF * sd_dprime
  )

print(paste("D-prime cutoff:", round(overall_dprime_stats$cutoff_dprime, 3)))

# Calculate overall performance statistics for exclusion
overall_performance_stats <- performance_data %>%
  summarise(
    mean_performance = mean(mean_accuracy, na.rm = TRUE),
    sd_performance = sd(mean_accuracy, na.rm = TRUE),
    cutoff_performance = mean_performance - PERFORMANCE_SD_CUTOFF * sd_performance
  )

print(paste("Performance cutoff:", round(overall_performance_stats$cutoff_performance, 3)))

# Calculate overall RT statistics for exclusion
overall_rt_stats <- rt_data %>%
  group_by(subject) %>%
  summarise(overall_rt = mean(mean_rt, na.rm = TRUE), .groups = 'drop') %>%
  summarise(
    mean_rt = mean(overall_rt, na.rm = TRUE),
    sd_rt = sd(overall_rt, na.rm = TRUE),
    cutoff_rt_high = mean_rt + RT_SD_CUTOFF * sd_rt,
    cutoff_rt_low = pmax(0, mean_rt - RT_SD_CUTOFF * sd_rt)
  )

print(paste("RT high cutoff:", round(overall_rt_stats$cutoff_rt_high, 3)))
print(paste("RT low cutoff:", round(overall_rt_stats$cutoff_rt_low, 3)))

# Identify subjects to exclude
excluded_subjects_dprime <- dprime_data %>%
  group_by(subject) %>%
  summarise(overall_dprime = mean(dprime, na.rm = TRUE)) %>%
  filter(overall_dprime < overall_dprime_stats$cutoff_dprime) %>%
  pull(subject)

excluded_subjects_performance <- performance_data %>%
  group_by(subject) %>%
  summarise(overall_performance = mean(mean_accuracy, na.rm = TRUE)) %>%
  filter(overall_performance < overall_performance_stats$cutoff_performance) %>%
  pull(subject)

excluded_subjects_rt <- rt_data %>%
  group_by(subject) %>%
  summarise(overall_rt = mean(mean_rt, na.rm = TRUE)) %>%
  filter(overall_rt > overall_rt_stats$cutoff_rt_high | overall_rt < overall_rt_stats$cutoff_rt_low) %>%
  pull(subject)

excluded_subjects_combined <- unique(c(excluded_subjects_dprime, excluded_subjects_performance, excluded_subjects_rt))

print(paste("Excluded subjects (d-prime):", paste(excluded_subjects_dprime, collapse = ", ")))
print(paste("Excluded subjects (performance):", paste(excluded_subjects_performance, collapse = ", ")))
print(paste("Excluded subjects (RT outliers):", paste(excluded_subjects_rt, collapse = ", ")))
print(paste("Total excluded subjects:", length(excluded_subjects_combined)))

# Filter out excluded subjects
dprime_data_filtered <- dprime_data %>%
  filter(!subject %in% excluded_subjects_combined)

performance_data_filtered <- performance_data %>%
  filter(!subject %in% excluded_subjects_combined)

rt_data_filtered <- rt_data %>%
  filter(!subject %in% excluded_subjects_combined)

clean_data_filtered <- clean_data %>%
  filter(!subject %in% excluded_subjects_combined)

print(paste("Remaining subjects after exclusion:", length(unique(dprime_data_filtered$subject))))

# STATISTICAL TESTS ----
print("=== STATISTICAL TESTS (FILTERED DATA - ALL TRIALS) ===")

# Initialize test variables
human_dprime_test <- NULL
monkey_dprime_test <- NULL
human_performance_test <- NULL
monkey_performance_test <- NULL
human_rt_test <- NULL
monkey_rt_test <- NULL

# D-prime statistical tests
human_dprime_filtered <- dprime_data_filtered %>% filter(species == "human")
if(nrow(human_dprime_filtered) > 0) {
  human_aligned_dprime <- human_dprime_filtered$dprime[human_dprime_filtered$monkey_alignment == "aligned"]
  human_misaligned_dprime <- human_dprime_filtered$dprime[human_dprime_filtered$monkey_alignment == "misaligned"]
  
  if(length(human_aligned_dprime) > 0 & length(human_misaligned_dprime) > 0) {
    human_dprime_test <- t.test(human_misaligned_dprime, human_aligned_dprime, paired = TRUE)
    print("HUMAN D-PRIME:")
    print(human_dprime_test)
  }
}

monkey_dprime_filtered <- dprime_data_filtered %>% filter(species == "monkey")
if(nrow(monkey_dprime_filtered) > 0) {
  monkey_aligned_dprime <- monkey_dprime_filtered$dprime[monkey_dprime_filtered$monkey_alignment == "aligned"]
  monkey_misaligned_dprime <- monkey_dprime_filtered$dprime[monkey_dprime_filtered$monkey_alignment == "misaligned"]
  
  if(length(monkey_aligned_dprime) > 0 & length(monkey_misaligned_dprime) > 0) {
    monkey_dprime_test <- t.test(monkey_misaligned_dprime, monkey_aligned_dprime, paired = TRUE)
    print("MONKEY D-PRIME:")
    print(monkey_dprime_test)
  }
}

# Performance statistical tests
human_performance_filtered <- performance_data_filtered %>% filter(species == "human")
if(nrow(human_performance_filtered) > 0) {
  human_aligned_acc <- human_performance_filtered$mean_accuracy[human_performance_filtered$monkey_alignment == "aligned"]
  human_misaligned_acc <- human_performance_filtered$mean_accuracy[human_performance_filtered$monkey_alignment == "misaligned"]
  
  if(length(human_aligned_acc) > 0 & length(human_misaligned_acc) > 0) {
    human_performance_test <- t.test(human_misaligned_acc, human_aligned_acc, paired = TRUE)
    print("HUMAN PERFORMANCE (ALL TRIALS):")
    print(human_performance_test)
  }
}

monkey_performance_filtered <- performance_data_filtered %>% filter(species == "monkey")
if(nrow(monkey_performance_filtered) > 0) {
  monkey_aligned_acc <- monkey_performance_filtered$mean_accuracy[monkey_performance_filtered$monkey_alignment == "aligned"]
  monkey_misaligned_acc <- monkey_performance_filtered$mean_accuracy[monkey_performance_filtered$monkey_alignment == "misaligned"]
  
  if(length(monkey_aligned_acc) > 0 & length(monkey_misaligned_acc) > 0) {
    monkey_performance_test <- t.test(monkey_misaligned_acc, monkey_aligned_acc, paired = TRUE)
    print("MONKEY PERFORMANCE (ALL TRIALS):")
    print(monkey_performance_test)
  }
}

# RT statistical tests - ALL TRIALS
human_rt_filtered <- rt_data_filtered %>% filter(species == "human")
if(nrow(human_rt_filtered) > 0) {
  human_aligned_rt <- human_rt_filtered$mean_rt[human_rt_filtered$monkey_alignment == "aligned"]
  human_misaligned_rt <- human_rt_filtered$mean_rt[human_rt_filtered$monkey_alignment == "misaligned"]
  
  if(length(human_aligned_rt) > 0 & length(human_misaligned_rt) > 0) {
    human_rt_test <- t.test(human_misaligned_rt, human_aligned_rt, paired = TRUE)
    print("HUMAN RT (ALL TRIALS):")
    print(human_rt_test)
  }
}

monkey_rt_filtered <- rt_data_filtered %>% filter(species == "monkey")
if(nrow(monkey_rt_filtered) > 0) {
  monkey_aligned_rt <- monkey_rt_filtered$mean_rt[monkey_rt_filtered$monkey_alignment == "aligned"]
  monkey_misaligned_rt <- monkey_rt_filtered$mean_rt[monkey_rt_filtered$monkey_alignment == "misaligned"]
  
  if(length(monkey_aligned_rt) > 0 & length(monkey_misaligned_rt) > 0) {
    monkey_rt_test <- t.test(monkey_misaligned_rt, monkey_aligned_rt, paired = TRUE)
    print("MONKEY RT (ALL TRIALS):")
    print(monkey_rt_test)
  }
}

# SUMMARY STATISTICS FOR PLOTTING ----
conditions <- expand.grid(
  species = c("human", "monkey"),
  monkey_alignment = c("aligned", "misaligned"),
  stringsAsFactors = FALSE
)

# D-prime summary
dprime_summary <- data.frame()
for(i in 1:nrow(conditions)) {
  subset_data <- dprime_data_filtered[
    dprime_data_filtered$species == conditions$species[i] & 
      dprime_data_filtered$monkey_alignment == conditions$monkey_alignment[i], ]
  
  if(nrow(subset_data) > 0) {
    result <- data.frame(
      species = conditions$species[i],
      monkey_alignment = conditions$monkey_alignment[i],
      n_subjects = nrow(subset_data),
      mean_dprime = mean(subset_data$dprime, na.rm = TRUE),
      sd_dprime = sd(subset_data$dprime, na.rm = TRUE),
      se_dprime = sd(subset_data$dprime, na.rm = TRUE) / sqrt(nrow(subset_data))
    )
    dprime_summary <- rbind(dprime_summary, result)
  }
}

# Performance summary
performance_summary <- data.frame()
for(i in 1:nrow(conditions)) {
  subset_data <- performance_data_filtered[
    performance_data_filtered$species == conditions$species[i] & 
      performance_data_filtered$monkey_alignment == conditions$monkey_alignment[i], ]
  
  if(nrow(subset_data) > 0) {
    result <- data.frame(
      species = conditions$species[i],
      monkey_alignment = conditions$monkey_alignment[i],
      n_subjects = nrow(subset_data),
      mean_accuracy = mean(subset_data$mean_accuracy, na.rm = TRUE),
      sd_accuracy = sd(subset_data$mean_accuracy, na.rm = TRUE),
      se_accuracy = sd(subset_data$mean_accuracy, na.rm = TRUE) / sqrt(nrow(subset_data))
    )
    performance_summary <- rbind(performance_summary, result)
  }
}

# RT summary
rt_summary <- data.frame()
for(i in 1:nrow(conditions)) {
  subset_data <- rt_data_filtered[
    rt_data_filtered$species == conditions$species[i] & 
      rt_data_filtered$monkey_alignment == conditions$monkey_alignment[i], ]
  
  if(nrow(subset_data) > 0) {
    result <- data.frame(
      species = conditions$species[i],
      monkey_alignment = conditions$monkey_alignment[i],
      n_subjects = nrow(subset_data),
      mean_rt = mean(subset_data$mean_rt, na.rm = TRUE),
      sd_rt = sd(subset_data$mean_rt, na.rm = TRUE),
      se_rt = sd(subset_data$mean_rt, na.rm = TRUE) / sqrt(nrow(subset_data))
    )
    rt_summary <- rbind(rt_summary, result)
  }
}

# Print summary statistics
print("=== SUMMARY STATISTICS (ALL TRIALS) ===")
print("D-prime summary:")
print(dprime_summary)
print("\nPerformance summary:")
print(performance_summary)
print("\nRT summary:")
print(rt_summary)

# PLOTTING FUNCTIONS ----
get_significance_label <- function(p_value) {
  if(is.null(p_value)) return("ns")
  if(p_value < 0.001) return("***")
  if(p_value < 0.01) return("**")
  if(p_value < 0.05) return("*")
  return("ns")
}

# D-prime plot
p_dprime <- ggplot() +
  geom_line(data = dprime_data_filtered, 
            aes(x = monkey_alignment, y = dprime, group = subject),
            color = "gray70", alpha = 0.5, size = 0.3) +
  geom_point(data = dprime_data_filtered,
             aes(x = monkey_alignment, y = dprime),
             color = "gray50", alpha = 0.6, size = 1.5) +
  geom_col(data = dprime_summary, 
           aes(x = monkey_alignment, y = mean_dprime, fill = monkey_alignment),
           alpha = 0.7, width = 0.6) +
  geom_errorbar(data = dprime_summary,
                aes(x = monkey_alignment,
                    ymin = mean_dprime - se_dprime, 
                    ymax = mean_dprime + se_dprime),
                width = 0.2, size = 0.8, color = "black") +
  facet_wrap(~species, scales = "free_y") +
  labs(title = "D-prime by Species and Chimpanzee Alignment (All Trials)",
       x = "Chimpanzee Alignment", y = "d-prime") +
  theme_minimal(base_size = 12) +
  scale_fill_manual(values = c("aligned" = "#02343F", "misaligned" = "#F0EDCC")) +
  theme(legend.position = "none", panel.grid.minor = element_blank())

# Add significance markers for d-prime
dprime_sig_data <- data.frame()
if(!is.null(human_dprime_test)) {
  max_y_human <- max(c(dprime_data_filtered$dprime[dprime_data_filtered$species == "human"],
                       dprime_summary$mean_dprime[dprime_summary$species == "human"] + 
                         dprime_summary$se_dprime[dprime_summary$species == "human"]), na.rm = TRUE)
  dprime_sig_data <- rbind(dprime_sig_data, 
                           data.frame(species = "human", x = 1.5, y = max_y_human * 1.1, 
                                      label = get_significance_label(human_dprime_test$p.value)))
}
if(!is.null(monkey_dprime_test)) {
  max_y_monkey <- max(c(dprime_data_filtered$dprime[dprime_data_filtered$species == "monkey"],
                        dprime_summary$mean_dprime[dprime_summary$species == "monkey"] + 
                          dprime_summary$se_dprime[dprime_summary$species == "monkey"]), na.rm = TRUE)
  dprime_sig_data <- rbind(dprime_sig_data, 
                           data.frame(species = "monkey", x = 1.5, y = max_y_monkey * 1.1, 
                                      label = get_significance_label(monkey_dprime_test$p.value)))
}

if(nrow(dprime_sig_data) > 0) {
  p_dprime <- p_dprime + 
    geom_text(data = dprime_sig_data, aes(x = x, y = y, label = label), size = 6)
}

print(p_dprime)

# Performance plot
p_performance <- ggplot() +
  geom_line(data = performance_data_filtered, 
            aes(x = monkey_alignment, y = mean_accuracy, group = subject),
            color = "gray70", alpha = 0.5, size = 0.3) +
  geom_point(data = performance_data_filtered,
             aes(x = monkey_alignment, y = mean_accuracy),
             color = "gray50", alpha = 0.6, size = 1.5) +
  geom_col(data = performance_summary, 
           aes(x = monkey_alignment, y = mean_accuracy, fill = monkey_alignment),
           alpha = 0.7, width = 0.6) +
  geom_errorbar(data = performance_summary,
                aes(x = monkey_alignment,
                    ymin = mean_accuracy - se_accuracy, 
                    ymax = mean_accuracy + se_accuracy),
                width = 0.2, size = 0.8, color = "black") +
  facet_wrap(~species) +
  labs(title = "Accuracy by Species and Chimpanzee Alignment (All Trials)",
       x = "Chimpanzee Alignment", y = "Accuracy") +
  theme_minimal(base_size = 12) +
  scale_fill_manual(values = c("aligned" = "#02343F", "misaligned" = "#F0EDCC")) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme(legend.position = "none", panel.grid.minor = element_blank())

# Add significance markers for performance
performance_sig_data <- data.frame()
if(!is.null(human_performance_test)) {
  max_y_human_acc <- max(c(performance_data_filtered$mean_accuracy[performance_data_filtered$species == "human"],
                           performance_summary$mean_accuracy[performance_summary$species == "human"] + 
                             performance_summary$se_accuracy[performance_summary$species == "human"]), na.rm = TRUE)
  performance_sig_data <- rbind(performance_sig_data, 
                                data.frame(species = "human", x = 1.5, y = max_y_human_acc * 1.02, 
                                           label = get_significance_label(human_performance_test$p.value)))
}
if(!is.null(monkey_performance_test)) {
  max_y_monkey_acc <- max(c(performance_data_filtered$mean_accuracy[performance_data_filtered$species == "monkey"],
                            performance_summary$mean_accuracy[performance_summary$species == "monkey"] + 
                              performance_summary$se_accuracy[performance_summary$species == "monkey"]), na.rm = TRUE)
  performance_sig_data <- rbind(performance_sig_data, 
                                data.frame(species = "monkey", x = 1.5, y = max_y_monkey_acc * 1.02, 
                                           label = get_significance_label(monkey_performance_test$p.value)))
}

if(nrow(performance_sig_data) > 0) {
  p_performance <- p_performance + 
    geom_text(data = performance_sig_data, aes(x = x, y = y, label = label), size = 6)
}

print(p_performance)

# RT plot
p_rt <- ggplot() +
  geom_line(data = rt_data_filtered, 
            aes(x = monkey_alignment, y = mean_rt, group = subject),
            color = "gray70", alpha = 0.5, size = 0.3) +
  geom_point(data = rt_data_filtered,
             aes(x = monkey_alignment, y = mean_rt),
             color = "gray50", alpha = 0.6, size = 1.5) +
  geom_col(data = rt_summary, 
           aes(x = monkey_alignment, y = mean_rt, fill = monkey_alignment),
           alpha = 0.7, width = 0.6) +
  geom_errorbar(data = rt_summary,
                aes(x = monkey_alignment,
                    ymin = mean_rt - se_rt, 
                    ymax = mean_rt + se_rt),
                width = 0.2, size = 0.8, color = "black") +
  facet_wrap(~species) +
  labs(title = "Reaction Time by Species and Chimpanzee Alignment (All Trials)",
       x = "Chimpanzee Alignment", y = "Mean RT (seconds)") +
  theme_minimal(base_size = 12) +
  scale_fill_manual(values = c("aligned" = "#02343F", "misaligned" = "#F0EDCC")) +
  theme(legend.position = "none", panel.grid.minor = element_blank())

# Add significance markers for RT
rt_sig_data <- data.frame()
if(!is.null(human_rt_test)) {
  max_y_human_rt <- max(c(rt_data_filtered$mean_rt[rt_data_filtered$species == "human"],
                          rt_summary$mean_rt[rt_summary$species == "human"] + 
                            rt_summary$se_rt[rt_summary$species == "human"]), na.rm = TRUE)
  rt_sig_data <- rbind(rt_sig_data, 
                       data.frame(species = "human", x = 1.5, y = max_y_human_rt * 1.05, 
                                  label = get_significance_label(human_rt_test$p.value)))
}
if(!is.null(monkey_rt_test)) {
  max_y_monkey_rt <- max(c(rt_data_filtered$mean_rt[rt_data_filtered$species == "monkey"],
                           rt_summary$mean_rt[rt_summary$species == "monkey"] + 
                             rt_summary$se_rt[rt_summary$species == "monkey"]), na.rm = TRUE)
  rt_sig_data <- rbind(rt_sig_data, 
                       data.frame(species = "monkey", x = 1.5, y = max_y_monkey_rt * 1.05, 
                                  label = get_significance_label(monkey_rt_test$p.value)))
}

if(nrow(rt_sig_data) > 0) {
  p_rt <- p_rt + 
    geom_text(data = rt_sig_data, aes(x = x, y = y, label = label), size = 6)
}

print(p_rt)

print("=== FINAL SUMMARY (ALL TRIALS) ===")
print(paste("Total subjects analyzed:", length(unique(dprime_data_filtered$subject))))
print(paste("Subjects excluded:", length(excluded_subjects_combined)))
if(length(excluded_subjects_combined) > 0) {
  print("Excluded subjects:")
  print(excluded_subjects_combined)
}

# Print test results summary
print("=== STATISTICAL RESULTS SUMMARY (ALL TRIALS) ===")
if(!is.null(human_dprime_test)) {
  print(paste("Human d-prime test p-value:", round(human_dprime_test$p.value, 4)))
}
if(!is.null(monkey_dprime_test)) {
  print(paste("Monkey d-prime test p-value:", round(monkey_dprime_test$p.value, 4)))
}
if(!is.null(human_performance_test)) {
  print(paste("Human performance test p-value:", round(human_performance_test$p.value, 4)))
}
if(!is.null(monkey_performance_test)) {
  print(paste("Monkey performance test p-value:", round(monkey_performance_test$p.value, 4)))
}
if(!is.null(human_rt_test)) {
  print(paste("Human RT test p-value:", round(human_rt_test$p.value, 4)))
}
if(!is.null(monkey_rt_test)) {
  print(paste("Monkey RT test p-value:", round(monkey_rt_test$p.value, 4)))

}
