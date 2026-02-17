# ============================================================
# FIX DATE FORMATTING IN supportTickets_fixed.csv
# ============================================================

library(tidyverse)
library(lubridate)

# --- Load fixed file ---
file_path <- "/Users/Apple/Desktop/Hubspot/data/processed/"
tickets <- read_csv(paste0(file_path, "supportTickets_fixed.csv"))

# --- Check current formats ---
cat("=== BEFORE FIXING ===\n")
cat("created_date samples:\n")
print(head(tickets$created_date, 10))
cat("\nclosed_date samples:\n")
print(head(tickets$closed_date, 10))
cat("\ncsat_date samples:\n")
print(head(tickets$csat_date, 10))

# --- Parse and standardize all date columns to YYYY-MM-DD ---
tickets <- tickets %>%
  mutate(
    created_date = as.Date(ymd_hms(created_date, quiet = TRUE)),
    closed_date  = as.Date(ymd_hms(closed_date, quiet = TRUE)),
    csat_date    = as.Date(ymd_hms(csat_date, quiet = TRUE))
  )

# --- Verify ---
cat("\n=== AFTER FIXING ===\n")
cat("created_date samples:\n")
print(head(tickets$created_date, 10))
cat("\nclosed_date samples:\n")
print(head(tickets$closed_date, 10))
cat("\ncsat_date samples:\n")
print(head(tickets$csat_date, 10))

cat("\n--- Date Ranges ---\n")
cat("created_date: ", as.character(min(tickets$created_date, na.rm=T)), " to ",
    as.character(max(tickets$created_date, na.rm=T)), "\n")
cat("closed_date:  ", as.character(min(tickets$closed_date, na.rm=T)), " to ",
    as.character(max(tickets$closed_date, na.rm=T)), "\n")
cat("csat_date:    ", as.character(min(tickets$csat_date, na.rm=T)), " to ",
    as.character(max(tickets$csat_date, na.rm=T)), "\n")

# --- Save back ---
write_csv(tickets, paste0(file_path, "supportTickets_fixed.csv"))
cat("\n✅ Dates standardized to YYYY-MM-DD and saved.\n")