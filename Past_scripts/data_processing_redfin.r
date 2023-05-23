# new dataset ------------------
df_redfin <- read.csv('Data\\city_market_tracker.tsv000', sep = '\t')


df_redfin_2023_mar <- df_redfin %>%
  mutate(
    period_begin = period_begin %>% as.Date(),
    period_end = period_end %>% as.Date(),
  ) %>%
  select(
    !period_duration
  ) %>%
  filter(
    (period_begin %>% data.table::year()) == 2023 &
      (period_begin %>% data.table::month()) == 3
  )

nrow(df_redfin)
nrow(df_redfin_2023_mar)

write.csv(
  df_redfin_2023_mar, 
  file = 'Data\\Redfin_city_market_tracker_2023_3.csv'
)
