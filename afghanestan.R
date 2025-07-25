# ==== Setup ====
install.packages(c("tidyverse", "tidytext", "readtext", "textclean", "textdata", 
                   "widyr", "igraph", "ggraph", "tm", "topicmodels"))

library(tidyverse)
library(tidytext)
library(readtext)
library(textclean)
library(textdata)
library(widyr)
library(igraph)
library(ggraph)
library(tm)
library(topicmodels)

# ==== 1. Read and preprocess ====
file_path <- file.choose()
raw_text <- readtext(file_path, encoding = "UTF-8")$text

clean_text <- raw_text %>%
  replace_non_ascii() %>%
  replace_html() %>%
  replace_contraction() %>%
  replace_emoji() %>%
  tolower() %>%
  str_replace_all("[0-9]+", "") %>%
  str_replace_all("[^a-z\\s]", "") %>%
  str_squish()

# ==== 2. Convert to tibble and tokenize ====
text_df <- tibble(line = 1, text = clean_text) %>%
  unnest_tokens(word, text)

# ==== 3. Remove ultra-common English stopwords and extras ====
data("stop_words")

extra_stopwords <- c(
  # domain-specific junk
  "iran", "afghanistan", "afghan", "unhcr", "grandi", 
  "kazemi", "minister", "refugees", "foreign", "national", "nationals",
  "islamic", "republic", "government", "tehran", "education", 
  "students", "school", "service", "services", "officials", "department",
  "meeting", "meeting", "support", "program", "platform", "certificate",
  "countries", "country", "people", "children", "year", "years", 
  "iranian", "reported", "statement", "according", "report", "website",
  "khabar", "online", "wrote", "news", "agency", "based"
)

all_stops <- bind_rows(stop_words, tibble(word = extra_stopwords, lexicon = "custom"))

text_df <- text_df %>%
  anti_join(all_stops, by = "word") %>%
  filter(str_detect(word, "^[a-z]+$"))  # only words, no numbers/punctuation

# ==== 4. Word frequency ====
text_df %>%
  count(word, sort = TRUE) %>%
  top_n(25) %>%
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(title = "Top 25 Most Frequent Words (After Deep Cleaning)")

# 5. Sentiment analysis (Bing lexicon)
bing_sentiment <- text_df %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE)

# Plot sentiment
bing_sentiment %>%
  group_by(sentiment) %>%
  top_n(10, n) %>%
  ggplot(aes(reorder(word, n), n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ sentiment, scales = "free_y") +
  coord_flip() +
  labs(title = "Sentiment Analysis", x = "Word", y = "Frequency")

# ==== 6. Co-occurrence network ====
# Group words into artificial chunks of ~50 lines
text_df <- text_df %>%
  mutate(section = row_number() %/% 50)

word_pairs <- text_df %>%
  pairwise_count(word, section, sort = TRUE) %>%
  filter(n >= 5)

word_pairs %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE) +
  geom_node_point(color = "tomato", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE, size = 4) +
  theme_void() +
  labs(title = "Word Co-occurrence Network")

# ==== 7. Topic Modeling (LDA) ====
dtm <- text_df %>%
  count(section, word) %>%
  cast_dtm(section, word, n)

lda_model <- LDA(dtm, k = 2, control = list(seed = 1234))

topics <- tidy(lda_model, matrix = "beta")

top_terms <- topics %>%
  group_by(topic) %>%
  top_n(8, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms %>%
  ggplot(aes(reorder_within(term, beta, topic), beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  scale_x_reordered() +
  coord_flip() +
  labs(title = "Top Words per Topic", x = "Term", y = "Beta")
