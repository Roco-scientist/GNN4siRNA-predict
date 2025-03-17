library(tidyverse)

gene_split_data <- read_csv("./GNN4siRNA-predict/accuracy.gene.csv") %>%
  gather(key=Measurement, value=Value, -HoldOut, -Validation)

random_split_data <- read_csv("./GNN4siRNA-predict/accuracy.random.csv") %>%
  gather(key=Measurement, value=Value, -HoldOut, -Validation)

p <- ggplot(gene_split_data, aes(x = Measurement, y = Value)) +
  theme_bw() +
  geom_jitter(col="gray") +
  geom_boxplot(outlier.shape = NA, fill=NA) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "", y = "", title = "Random Split by Gene")
p
ggsave("./GNN4siRNA-predict/accuracy.gene.svg")

p <- ggplot(random_split_data, aes(x = Measurement, y = Value)) +
  theme_bw() +
  geom_jitter(col="gray") +
  geom_boxplot(outlier.shape = NA, fill=NA) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "", y = "", title = "Random Split")
p
ggsave("./GNN4siRNA-predict/accuracy.random.svg")