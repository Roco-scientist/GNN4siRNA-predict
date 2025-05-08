library(tidyverse)

gibbs_data <- read_csv("./GNN4siRNA-predict/LaRosa_Gibbs.csv")

gibbs_plot <- gather(gibbs_data[, c("Sense_siRNA_Gibbs_Free_Energy", "Antisense_siRNA_Gibbs_Free_Energy", "LaRosa_Gibbs_Free_Energy")], key = "Strand", value = "Calculated_Gibbs", -LaRosa_Gibbs_Free_Energy)
gibbs_plot$Strand <- sapply(gibbs_plot$Strand, function(strand) {
  if (strand == "Sense_siRNA_Gibbs_Free_Energy") {
    "siRNA Sense Strand"
  }else{
    "siRNA Antisense Strand"
  }
})

p <- ggplot(gibbs_plot, aes(x = Calculated_Gibbs, y = LaRosa_Gibbs_Free_Energy, col = Strand)) +
  theme_bw() +
  geom_point(size = 4) +
  coord_cartesian(ylim = c(-40, 0), xlim = c(-40, 0)) +
  scale_color_brewer(palette = "Set1", name = "") +
  labs(x = "Calculated Gibbs Free\nEnergy (kcal/mol)", y = "LaRosa Gibbs Free\nEnergy Values (kcal/mol)") +
  theme(text = element_text(size = 30, face="bold", colour = "black"))
p
ggsave("./GNN4siRNA-predict/Gibbs_comparison.svg", dpi = 1000, width = 16, height = 12)


gibbs_data$siRNA_Antisense <- sapply(gibbs_data$Antisense_siRNA_matching_mRNA_sequence, nchar)
gibbs_data$siRNA_Sense <- sapply(gibbs_data$Sense_siRNA_matching_mRNA_sequence, nchar)


length_plot <- gather(gibbs_data[, c("siRNA_Antisense", "siRNA_Sense")], key = "Strand", value = "Nucleotides")

p <- ggplot(length_plot, aes(x = Strand, y = Nucleotides, col = Strand)) +
  theme_bw() +
  geom_boxplot(outliers = F) +
  geom_jitter(size = 4) +
  scale_color_brewer(palette = "Set1", guide = "none") + 
  labs(x = "", y = "RNAup Identified Binding\nmRNA length (Nucloetides)") +
  theme(text = element_text(size = 30, face="bold", colour = "black"), axis.text.x = element_text(hjust = 1, angle = 45))
p
ggsave("./GNN4siRNA-predict/Length_comparison.svg", dpi = 1000, width = 16, height = 12)
