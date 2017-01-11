#install.packages("ggplot2")
library(ggplot2)

pixel_count <- 28 * 28

train_labels <- read.csv("data/train.csv", header = TRUE)[,c("label")]

num_labels <- sapply(c(0:9), (function(i) sum(train_labels == i)))
cbind(c(0:9), num_labels)

train_labels_freq <- table(train_labels)
barplot(train_labels_freq)
