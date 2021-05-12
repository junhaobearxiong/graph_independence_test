library(MASS)

for (i in c(1:18)) {
  dirname <- '~/Documents/Projects/graph_independence_test/v2/data_utils/data/enron/graphs/'
  filename <- paste(i, '.txt', sep='')
  write.matrix(Adj_list[[i]], file=paste(dirname, filename, sep=''))
}