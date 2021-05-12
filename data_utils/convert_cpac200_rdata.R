library(MASS)

for (i in c(1:300)) {
  dirname <- '~/Documents/Projects/graph_independence_test/v2/data_utils/data/cpac200/'
  filename <- paste(subid_list[[i]], '_', scanid_list[i], '.txt', sep='')
  write.matrix(Adj_list[[i]], file=paste(dirname, filename, sep=''))
}