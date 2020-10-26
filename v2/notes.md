10-22-2020

The `v2` directory is created to update the manuscript on arxiv. The main changes involve:
1. Implement a new test (based on pearson correlation) that has test statistic equals 0 when the null is true
2. Update the code base so it depends on the package `graspologic`
3. Simplify so there needs no mention of dcorr or mgc

The ultimate goal is to merge this work into `graspologic` under the `inference`module, as an approach to perform independence test between a pair of graphs. But since the priority now is to update the manuscript, I didn't directly try to integrate directly into `graspologic`, so it's easier to structure code here. But to keep the code clean, I should do the following:
1. Use existing `graspologic` functions as much as possible, only implement new functionalities when necessary.
2. Include unit tests for each function to make sure they are correct, in a directory named `tests`
3. I can use notebooks to visualize and play with functions, but to keep things clean, move code into `.py` script as soon as I know they are working, and don't keep the draft notebooks. Don't commit any notebook file, unless it is a demo.

Below is a rough working plan
1. Implement correlated sbm with different marginals, based on the correlated sbm implementation in `graspologic`. Check correlation based on unit tests. Test should be ER and SBM, both with same & different marginals, SBM with different block sizes
2. Implement the new test statistic: `gcorr`
3. Implement test statistic simulation (figure 2): test statistic vs. true correlation. same 4 tests as below. For figure, just show ER & SBM w/ different marginals.
4. Implement power simulation (figure 3): exact pearson, pearson with vertex permutation, pearson with block permutation, gcorr with block permutation

After I have a working code base based on the new test, I should think about how to address some of the comments in the nips reviews
1. The assumption that the 2 graphs share the same community structure might be too strong. In that case, it makes sense as long as we account for the block structure, we can ignore the graph structure and do something like pearson e.g. gcorr. Although I feel like it would be a different problem (e.g. unconditional independence) if we are not making this assumption, since in that case, the fact that the 2 graphs share some community structure is indicative of dependence, whereas in our case we are finding correlation **given** the community structure, which is only going to be edge correlations. Maybe in that case we are testing if the joint are independent: (A, ZA) vs. (B, ZB)
2. If we are making this assumption, we need a more principled way to account for the error in the MASE community assignment. That will be crucial in making sure the method works in practice (and how much samples it requires to work). The new thoerem should serve this purpose: if the null is true, when we estimate $\hat{Z}$, as n goes to infinity, the test statistic should go to zero. Maybe use a simulation or real data example to show this

Some ideas for real applications
1. Mouse connectomes. Have ground truths of which mouse species are more related than others. Does `gcorr` reflect "relateness"? Similarly, if we have some connectomes measured over time, closer time points should be more correlated than others? 
2. Some example where we know for sure the 2 graphs should **not** be correlated. Does `gcorr` rejects the null? This also allows us to check the new theorem.
3. Is there a scientific discovery setting? For some variable x, we measure a bunch of dependent variable y that's potentially related, and we can actually see this method at work in rejecting null hypothesis? 
4. The drosophila learning experiment mentioned in rebuttal: flies reared apart vs. together, how does the correlation between the two groups compared after the training vs. before the training? If it didn't change much, then the rearing conditional did **not** change how the connectomes changes, if it did, then the rearing condition is important. But here we actually need to compare to a base line correlation i.e. before training.