10-22-2020

The `v2` directory is created to update the manuscript on arxiv. The main changes involve:
1. Implement a new test (based on pearson correlation) that has test statistic equals 0 when the null is true
2. Update the code base so it depends on the package `graspologic`
3. Simplify so there needs no mention of dcorr or mgc

The ultimate goal is to merge this work into `graspologic` under the `inference`module, as an approach to perform independence test between a pair of graphs. But since the priority now is to update the manuscript, I didn't directly try to integrate directly into `graspologic`, so it's easier to structure code here. But to keep the code clean, I should do the following:
1. Use existing `graspologic` functions as much as possible, only implement new functionalities when necessary.
2. Include unit tests for each function to make sure they are correct, in a directory named `tests`
3. I can use notebooks to visualize and play with functions, but to keep things clean, move code into `.py` script as soon as I know they are working, and don't keep the draft notebooks. Don't commit any notebook file, unless it is a demo.

