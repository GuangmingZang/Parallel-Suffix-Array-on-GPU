----------------------------
Parallel Suffix Array on GPU
----------------------------

###Dynamic Parallel Skew Algorithm for Suffix Array on GPU


###Introduction

In bioinformatics applications, suffix arrays are widely used to DNA sequence alignments in the initial exact match phase of heuristic algorithms. With the exponential growth and availability of data, using many-core accelerators, like GPUs, to optimize existing algorithms is very common. 

###Our Work
We present a new implementation of suffix array on GPU. As a result, suffix array construction on GPU achieves around 10x speedup on standard large data sets, which contain more than 100 million characters. The idea is simple, fast and scalable that can be easily scale to multi-core processors and even heterogeneous architectures.

###COPYRIGHT
Gang Liao, liao.gang@kaust.edu.sa


`Accepted By 15th IEEE/ACM International Symposium CCGrid 2015`
