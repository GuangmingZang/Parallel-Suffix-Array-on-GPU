----------------------------
Parallel Suffix Array on GPU
----------------------------

###Dynamic Parallel Skew Algorithm for Suffix Array on GPU


>Name: Gang Liao
>
>Email: liao.gang@kaust.edu.sa
>
>Homepage: [gangliao.me](http://gangliao.me)
>
>King Abdullah University of Science and Technology (KAUST)
>
>Computer, Electrical and Mathematical Sciences and Engineering (CEMSE) Division

##Content

1. [Abstraction](#abstraction)
1. [Introduction](#introduction)
    1. [Parallel Computing](#parallel)
    1. [CUDA GPU Programming](#GPU)
1. [The Problem](#problem)
    1. [Suffix Tree and Suffix Array](#suffix)
    1. [DC3 Algorithm](#dc3)
1. [My idea](#idea)
    1. [Parallel DC3 algorithm](#parallel_dc3)
        1. [Parallel Radix Sort](#radix)
        1. [Parallel Merge](#merge)
    1. [Performance Optimization](#performance)
1. [Test and Result](#test)
1. [Conclusion](#conclusion) 
1. [Future Work](#future)
1. [Reference](#reference)


##<a id = “abstraction”>Abstraction</a>

###Background

Suffix array has been widely used to store and retrieve numerous datasets in
bioinformatics applications. Especially, for DNA sequence alignments in the initial 
exact match phase of heuristic algorithms. In the era of post petascale computing, how
to parallel those algorithms more efficiency is one of the most important things both academics and 
industries need to figure out. 

###My Work

The paper proposes a novel way to optimize dc3 algorithm for suffix array construction  using dynamic parallel model
and typical parallel patterns - prefix sum and reduction. The comparative performance 
evaluation with the state-of-the-art implementation is then carried out.

###Conclusion

The study shows that massively parallel dc3 algorithm under heterogeneous architecture
is an efficient approach to high-performance bioinformatics applications and web-search engines.

##<a id = “introduction”>Introdution</a>

###<a id= “parallel”>Parallel Computing</a>

From LLNL’s tutorials, parallel computing is the simultaneous use of multiple compute resources
to solve a computational problem:

1. A problem is broken into discrete parts that can be solved concurrently;
1. Each part is further broken down to a series of instructions;
1. Instructions from each part execute simultaneously on different processors;
1. An overall control/coordination mechanism is employed.

Computational problem should be able to:

1. Be broken apart into discrete pieces of work that can be solved simultaneously;
2. Execute multiple program instructions at any moment in time;
3. Be solved in less time with multiple compute resources than with a single compute resource.

The compute resources are typically:

1. A single computer with multiple processors/cores;
2. An arbitrary number of such computers connected by a network.

###<a id = “GPU”>GPU CUDA Programming</a>

In recent years, modern multi-core and many-core architectures have brought about a 
revolution in high performance computing. Because it is now possible to incorporate 
more and more processor cores into a single chip, the era of the many-core processor 
is not far away. The emergence of many-core architectures, such as Compute Unified 
Device Architecture (CUDA)-enabled GPUs and other accelerator technologies 
including Field-Programmable Gate Arrays (FPGAs) and the Cell/BE architecture, 
makes it possible to reduce significantly the running time of many biological 
algorithms. Moreover, these architectures have proved to be very versatile, have 
enormous computing power, and are low cost. Until now, more than 100 million 
computers with CUDA capable GPUs have been shipped to end-users. Currently, 
many researchers are interested in employing the powerful ability of floating point 
computation to create new biological algorithms because of the low entry threshold.

