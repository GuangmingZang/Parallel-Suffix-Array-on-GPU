----------------------------
Parallel Suffix Array on GPU
----------------------------

###Dynamic Parallel Skew Algorithm for Suffix Array on GPU

###AMCS 312 Final Project###

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
1. [Introduction] (#introduction)
2. [Genome Matching and Alignment](#genome)
3. [Parallel Computing](#parallel)
4. [CUDA GPU Programming](#GPU)
5. [Suffix Tree and Suffix Array](#suffix)
6. [DC3 Algorithm](#dc3)
7. [Parallel DC3 algorithm](#parallel_dc3)
    1. [Parallel Radix Sort](#radix)
    2. [Parallel Merge](#merge)
8. [Performance Optimization](#performance)
9. [Test and Result](#test)
10. [Conclusion](#conclusion) 
11. [Future Work](#future)
12. [Reference](#reference)


##<a id =”Abstraction”>Abstraction</a>
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

<a id =”introduction”>Introdution</a>



