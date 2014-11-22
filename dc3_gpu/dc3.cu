//construct suffix array in linear time
//Author: Gang Liao

#include "head.h"



void read_data(char *filename, char *buffer, int num){
	FILE *fh;
	fh = fopen(filename, "r");
	fread(buffer, 1, num, fh);
	buffer[num] = '\0';
	fclose(fh);
}





int main(int argc, char* argv[])
{
	//freopen("data","r",stdin);
	//freopen("output.txt","w",stdout);

	clock_t start, end;						    //record time
	double runTime;


	char* filename = "genome.txt";				//load the local data set


	int n;										//input size

	char *data;									//data set pointer
	int i = 0;									//index
	//int *inp;									//transformed data pointer
	int *SA;									//Suffix Array pointer

	printf("Please input the size of dataset you want to evaluate (10 - 1000000): \t");
	scanf("%d", &n);

	data = (char *) malloc((n+1)*sizeof(char));

	read_data(filename, data, n);				//read data set from the local file


	//inp = (int *)malloc( (n+3)*sizeof(int) );	//dynamic allocate memory
	//SA  = (int *)malloc( (n+3)*sizeof(int) );
	thrust::host_vector<int> h_inp(n+3);
	thrust::host_vector<int> h_SA(n+3, 0);
	thrust::device_vector<int>d_inp;
	thrust::device_vector<int>d_SA;


	for(i=0;i<n;i++)							//Ascii 'A' -> integer 0 by 'A' - 65
	{
		//inp[i] = to_i(data[i]);
		//inp[i] = data[i];
		h_inp[i] = to_i(data[i]);
	}

	h_inp[i]=0;h_inp[i+1]=0;h_inp[i+2]=0;				//prepare for triples
	d_inp = h_inp;
	d_SA = h_SA;
   // memset(h_SA,0,sizeof(int)*(n+3));      		//initialize the SA array

    start = clock();							//record the start time

	suffixArray(d_inp, d_SA, n, MAX_ALPHA);	        //dc3/skew algorithm

	end = clock();								//record the end time
	runTime = (end - start) / (double) CLOCKS_PER_SEC ;   //run time

	h_SA = d_SA;

	for(i = 0 ; i < n ; i++)					//print sorted suffixes from data set
	{
		printf("No.%d Index.", i);
		print_suffix(data, h_SA[i]);
	}

	printf("CPU linear construct Suffix Array\nNUM: %d \t Time: %f Sec\n", n, runTime);


	free(data);									//free allocated memory
	return 0;
}


struct mapping {
				__host__ __device__ int operator()(const int& x) const
				{
					return x + x/2 + 1;
				}
		};

void suffixArray(thrust::device_vector<int>& s, thrust::device_vector<int>& SA, int n, int K) {

	int n0=(n+2)/3, n1= (n+1)/3, n2=n/3, n02=n0+n2;

	thrust::device_vector<int>d_s12(n02+3);
	thrust::device_vector<int>d_SA12(n02+3, 0);
	thrust::device_vector<int>d_s0(n0, 0);
	thrust::device_vector<int>d_SA0(n0, 0);
	thrust::device_vector<int>d_scan(n02+3);

	// S12 initialization:
	thrust::sequence(d_s12.begin(), d_s12.end());
	thrust::transform(d_s12.begin(), d_s12.end(), d_s12.begin(), mapping());


	dim3 numThreads(1024,1,1);
	dim3 numBlocks((n02-1)/1024 + 1);

	int *pd_s12 = thrust::raw_pointer_cast( &d_s12[0] );
	int *pd_SA12 = thrust::raw_pointer_cast( &d_SA12[0] );
	int *pd_s = thrust::raw_pointer_cast( &s[0] );
	int *pd_s0 = thrust::raw_pointer_cast( &d_s0[0] );
	int *pd_SA0 = thrust::raw_pointer_cast( &d_SA0[0] );
	int *pd_SA = thrust::raw_pointer_cast( &SA[0] );
	//radix sort - using SA12 to store keys
	keybits<<<numBlocks, numThreads>>>(pd_SA12, pd_s12, pd_s , n02, 2);
	thrust::sort_by_key(d_s12.begin(), d_s12.end(), d_SA12.begin());

	keybits<<<numBlocks, numThreads>>>(pd_SA12, pd_s12, pd_s , n02, 1);
	thrust::sort_by_key(d_s12.begin(), d_s12.end(), d_SA12.begin());

	keybits<<<numBlocks, numThreads>>>(pd_SA12, pd_s12, pd_s , n02, 0);
	thrust::sort_by_key(d_s12.begin(), d_s12.end(), d_SA12.begin());


	d_SA12 = d_s12;


	// stably sort the mod 0 suffixes from SA12 by their first character
	// find lexicographic names of triples
	int *pd_scan = thrust::raw_pointer_cast( &d_scan[0] );
	InitScan<<<numBlocks, numThreads>>>(pd_s, pd_SA12, pd_scan, n02);
	thrust::exclusive_scan(d_scan.begin(), d_scan.end(), d_scan.begin());
	Set_suffix_rank<<<numBlocks, numThreads>>>(pd_s12, pd_SA12, pd_scan, n02, n0);

	int max_rank = d_scan[n0-1] + 1;
	//int max_rank = set_suffix_rank(s,s12,SA12,n02,n0);


	// if max_rank is less than the size of s12, we have a repeat. repeat dc3.
	// else generate the suffix array of s12 directly

	if(max_rank < n02)
	{
		suffixArray(d_s12,d_SA12,n02,max_rank);
		Store_unique_ranks<<<numBlocks, numThreads>>>(pd_s12, pd_SA12, n02);
	}else{
		Compute_SA_From_UniqueRank<<<numBlocks, numThreads>>>(pd_s12, pd_SA12, n02);
	}




	InitScan2<<<numBlocks, numThreads>>>(pd_SA12, pd_scan, n0, n02);
	thrust::exclusive_scan(d_scan.begin(), d_scan.end(), d_scan.begin());
	Set_S0<<<numBlocks, numThreads>>>(pd_s0, pd_SA12, pd_scan, n0, n02);

	dim3 numBlocks3((n0-1)/1024 + 1);
	keybits<<<numBlocks3, numThreads>>>(pd_SA0, pd_s0, pd_s, n0, 0);
	thrust::sort_by_key(d_s0.begin(), d_s0.end(), d_SA0.begin());
	d_SA0 = d_s0;


	// merge sorted SA0 suffixes and sorted SA12 suffixes
	dim3 numBlocks2((n-1)/1024 + 1);
	merge_suffixes<<<numBlocks, numThreads>>>(pd_SA0, pd_SA12, pd_SA, pd_s, pd_s12, n0, n02, n);


	cudaFree(pd_s12);
	cudaFree(pd_SA12);
	cudaFree(pd_s0);
	cudaFree(pd_SA0);
	cudaFree(pd_scan);
	//printf("End of suffix array !!\n");
}






//	merge_suffixes(SA0, SA12, SA, s, s12, n0, n1, n02, n);
/*
void merge_suffixes(int * SA0, int * SA12, int * SA, int * s, int * s12, int n0, int n1, int n02, int n){
	int p,t,k,i,j;
	for (p=0,  t=n0-n1,  k=0;  k < n;  k++) {
		int i = GetI(); // pos of current offset 12 suffix
		int j = SA0[p]; // pos of current offset 0  suffix
		if (SA12[t] < n0 ? leq(s[i], s12[SA12[t] + n0], s[j], s12[j/3]) : leq2(s[i],s[i+1],s12[SA12[t]-n0+1],s[j],s[j+1],s12[j/3+n0]))
		{ // suffix from SA12 is smaller
			SA[k] = i;  t++;
			if (t == n02) { // done --- only SA0 suffixes left
				for (k++;  p < n0;  p++, k++) SA[k] = SA0[p];
			}
		} else {
			SA[k] = j;  p++;
			if (p == n0)  { // done --- only SA12 suffixes left
				for (k++;  t < n02;  t++, k++) SA[k] = GetI();
			}
		}
	}
}*/
