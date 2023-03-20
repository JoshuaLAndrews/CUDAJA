// nvcc histogramAtomics.cu -o temp

#include <sys/time.h>
#include <stdio.h>
#include "./MyCuda.h"

//Max int value is 2,147,483,647       2147483
//Chat said that the length of the sequence of random number that srand generates in 2^32
//That is 4,294,967,296 this is bigger than the largest int but the max for an unsigned int.
#define NUMBER_OF_RANDOM_NUMBERS 2147483 
#define NUMBER_OF_BINS 10
#define MAX_RANDOM_NUMBER 100.0f
#define BLOCK_SIZE 256
#define MULTIPROCESSOR_MULTIPLIER 2

//Function prototypes
void SetUpCudaDevices();
void AllocateMemory();
void Innitialize();
void CleanUp();
void fillHistogramCPU();
__global__ void fillHistogramGPU(float *, int *);

//Globals
float *RandomNumbersGPU;
int *HistogramGPU;
float *RandomNumbersCPU;
int *HistogramCPU;
int *HistogramFromGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
cudaEvent_t StartEvent, StopEvent;

//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = BLOCK_SIZE;
	if(prop.maxThreadsDim[0] < BlockSize.x)
	{
		printf("\n You are trying to create more threads (%d) than your GPU can suppport on a block (%d).\n Good Bye\n", BlockSize.x, prop.maxThreadsDim[0]);
		exit(0);
	}
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//*********** Setting the number of blocks to a multiple of the number of streaming multiprocessors
	// ???? This is new. Nothing to do here just see that the number of blocks is a multiple of the multiprocessors. Two seems to work well.
	GridSize.x = MULTIPROCESSOR_MULTIPLIER*prop.multiProcessorCount;
	//***********
	if(prop.maxGridSize[0] < GridSize.x)
	{
		printf("\n You are trying to create more blocks (%d) than your GPU can suppport (%d).\n Good Bye\n", GridSize.x, prop.maxGridSize[0]);
		exit(0);
	}
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&RandomNumbersGPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&HistogramGPU, NUMBER_OF_BINS*sizeof(int));
	myCudaErrorCheck(__FILE__, __LINE__);

	//Allocate Host (CPU) Memory
	RandomNumbersCPU = (float*)malloc(NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	HistogramCPU = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	HistogramFromGPU = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	
	//Setting the the histograms to zero.
	cudaMemset(HistogramGPU, 0, NUMBER_OF_BINS*sizeof(int));
	myCudaErrorCheck(__FILE__, __LINE__);
	memset(HistogramCPU, 0, NUMBER_OF_BINS*sizeof(int));
}

//Loading random numbers.
void Innitialize()
{
	time_t t;
	srand((unsigned) time(&t));
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++)
	{		
		RandomNumbersCPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	printf("\n Cleanup Start\n");
	cudaFree(RandomNumbersGPU); 
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaFree(HistogramGPU);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StartEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	free(RandomNumbersCPU); 
	free(HistogramCPU);
	free(HistogramFromGPU);
	printf("\n Cleanup Finished\n");
}

void fillHistogramCPU()
{
	float breakPoint;
	int k, done;
	float stepSize = MAX_RANDOM_NUMBER/(float)NUMBER_OF_BINS;
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++)
	{
		breakPoint = stepSize;
		k = 0;
		done =0;
		while(done == 0)
		{
			if(RandomNumbersCPU[i] < breakPoint)
			{
				HistogramCPU[k]++; 
				done = 1;
			}
			if(NUMBER_OF_BINS < k)
			{
				printf("\n k is too big (on CPU) k = %d\n", k);
				exit(0);
			}
			k++;
			breakPoint += stepSize;
		}
	}
}

//This is the kernel. It is the function that will run on the GPU.
__global__ void fillHistogramGPU(float *randomNumbers, int *hist)
{
	float breakPoint;
	int i, k, done;
	float stepSize = MAX_RANDOM_NUMBER/(float)NUMBER_OF_BINS;
	int jump = blockDim.x*gridDim.x;
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	
	//****************************************
	// Follow the model set in the CPU code and do a histogram with atomic adds to shared memory 
	// then atomic adds to glogal memory. Because we set the number of blocks to be a multiple of the 
	// the number of multiprocessors there may not be enough to threads to finish the job. Soo you will
	// have to "jump" through the random numbers in blockDim.x*gridDim.x strides.
	//****************************************
	__shared__ float temp[BLOCK_SIZE];
	temp[threadIdx.x] = 0;
	__syncthreads();
	
	for(i = id; i < NUMBER_OF_RANDOM_NUMBERS; i+=jump)
	{
		breakPoint = stepSize;
		k = 0;
		done =0;
		while(done == 0)
		{
			if(randomNumbers[i] < breakPoint)
			{
				atomicAdd(&temp[k],1); 
				done = 1;
			}
			if(NUMBER_OF_BINS < k)
			{
				printf("\n k is too big (on GPU) k = %d\n", k);
			}
			k++;
			breakPoint += stepSize;
		}
	}
	
	__syncthreads();
	atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
}

int main()
{
	float time, timeEvent;
	timeval start, end;
	
	cudaEventCreate(&StartEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	long int test = NUMBER_OF_RANDOM_NUMBERS;
	if(2147483647 < test)
	{
		printf("\nThe length of your vector is longer than the largest integer value allowed of 2147483647.\n");
		printf("You should check your code.\n Good Bye\n");
		exit(0);
	}
	
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using and padding with zero vector will be a factor of block size.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	gettimeofday(&start, NULL);
	fillHistogramCPU();
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\n Time on CPU = %.15f milliseconds\n", (time/1000.0));
	
	//Doing two timers 
	gettimeofday(&start, NULL);
	cudaEventRecord(StartEvent, 0);
	myCudaErrorCheck(__FILE__, __LINE__);
	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(RandomNumbersGPU, RandomNumbersCPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float), cudaMemcpyHostToDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	fillHistogramGPU<<<GridSize,BlockSize>>>(RandomNumbersGPU, HistogramGPU);
	myCudaErrorCheck(__FILE__, __LINE__);
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(HistogramFromGPU, HistogramGPU, NUMBER_OF_BINS*sizeof(int), cudaMemcpyDeviceToHost);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaEventRecord(StopEvent, 0);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	myCudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using events= %3.1f milliseconds", timeEvent);
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf("\n Time on GPU using straight up time= %.15f milliseconds\n", (time/1000.0));
	
	//Check
	for(int i = 0; i < NUMBER_OF_BINS; i++)
	{
		if(10.0 < (abs(HistogramFromGPU[i] - HistogramCPU[i])))
		{
			printf("\nError = histogram element %d is off by %d histoGPU = %d histCPU = %d.\n", i, abs(HistogramFromGPU[i] - HistogramCPU[i]), HistogramFromGPU[i], HistogramCPU[i]);
		}
	}
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	return(0);
}
