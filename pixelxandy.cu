// large Vector addition on the GPU with a fixed number of blocks.
// nvcc pixelxandy.cu -o pixelxandy -lglut -lGL -lm

#include <sys/time.h>
#include <stdio.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//Length of vectors to be added.
#define N 1024 // ** Don't change this
#define A  0.276387625  //real
#define B  -0.4953556789211   //imaginary
// ** Don't change this

//Globals
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float *pixels; 
float *pixelsg;
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

unsigned int window_width = 1024;
unsigned int window_height = 1024;


float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);


//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	BlockSize.x = 1024; 
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1024; 
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*N*sizeof(float));
	cudaMalloc(&C_GPU,N*N*sizeof(float));
	cudaMalloc(&pixelsg,N*3*N*sizeof(float));

	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*N*sizeof(float));
	C_CPU = (float*)malloc(N*N*sizeof(float));
	pixels = (float *)malloc(N*3*N*sizeof(float));
}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;
	float x;
	x = xMin;
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = x;
		x += stepSizeX;
	}
}



//Cleaning up memory after we are finished.
void CleanUp()
{
	free(A_CPU); free(B_CPU); free(C_CPU); free(pixels);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU); cudaFree(pixelsg);
}




__device__ float color (float x, float y) 
{
	float mag,maxMag,temp;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;
	
	while (mag < maxMag && count < maxCount) 
	{
		// Zn = Zo*Zo + C
		// or xn + yni = (xo + yoi)*(xo + yoi) + A + Bi
		// xn = xo*xo - yo*yo + A (real Part) and yn = 2*xo*yo + B (imagenary part)
		temp = x; // We will be changing the x but we need its old value to hind y.	
		x = x*x - y*y + A;
		y = (2.0 * temp * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}




//This is the kernel. It is the function that will run on the GPU.
//It adds vectors A and B then stores result in vector C
__global__ void AdditionGPU(float *a, float *b, float *c,float *d, int n)
{
	int id = blockIdx.x *blockDim.x + threadIdx.x;
	int id2 = blockIdx.x;
	float x;
	float y;
	
	while(id < N*N)
	{
		b[id] = a[id2];
		c[id] = a[threadIdx.x];
		id += blockDim.x * gridDim.x;
		if(id2>N)
		{
		id2+=1;
		}
		
	}
	id=0;
	id = blockIdx.x *blockDim.x + threadIdx.x;
	int k;
	k=3*blockIdx.x *blockDim.x + 3*threadIdx.x;
	while(id < N*N)
	{
	x = c[id];
	y = b[id];
	d[k] = color(x,y);	//Red on or off returned from color	
	d[k+1] = 0.0;	//Green off
	d[k+2] = 0.0;	//Blue off
	id += blockDim.x * gridDim.x;
	}

}

void display(void) 
{ 
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}

void errorCheck(const char *file, int line)
{
cudaError_t error;
error = cudaGetLastError();

if(error != cudaSuccess)
{
printf("\n CUDA message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
exit(0);
}
}

int main(int argc, char** argv)
{
	//int i=0;
	timeval start, end;
	//cudaError_t deviceError;
	
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	//Starting the timer
	gettimeofday(&start, NULL);

	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	errorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*N*sizeof(float), cudaMemcpyHostToDevice);
	errorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(pixelsg, pixels, N*N*3*sizeof(float), cudaMemcpyHostToDevice);
	errorCheck(__FILE__, __LINE__);
	
	
	//Calling the Kernel (GPU) function.	
	AdditionGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU,pixelsg,N);
	errorCheck(__FILE__, __LINE__);
	
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_CPU, B_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(pixels, pixelsg, N*N*3*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck(__FILE__, __LINE__);
	
	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
//	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
/*	
	// Displaying the vector. You will want to comment this out when the vector gets big.
	// This is just to make sure everything is running correctly.	
	for(i = 0; i < N*N; i++)		
	{		
		printf("B[%d] = %.15f  C[%d] = %.15f\n P[%d] = %.15f\n", i, B_CPU[i], i, C_CPU[i], i, pixels[i*3]);
	}
*?


	//Displaying the last value of the addition for a check when all vector display has been commented out.
	printf("Last Values are A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", N-1, A_CPU[N-1], N*N-1, B_CPU[N*N-1], N*N-1, C_CPU[N*N-1]);
	
	//Displaying the time 
	printf("Time in milliseconds= %.15f\n", (time/1000.0));	

	
/*	
	bool success = true;
for (int i=0; i<N; i++) 
	{
	if ((A_CPU[i] + B_CPU[i]) != C_CPU[i]) 
		{
		printf("Error: i=%d %f + %f != %f\n",i, A_CPU[i], B_CPU[i], C_CPU[i] );
		success = false;
		}
	}
if (success) printf( "We did it!\n" );
if (!success) printf( "Sh*t Broken Go Fix it!\n" );
*/


	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	glutMainLoop();
	//You're done so cleanup your mess.
	CleanUp();	
	
	return(0);
}
