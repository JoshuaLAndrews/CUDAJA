//nvcc SimpleJuliaSetGPUsimplified.cu.cu -o SimpleJuliaSetGPUsimplified -lglut -lGL -lm
// This is a simple Julia set which is repeated iterations of 
// Znew = Zold + C whre Z and Care imaginary numbers.
// After so many tries if Zinitial escapes color it black if it stays around color it red.

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


#define A  0.276387625  //real
#define B  -0.4953556789211   //imaginary
#define N 1024 // ** Don't change this

unsigned int window_width = 1024;
unsigned int window_height = 1024;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);
float *pixels; 
float *pixelsg;

dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

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
	cudaMalloc(&pixelsg,N*3*N*sizeof(float));
	pixels = (float *)malloc(N*3*N*sizeof(float));
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
		temp = x; // We will be changing the x but weneed its old value to hind y.	
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

__global__ void initialization(float *g,float stepSizeX ,float stepSizeY ,float yMin,float xMin) 
{ 
	float x, y;
	int id=(blockIdx.x*blockDim.x + threadIdx.x)*3;

	y = yMin + blockIdx.x*stepSizeY;
	x = xMin + threadIdx.x*stepSizeX;
	
	g[id] = color(x,y);	//Red on or off returned from color
	g[id+1] = 0.0; 	//Green off
	g[id+2] = 0.0;	//Blue off	
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

void display(void)
{
	initialization<<<GridSize,BlockSize>>>(pixelsg,stepSizeX ,stepSizeY ,yMin,xMin);
	cudaMemcpyAsync(pixels, pixelsg, N*N*3*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck(__FILE__, __LINE__);
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}


int main(int argc, char** argv)
{ 
	SetUpCudaDevices();
	AllocateMemory();
	cudaMemcpyAsync(pixelsg, pixels, N*N*3*sizeof(float), cudaMemcpyHostToDevice);
	errorCheck(__FILE__, __LINE__);
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	glutMainLoop();
}
