//nvcc RayTracerAnimation.cu -o Regular -lglut -lGL -lm

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "./MyCuda.h"

#define WINDOWWIDTH 1024
#define WINDOWHEIGHT 1024
#define XMIN -1.0f
#define XMAX 1.0f
#define YMIN -1.0f
#define YMAX 1.0f
#define ZMIN -1.0f
#define ZMAX 1.0f

#define MAXRADIUS ((XMAX- XMIN)*0.1)
#define MINRADIUS (MAXRADIUS*0.5)
#define NUMSPHERES 20000
#define STEPS 100

struct sphereStruct 
{
	float r,b,g; // Sphere color
	float radius;
	float x,y,z; // Sphere center
};

static int Window;
unsigned int WindowWidth = WINDOWWIDTH;
unsigned int WindowHeight = WINDOWHEIGHT;

dim3 BlockSize, GridSize;
float *PixelsCPU, *PixelsGPU; 
sphereStruct *SpheresCPU, *SpheresCPUNext, *SpheresGPU;

// prototyping functions
void Display();
void idle();
void KeyPressed(unsigned char , int , int );
__device__ float hit(float , float , float *, float , float , float , float );
__global__ void makeSphersBitMap(float *, sphereStruct *);
void makeRandomSpheres(sphereStruct *sphere);
void animateBitMap();
void makeBitMap();
void paintScreen();
void setup();

void idle()
{
	animateBitMap();
}

void display()
{
	//animateBitMap();
	paintScreen();	
}

void KeyPressed(unsigned char key, int x, int y)
{	
	if(key == 'q')
	{
		glutDestroyWindow(Window);
		printf("\n Good Bye\n");
		exit(0);
	}
}

__device__ float hit(float pixelx, float pixely, float *dimingValue, sphereStruct sphere)
{
	float dx = pixelx - sphere.x;  //Distance from ray to sphere center in x direction
	float dy = pixely - sphere.y;  //Distance from ray to sphere center in y direction
	float r2 = sphere.radius*sphere.radius;
	if(dx*dx + dy*dy < r2) // if the ray hits the sphere, then we need to find distance
	{
		float dz = sqrtf(r2 - dx*dx - dy*dy); // Distance from ray to edge of sphere?
		*dimingValue = dz/sphere.radius; // n is value between 0 and 1 used for darkening points near edge.
		return dz + sphere.z; //  Return the distance to be scaled by
	}
	return (ZMIN- 1.0); //If the ray doesn't hit anything return a number behind the box.
}

__global__ void makeSphersBitMap(float *pixels, sphereStruct *sphereInfo)
{
	float stepSizeX = (XMAX - XMIN)/((float)WINDOWWIDTH - 1);
	float stepSizeY = (YMAX - YMIN)/((float)WINDOWHEIGHT - 1);
	
	// Asigning each thread a pixel
	float pixelx = XMIN + threadIdx.x*stepSizeX;
	float pixely = YMIN + blockIdx.x*stepSizeY;
	
	// Finding this pixels location in memory
	int id = 3*(threadIdx.x + blockIdx.x*blockDim.x);
	
	//initialize rgb values for each pixel to zero (black)
	float pixelr = 0.0f;
	float pixelg = 0.0f;
	float pixelb = 0.0f;
	float hitValue;
	float dimingValue;
	float maxHit = ZMIN -1.0f; // Initializing it to be out of the back of the box.
	for(int i = 0; i < NUMSPHERES; i++)
	{
		//hitValue = hit(pixelx, pixely, &dimingValue, sphereInfo[i].x, sphereInfo[i].y, sphereInfo[i].z, sphereInfo[i].radius); 
		hitValue = hit(pixelx, pixely, &dimingValue, sphereInfo[i]);
		// do we hit any spheres? If so, how close are we to the center? (i.e. n)
		if(maxHit < hitValue)
		{
			// Setting the RGB value of the sphere but also diming it as it gets close to the side of the sphere.
			pixelr = sphereInfo[i].r * dimingValue; 	
			pixelg = sphereInfo[i].g * dimingValue;	
			pixelb = sphereInfo[i].b * dimingValue; 	
			maxHit = hitValue; // reset maxHit value to be the current closest sphere
		}
	}
	
	pixels[id] = pixelr;
	pixels[id+1] = pixelg;
	pixels[id+2] = pixelb;
}

void makeRandomSpheres(sphereStruct *sphere)
{	
	float rangeX = XMAX - XMIN;
	float rangeY = YMAX - YMIN;
	float rangeZ = ZMAX - ZMIN;
	
	for(int i = 0; i < NUMSPHERES; i++)
	{
		sphere[i].x = (rangeX*rand()/RAND_MAX) + XMIN;
		sphere[i].y = (rangeY*rand()/RAND_MAX) + YMIN;
		sphere[i].z = (rangeZ*rand()/RAND_MAX) + ZMIN;
		sphere[i].r = 1.0*rand()/RAND_MAX;
		sphere[i].g = 1.0*rand()/RAND_MAX;
		sphere[i].b = 1.0*rand()/RAND_MAX;
		sphere[i].radius = ((MAXRADIUS - MINRADIUS)*rand()/RAND_MAX) + MINRADIUS;
	}
}

void animateBitMap()
{
	makeRandomSpheres(SpheresCPUNext);
	float dx[NUMSPHERES], dy[NUMSPHERES], dz[NUMSPHERES], dradius[NUMSPHERES], dr[NUMSPHERES], dg[NUMSPHERES], db[NUMSPHERES];
	
	for(int i = 0; i < NUMSPHERES; i++)
	{
		dx[i] = (SpheresCPUNext[i].x - SpheresCPU[i].x)/STEPS;
		dy[i] = (SpheresCPUNext[i].y - SpheresCPU[i].y)/STEPS;
		dz[i] = (SpheresCPUNext[i].z - SpheresCPU[i].z)/STEPS;
		dradius[i] = (SpheresCPUNext[i].radius - SpheresCPU[i].radius)/STEPS;
		dr[i] = (SpheresCPUNext[i].r - SpheresCPU[i].r)/STEPS;
		dg[i] = (SpheresCPUNext[i].g - SpheresCPU[i].g)/STEPS;
		db[i] = (SpheresCPUNext[i].b - SpheresCPU[i].b)/STEPS;
	}
	
	makeBitMap();
	
	for(int j = 0; j < STEPS; j++)
	{
		for(int i = 0; i < NUMSPHERES; i++)
		{
			SpheresCPU[i].x += dx[i];
			SpheresCPU[i].y += dy[i];
			SpheresCPU[i].z += dz[i];
			SpheresCPU[i].radius += dradius[i];
			SpheresCPU[i].r += dr[i];
			SpheresCPU[i].g += dg[i];
			SpheresCPU[i].b += db[i];
		}
		makeBitMap();
	}
}	

void makeBitMap()
{
	cudaMemcpy(SpheresGPU, SpheresCPU, NUMSPHERES*sizeof(sphereStruct), cudaMemcpyHostToDevice);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	makeSphersBitMap<<<GridSize, BlockSize>>>(PixelsGPU, SpheresGPU);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	cudaMemcpyAsync(PixelsCPU, PixelsGPU, WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float), cudaMemcpyDeviceToHost);
	myCudaErrorCheck(__FILE__, __LINE__);
	
	paintScreen();
}

void paintScreen()
{
	//Putting pixels on the screen.
	glDrawPixels(WINDOWWIDTH, WINDOWHEIGHT, GL_RGB, GL_FLOAT, PixelsCPU); 
	glFlush();
}

void setup()
{
	//We need the 3 because each pixel has a red, green, and blue value.
	PixelsCPU = (float *)malloc(WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float));
	cudaMalloc(&PixelsGPU,WINDOWWIDTH*WINDOWHEIGHT*3*sizeof(float));
	myCudaErrorCheck(__FILE__, __LINE__);
	
	SpheresCPU = (sphereStruct*)malloc(NUMSPHERES*sizeof(sphereStruct));
	SpheresCPUNext = (sphereStruct*)malloc(NUMSPHERES*sizeof(sphereStruct));
	cudaMalloc(&SpheresGPU, NUMSPHERES*sizeof(sphereStruct));
	myCudaErrorCheck(__FILE__, __LINE__);
	
	//Threads in a block
	if(WINDOWWIDTH > 1024)
	{
	 	printf("The window width is too large to run with this program\n");
	 	printf("The window width must be less than 1024.\n");
	 	printf("Good Bye and have a nice day!\n");
	 	exit(0);
	}
	BlockSize.x = WINDOWWIDTH;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//Blocks in a grid
	GridSize.x = WINDOWHEIGHT;
	GridSize.y = 1;
	GridSize.z = 1;
	
	// Seading the random number generater.
	time_t t;
	srand((unsigned) time(&t));
}

int main(int argc, char** argv)
{ 
	setup();
	//makeRandomSpheres();
	makeRandomSpheres(SpheresCPU);
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WINDOWWIDTH, WINDOWHEIGHT);
	Window = glutCreateWindow("Random Spheres");
	glutKeyboardFunc(KeyPressed);
   	glutDisplayFunc(display);
	//glutReshapeFunc(reshape);
	//glutMouseFunc(mymouse);
	glutIdleFunc(idle);
   	glutMainLoop();
}

