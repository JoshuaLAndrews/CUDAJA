// gcc twoBodyInBox.c -o temp2 -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define XWindowSize 1000
#define YWindowSize 1000

#define STOP_TIME 10000.0
#define DT        0.0001

#define GRAVITY 10.0 

#define MASS 10.0  	
#define DIAMETER 1.0

#define SPRING_STRENGTH 5000.0
#define SPRING_REDUCTION 0.1

#define DAMP 0.0

#define DRAW 100

#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0

#define NUMBER_OF_SPHERES 10

const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

// Globals
typedef struct sphereStruct{

    float px, py, pz; //Position
    float vx, vy, vz; //Velocity  
    float fx, fy, fz; //Force
    float mass; //Mass
    float r,g,b;
}sphereStruct;

sphereStruct *SpheresCPU;

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	int yeahBuddy;
	int i=0;
	int j;
	float dx, dy, dz, seperation;
	
	
	if (i=0)
	{
		SpheresCPU[i].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		SpheresCPU[i].py= (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		SpheresCPU[i].pz= (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	}
	
	i++;
	while (i<NUMBER_OF_SPHERES)
	{
		SpheresCPU[i].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		SpheresCPU[i].py = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		SpheresCPU[i].pz = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		
		j=i;

		while(j>=0)
		{
			j--;
			yeahBuddy = 0;
			while(yeahBuddy == 0)
			{
				SpheresCPU[i].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
				SpheresCPU[i].py = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
				SpheresCPU[i].pz = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		
				dx = SpheresCPU[j].px - SpheresCPU[i].px;
				dy = SpheresCPU[j].px - SpheresCPU[i].px;
				dz = SpheresCPU[j].px - SpheresCPU[i].px;
				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				yeahBuddy = 1;
				if(seperation < DIAMETER) yeahBuddy = 0;
			}
		}
		i++;
	}
	
	i=0;
	while(i<NUMBER_OF_SPHERES)
	{
		SpheresCPU[i].vx = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		SpheresCPU[i].vy = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		SpheresCPU[i].vz = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		SpheresCPU[i].mass = 1.0;
		SpheresCPU[i].r= 1.0*rand()/RAND_MAX;
		SpheresCPU[i].g= 0.5*rand()/RAND_MAX;
		SpheresCPU[i].b= 1.0*rand()/RAND_MAX;
	i++;
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
	glColor3d(SpheresCPU[i].r,SpheresCPU[i].g,SpheresCPU[i].b);
	glPushMatrix();
	glTranslatef(SpheresCPU[i].px, SpheresCPU[i].py, SpheresCPU[i].pz);
	glutSolidSphere(radius,20,20);
	glPopMatrix();
    }
	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	
	for (int i=0; i<NUMBER_OF_SPHERES; i++)
	{
		if(SpheresCPU[i].px > halfBoxLength)
		{
			SpheresCPU[i].px = 2.0*halfBoxLength - SpheresCPU[i].px;
			SpheresCPU[i].vx = - SpheresCPU[i].vx;
		}
		else if(SpheresCPU[i].px < -halfBoxLength)
		{
			SpheresCPU[i].px = -2.0*halfBoxLength - SpheresCPU[i].px;
			SpheresCPU[i].vx = - SpheresCPU[i].vx;
		}
	
		if(SpheresCPU[i].py > halfBoxLength)
		{
			SpheresCPU[i].py = 2.0*halfBoxLength - SpheresCPU[i].py;
			SpheresCPU[i].vy = - SpheresCPU[i].vy;
		}
		else if(SpheresCPU[i].py < -halfBoxLength)
		{
			SpheresCPU[i].py = -2.0*halfBoxLength - SpheresCPU[i].py;
			SpheresCPU[i].vy = - SpheresCPU[i].vy;
		}
			
		if(SpheresCPU[i].pz > halfBoxLength)
		{
			SpheresCPU[i].pz = 2.0*halfBoxLength - SpheresCPU[i].pz;
			SpheresCPU[i].vz = - SpheresCPU[i].vz;
		}
		else if(SpheresCPU[i].pz < -halfBoxLength)
		{
			SpheresCPU[i].pz = -2.0*halfBoxLength - SpheresCPU[i].pz;
			SpheresCPU[i].vz = - SpheresCPU[i].vz;
		}
	}
	
}

void get_forces()
{
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
	for (int i = 0; i < NUMBER_OF_SPHERES - 1; i++)
	{
		for (int j = 0; j < NUMBER_OF_SPHERES; j++)
		{
			dx = SpheresCPU[j].px - SpheresCPU[i].px;
			dy = SpheresCPU[j].py - SpheresCPU[i].py;
			dz = SpheresCPU[j].pz - SpheresCPU[i].pz;
				
			r2 = dx*dx + dy*dy + dz*dz;
			r = sqrt(r2);
			
			forceMag =  SpheresCPU[i].mass*SpheresCPU[j].mass*GRAVITY/r2;
			
			if (r < DIAMETER)
			{
				dvx = SpheresCPU[j].vx - SpheresCPU[i].vx;
				dvy = SpheresCPU[j].vy - SpheresCPU[i].vy;
				dvz = SpheresCPU[j].vz - SpheresCPU[i].vz;
				inout = dx*dvx + dy*dvy + dz*dvz;
				if(inout <= 0.0)
				{
					forceMag +=  SPRING_STRENGTH*(r - DIAMETER);
				}
				else
				{
					forceMag +=  SPRING_REDUCTION*SPRING_STRENGTH*(r - DIAMETER);
				}
			}
			SpheresCPU[i].fx = forceMag*dx/r;
			SpheresCPU[i].fy = forceMag*dy/r;
			SpheresCPU[i].fz = forceMag*dz/r;
			SpheresCPU[j].fx = -forceMag*dx/r;
			SpheresCPU[j].fy = -forceMag*dy/r;
			SpheresCPU[j].fz = -forceMag*dz/r;
			
       	}
       }
}

void move_bodies(float time)
{
	if(time == 0.0)
	{
		for (int i=0; i<NUMBER_OF_SPHERES; i++)
		{
			SpheresCPU[i].vx += 0.5*DT*(SpheresCPU[i].fx - DAMP*SpheresCPU[i].vx)/SpheresCPU[i].mass;
			SpheresCPU[i].vy += 0.5*DT*(SpheresCPU[i].fy - DAMP*SpheresCPU[i].vy)/SpheresCPU[i].mass;
			SpheresCPU[i].vz += 0.5*DT*(SpheresCPU[i].fz - DAMP*SpheresCPU[i].vz)/SpheresCPU[i].mass;
		}
	}
	else
	{
		for (int i=0; i<NUMBER_OF_SPHERES; i++)
		{
			SpheresCPU[i].vx += DT*(SpheresCPU[i].fx - DAMP*SpheresCPU[i].vx)/SpheresCPU[i].mass;
			SpheresCPU[i].vy += DT*(SpheresCPU[i].fy - DAMP*SpheresCPU[i].vy)/SpheresCPU[i].mass;
			SpheresCPU[i].vz += DT*(SpheresCPU[i].fz - DAMP*SpheresCPU[i].vz)/SpheresCPU[i].mass;
		}
	}

	for (int i=0; i<NUMBER_OF_SPHERES; i++)
	{
		SpheresCPU[i].px += DT*SpheresCPU[i].vx;
		SpheresCPU[i].py += DT*SpheresCPU[i].vy;
		SpheresCPU[i].pz += DT*SpheresCPU[i].vz;	
	}
	
	
	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;
	SpheresCPU= (sphereStruct*)malloc(NUMBER_OF_SPHERES*sizeof(sphereStruct));
	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}

