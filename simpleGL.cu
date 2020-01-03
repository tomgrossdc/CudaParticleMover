////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/
#include <typeinfo>


//#include "main.h"
//#include "mesh.h"
//#include "data.h"
//#include "simpleGL.h"


// includes, system
//#include "simpleGL.h"
// try to add a global struct for reference to internal routines
/*
PPart *host_P;
PPart *dev_P;
MMesh *dev_MM;
MMesh *MM;
DData *dev_DD;
DData *DD;
int DD3[4];
*/

//void move(float4 *pos, struct PPart *PP,struct MMesh *MM, struct DData *DD, float time_now);

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////




void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL()
{
    int argc = 1;
    char *argv[1] = {(char*)"Something"};
    glutInit(&argc, argv);
    //glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Regular NetCDF Particle Tracking");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0050, 0.005, 0.0050, 1.0);  // background color  0,0,0,1 is black
    glColor4f(0.0,1.0,0.0,1.0);   // set color
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//  Initialize a few gl, timer and cuda 
//  then start gl looping to call function display
////////////////////////////////////////////////////////////////////////////////
bool GLmoveparticle(struct PPart *PP, struct MMesh *MM, struct DData *DD)
{
    //int DD3[4];
    // Create the CUTIL timer
    sdkCreateTimer(&timer);
    printf("TFG GLmoveparticle   starting...\n");
    printf("TFG GLmoveparticle DD[0].time      %g %g %g %g\n",DD[0].time/3600,DD[1].time/3600,DD[2].time/3600,DD[3].time/3600);
    printf("TFG GLmoveparticle DD[0].DD3      %d %d %d %d\n",DD[0].DD3[0],DD[0].DD3[1],DD[0].DD3[2],DD[0].DD3[3]);
    g_time_now = (DD[0].time + DD[1].time)/2.0; 

    //initial the cudaDevice to use, as if there is a choice?
    cudaDeviceProp deviceProp;
    int devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n"
         , devID, deviceProp.name, deviceProp.major, deviceProp.minor);


        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        //if (false == initGL(&argc, argv))        
        if (false == initGL())

        {
            return false;
        }

        // register callbacks.   these are locally defined functions
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        // create VBO
        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // size_t DDSizeGeneral = sizeof(DData)*4;
        // cudaMemcpy(DD, dev_DD,DDSizeGeneral,cudaMemcpyDeviceToHost);


        // run the cuda part from routine display 
        // specified in glutDisplayFunc(display);
        // which is triggered by glutMainLoop
        //runCuda(&cuda_vbo_resource);

        // start rendering mainloop
        printf(" Start glutMainLoop  >display>runCuda \n\n");

        glutMainLoop();

        printf(" Return from glutMainLoop\n");

//    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation,  called from display
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    //printf("TFG runCuda host_P[10].x_present %g\n",host_P[10].x_present);
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    float time_now;

    size_t DDSizeGeneral = sizeof(DData)*4;
    size_t MMSizeGeneral = sizeof(MMesh)*4;

    if (iDD==-1){   // First update, need to localize DD, MM only once
                    // initialized in simpleGL.h, global to this file
                    
        printf("\n runCuda First Pass\n");
            try {
                printf(" Can I print DD[0].time_now %g\n",DD[0].time_now);
            } catch (const std::runtime_error& e){
                printf(" Error on print DD[0].time_now Message: %s\n",e.what());
            }
        cudaMemcpy(DD, dev_DD,DDSizeGeneral,cudaMemcpyDeviceToHost);
        cudaMemcpy(MM, dev_MM,MMSizeGeneral,cudaMemcpyDeviceToHost);
                printf(" After cudaMemcpy  DD[0].time_now %fsec %f hr\n",DD[0].time_now,DD[0].time_now/3600.);

        iDD=0;
    }
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
    *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
    

    //int DD33=1;   256,64
    move3d<<< 256,64 >>>(dptr,dev_P,dev_MM,dev_DD);
    cudaDeviceSynchronize();
    DD[0].time_now += CUDA_STEPS* DT_SEC;   // 0.01f;   
    time_now = DD[0].time_now;
//printf("After cuda move3d time_now = %fs %ghr\n",time_now, time_now/3600.);
    float time_frac=(time_now - DD[DD[0].DD3[0]].time)/(DD[DD[0].DD3[2]].time - DD[DD[0].DD3[0]].time);
    bool timetest =  (time_frac > .75);
    if (timetest ){
        //  Every hour a new data file is needed. Read dev_DD to obtain time_now
        
        // Assume or test that the fourth ReadData thread is finished and move to dev_DD  BROKEN
        cudaMemcpy(dev_DD,DD,DDSizeGeneral,cudaMemcpyHostToDevice);

        //  Update DD3  
        for (int i=0; i<4 ; i++)DD[0].DD3[i]=(DD[0].DD3[i]+1)%4;

        // DD3[3] is next spot to be updated, will be updated in this section
        //  Thread this off to execute while elsewhere.
        //        printf(" DD[# 1].time = %g %g %g %g\n",DD[0].time/3600.,DD[1].time/3600.,DD[2].time/3600.,DD[3].time/3600.);
        
        //  New generated filename routine:
        DD[0].ToDay +=3600;  // for hourly files
        string newername = NetCDFfiledate(DD[0].filetemplate,DD);

        bool RunThreadRead = true;
        if (RunThreadRead){
            std::thread t1(ReadFieldNetCDF, std::ref(newername),std::ref(DD[0].DD3[3]),
            std::ref(DD),std::ref(MM) );
            t1.join();   // Wait here for thread to finish. Makes threading moot.  Testing only.
            //t1.detach();    // Let it loose, but with no test for finished crashes
            }
        else{
            ReadFieldNetCDF(newername,DD[0].DD3[3],DD,MM);
            }
/*
// List Nvidia resources to see if it is growing.
//checkGpuMem();
float free_m,total_m,used_m;
size_t free_t,total_t;
cudaMemGetInfo(&free_t,&total_t);
free_m =(uint)free_t/1048576.0 ;
total_m=(uint)total_t/1048576.0;
used_m=total_m-free_m;
printf ( "  mem free %ld .... %f MB \n  mem total %ld....%f MB mem used %f MB\n"
  ,free_t,free_m,total_t,total_m,used_m);
*/
        float dhr=3600.;
        printf(" DD[     0:3].time = %g %g %g %g\n",DD[0].time/dhr,DD[1].time/dhr,DD[2].time/dhr,DD[3].time/dhr);
        printf(" DD[DDT[0:3]].time = %g %g %g %g\n",
           DD[DD[0].DD3[0]].time/dhr,DD[DD[0].DD3[1]].time/dhr,DD[DD[0].DD3[2]].time/dhr,DD[DD[0].DD3[3]].time/dhr);

        iDD+=1;
        printf(" iDD = %d\n\n",iDD);
    }    // End of hourly DD update

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    size = MAX_GLPARTICLES *4*sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.0, 1.0, 0.0);     // Color of points

    glPointSize(1);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);

    //glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDrawArrays(GL_POINTS, 0, MAX_GLPARTICLES);

    glDisableClientState(GL_VERTEX_ARRAY);
    char buffer[25] ;
    float myFloat = g_time_now/3600.;   // convert to hours
    int ret =snprintf(buffer, sizeof buffer, "time_now = %.2f hr", myFloat);
    //and more";
    //char hello[] = str;
    //glutSetWindowTitle(buffer);
    
    /*if ( myFloat > 5.) {
        glutDestroyWindow(glutGetWindow());
        return;
        }
    */
    glutSwapBuffers();

    sdkStopTimer(&timer);
    computeFPS();    

}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
        case (112) : 
        { std::chrono::milliseconds timespan(5000); std::this_thread::sleep_for(timespan);}    // sleep for 5sec = 5000ms
        break ;
        case (104) : 
        {printf("\nesc = stop\n p = 5sec pause\n h = this help\n");} break;
    }
    printf("key = %d\n",key);  // p pause is 112
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)     // Rotate around x and y axis pitch and yaw
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 2) // magnification  z axis move push down on scroll button and move mouse
    {
        translate_z += dy * 0.01f;
    }
    else if(mouse_buttons & 4)    // Translate side to side or up and down
    { //printf("mouse button 2\n");
        translate_x += dx * 0.01f;
        translate_y -= dy * 0.01f;}

    else if(mouse_buttons & 3)
    { printf("mouse button 3\n");}
    else if(mouse_buttons & 0)
    { printf("mouse button 0\n");}
    //else 
    //   printf(" else mouse button = %d\n",mouse_buttons);

    mouse_old_x = x;
    mouse_old_y = y;
}



//Fancy cuda kernel can be called using dev_P, dev_MM, dev_DD   but define it with local names
// move<<<  >>> ( pos,dev_P,dev_MM,dev_DD);

__global__
void move(float4 *pos, struct PPart *PP,struct MMesh *MM, struct DData *DD){

//Make sure that PP will fit into pos
    //unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    //unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
unsigned int maxGLnum = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
maxGLnum = MAX_GLPARTICLES;
float scale = SCALE_GL;
unsigned int Ipx;
int DDT0, DDT2;
int DD3[3];
DD3[0]=DD[0].DD3[0];
DD3[1]=DD[0].DD3[1];
DD3[2]=DD[0].DD3[2];
DDT0=DD3[0];
//DDT1=DD3[1];
DDT2=DD3[2];

/*   real stuff now  */
int DeBuG = false;   //   true or false
int DeBuGIP = NUM_PARTICLES/2;
int i_ele, keepgoing, k;
float xpart, ypart;
float smallest_value = -0.01000; // -0.001; 
//float time_now;
// float      now a passed argument
float time_now = DD[0].time_now;    // Will use dev_DD after the first pass with new DD
float time_frac=(time_now - DD[DDT0].time)/(DD[DDT2].time - DD[DDT0].time);
double dt_sec=DT_SEC;
int igrounded;

//  Cuda strides
int cudaindex = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
 
if (cudaindex==-1 | cudaindex==-2){  //  0 2500
    printf(" t.x=%d blockIdx.x=%d blockDim.x=%d gridDim.x=%d cudaindex=%d stride=%d\n",
threadIdx.x,blockIdx.x,blockDim.x, gridDim.x,cudaindex,stride);
}
// Main time loop. Loop CUDA_STEPS times between returns for plotting
for (int itime=0; itime<CUDA_STEPS; itime++){

//loop on particles for this cuda up to max PP[0].num_P
    igrounded =0;
for(int Ip = cudaindex; Ip <NUM_PARTICLES; Ip += stride){

  // Find surrounding triangle of Particle
  i_ele = PP[Ip].i_ele;
  xpart = PP[Ip].x_present;
  ypart = PP[Ip].y_present;

  if (DeBuG && (itime%100==0) && Ip==DeBuGIP) 
  {printf(" Move itime=%d i_ele=%d, PP.i_ele=%d,  MM[0].ele0=%ld, MM[0].ele2=%ld MM[0].ele2=%ld\n",
  itime, i_ele,PP[Ip].i_ele,MM[0].ele[i_ele][0],MM[0].ele[i_ele][1],MM[0].ele[i_ele][2]);}
  

    //  Check for out of domain/ grounded particle
    //  do work if in-domain  else increment igrounded and skip main part of move
if (i_ele >= 0) { 

  keepgoing = 1; 
  while (keepgoing  == 1){


//  if any of the f's are negative, walk that way and restart while loop
   k=0;
   PP[Ip].factor[k]=MM[0].a_frac[i_ele][k]*xpart + 
      MM[0].b_frac[i_ele][k]*ypart + MM[0].c_frac[i_ele][k];
   if ( PP[Ip].factor[k] < smallest_value) { 
   	i_ele = MM[0].tri_connect[i_ele][0]; 
   }
   else { 
      k=1;
      PP[Ip].factor[k]=MM[0].a_frac[i_ele][k]*xpart + MM[0].b_frac[i_ele][k]*ypart + MM[0].c_frac[i_ele][k];
      if ( PP[Ip].factor[k] < smallest_value ) { 
      	  i_ele = MM[0].tri_connect[i_ele][1] ; 
      }
      else { 
          k=2;
          PP[Ip].factor[k]=MM[0].a_frac[i_ele][k]*xpart + MM[0].b_frac[i_ele][k]*ypart + MM[0].c_frac[i_ele][k];
          if ( PP[Ip].factor[k] < smallest_value ) { 
	  	i_ele = MM[0].tri_connect[i_ele][2] ;
          }
          else {
             //  Found it, iele,   all f's are positive 
             keepgoing = 0;
	  }
      }
   }
   if (i_ele < 0) {    // newly grounded particle, zero him out.
     	PP[Ip].factor[0]=0.0; PP[Ip].factor[1]=0.0; PP[Ip].factor[2]=0.0;
         keepgoing = 0;
         igrounded++;
         // newly grounded
         PP[Ip].i_ele = i_ele;
   }
 }   // end of while keepgoing 

if (i_ele>=0){     // good particle still in the mesh
 PP[Ip].i_ele =i_ele; 

 // moveing through particle.move
   float factor0=PP[Ip].factor[0];
   float factor1=PP[Ip].factor[1];
   float factor2=PP[Ip].factor[2];
 
   // i_ele is element, ele1 is node index of corner
   //  node = ele[ele_index][corner_index] 
   long ele0 = MM[0].ele[i_ele][0];
   long ele1 = MM[0].ele[i_ele][1];
   long ele2 = MM[0].ele[i_ele][2];

// Found new i_ele , ele0, ele1, ele2 and have factor1, factor2, factor0

  int sigma_level=0;   // Need to upgrade to 3d sometime....
  double u[3];   // velocities at corners of surrounding triangle
  double v[3];
  double w[3];
  float a,b,c;
// UVW space interpolation from the three surrounding points[ele 012]
  for (int i=0; i<3; i++){
    u[i]= factor0 * DD[DD3[i]].U[sigma_level][ele0] 
    + factor1*DD[DD3[i]].U[sigma_level][ele1]
    + factor2*DD[DD3[i]].U[sigma_level][ele2];    
    v[i]= factor0 * DD[DD3[i]].V[sigma_level][ele0] 
    + factor1*DD[DD3[i]].V[sigma_level][ele1]
    + factor2*DD[DD3[i]].V[sigma_level][ele2];
    w[i]= factor0 * DD[DD3[i]].W[sigma_level][ele0] 
    + factor1*DD[DD3[i]].W[sigma_level][ele1]
    + factor2*DD[DD3[i]].W[sigma_level][ele2];
    }
  // formula for quadratic time interpolation of three points, assuming equal spacing
  //  time_frac = 0  at DD[0].time,  time_frac = .5 DD[1].time,  time-frac = 1.0 DD[2].time
  a =  2.*u[2] -4.*u[1] +2.*u[0];
  b = -   u[2] +4.*u[1] +   u[0];
  c =                       u[0];
  float Upnow = a*time_frac*time_frac + b*time_frac + c;

  a =  2.*v[2] -4.*v[1] +2.*v[0];
  b = -   v[2] +4.*v[1] +   v[0];
  c =                       v[0];   
  float Vpnow = a*time_frac*time_frac + b*time_frac + c;

  a =  2.*w[2] -4.*w[1] +2.*w[0];
  b = -   w[2] +4.*w[1] +   w[0];
  c =                       w[0];   
  float Wpnow = a*time_frac*time_frac + b*time_frac + c; 


  /*  Now have time and space interpolates of U,V,W for particle */
  /* Apply them to the particle coordinates and done! 
   (unless temporal runge kutta is needed. 
    Running goofy small time steps)*/

    PP[Ip].x_present += dt_sec*(Upnow*1.) ; 
    PP[Ip].y_present += dt_sec*(Vpnow*1.); 
    PP[Ip].z_present += dt_sec*Wpnow*1.;
    
    // using an Xbox from Particles in meters
    /*float shrinkage = 1.0;   //dec 4 1.5 works
            if (PP[Ip].x_present < MM[0].Xbox[0]/shrinkage) PP[Ip].x_present = MM[0].Xbox[0]/shrinkage;
            if (PP[Ip].x_present > MM[0].Xbox[1]/shrinkage) PP[Ip].x_present = MM[0].Xbox[1]/shrinkage;
            if (PP[Ip].y_present < MM[0].Xbox[2]/shrinkage) PP[Ip].y_present = MM[0].Xbox[2]/shrinkage;
            if (PP[Ip].y_present > MM[0].Xbox[3]/shrinkage) PP[Ip].y_present = MM[0].Xbox[3]/shrinkage;
    */

    Ipx = Ip%maxGLnum;   // incase we are moving more particles than can be plotted
    Ipx = Ip%MAX_GLPARTICLES;
    pos[Ipx] = make_float4(scale*PP[Ip].x_present,-scale*PP[Ip].z_present,-scale*PP[Ip].y_present,  1.0f);

}  //other if iele>0 loop end    
}
else
{    PP[Ip].factor[0]=0.0; PP[Ip].factor[1]=0.0; PP[Ip].factor[2]=0.0;
    PP[Ip].i_ele =i_ele; 
    //return i_ele;   leave the particle loop
    igrounded++;
    //break;
}

   PP[Ip].time_now = time_now;
   // End of Particle loop on Ip
 
}

// End of a time step, increment to next  time_now += dt_sec;
// if time_frac >1, then it will fall out of the loop and not increment PP.timenow

time_now += dt_sec;
time_frac=(time_now - DD[DDT0].time)/(DD[DDT2].time - DD[DDT0].time);

}
//printf("end of move");
// end of move()
if ( cudaindex==0) DD[0].time_now = time_now;   // Only update dev_DD[] once
//  Hopefully the other cudas have gotten started by now and don't need to read dev_DD[0].time_now
}
//  End of move()


////////////////////////////////////////////////////////////////////////
///////////////////  move3d  ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
//cuda kernel with four meshes and depth for 3d UVW's read from Field files
//uses MM[0:2] for the UVW and MM[2] to provide angle and depth
// move3d<<<  >>> ( pos,dev_P,dev_MM,dev_DD);

__global__
void move3d(float4 *pos, struct PPart *PP,struct MMesh *MM, struct DData *DD){
// Cuda Kernal to move the particles
// loop on all particles using cudaindex and stride
// for each particle find i_ele, depth angle     findiele
// interpolate sigma coordinate, find three corner values, average them to PP[iP].xyz
// Did that with all three time steps. Time interpolate
// Step PP[iP] position forward.

int IpTest=-250;
//int DeBuG = false;   //   true or false

/*   real stuff now  */
//float time_now;       //    now a passed argument
//float time_now = DD[0].time_now;    // Will use dev_DD after the first pass with new DD
//float time_frac=(time_now - DD[DDT0].time)/(DD[DDT2].time - DD[DDT0].time);
double dt_sec=DT_SEC;
//float deg2pi = 3.1415926/180.;

//  Cuda strides
int cudaindex = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

//if (cudaindex==-1 | cudaindex==-2){  //  0 2500
//    printf(" t.x=%d blockIdx.x=%d blockDim.x=%d gridDim.x=%d cudaindex=%d stride=%d\n",
//    threadIdx.x,blockIdx.x,blockDim.x, gridDim.x,cudaindex,stride);
//}

// Main time loop. Loop CUDA_STEPS times between returns for plotting
double  time_now = DD[0].time_now;
//if (cudaindex==0) printf("move3d %d, time_now= %fs %gh\n",cudaindex,time_now,time_now/3600.);
for (int itime=0; itime<CUDA_STEPS; itime++){
    
    
    for(int Ip = cudaindex; Ip <NUM_PARTICLES; Ip += stride){
        
        // Find surrounding triangle of Particle for all three meshes
        //  PP[Ip].i_ele4[iMM]
        //  PP[Ip].factor4[iMM][0:2]
        for (int iMM=0; iMM<3; iMM++) 
        {
          /*  if (Ip==-1 && itime==0) {
                printf("move3d before findiele itime= %d, time_now= %fs %gh\n",itime,time_now,time_now/3600.);
                for (int i_ele=0; i_ele<47184; i_ele+=12500)
                printf(" iMM = %d ele_func_tripart i_ele = %d ; triconnect = %ld\n",
                    iMM, i_ele, MM[iMM].tri_connect[i_ele][0]);
            }
          */
          //    findiele(Ip,iMM,PP,MM); 
                findielefake(Ip,iMM,PP,MM); 

        }  
        PP[Ip].answer[0]=0.0;

        if (Ip==IpTest && itime==0) printf(" move3d finished findielefake %d\n",itime);
        // interpolate values for angle and depth at PP[Ip].x,y    
        float VAR[3];
        int iMM=2;    //  mesh for w, angle and depth
        for (int i=0; i<3; i++) 
        { // i_ele is the element, ele[i_ele[0:2] are the nodes at corners of triangle i_ele
            long elei = MM[iMM].ele[PP[Ip].i_ele4[iMM]][i];
            VAR[i]=MM[iMM].ANGLE[elei];
        }
        if (Ip==IpTest && itime==0) printf("move3d before Interpolate2D findiele itime= %d, time_now= %fs %gh\n",itime,time_now,time_now/3600.);
        Interpolate2D(Ip,iMM,PP,VAR); 
        float angle=PP[Ip].answer[0];
        //or
        iMM=2; for (int i=0; i<3; i++) VAR[i]=MM[iMM].depth[MM[iMM].ele[PP[Ip].i_ele4[iMM]][i]];
        if (Ip==IpTest) printf(" depths = %g %g %g \n",VAR[0],VAR[1],VAR[2]);
        Interpolate2D(Ip,iMM,PP,VAR);  
        float depth=PP[Ip].answer[0]; 
        if (Ip==IpTest && itime==0) printf("move3d after Interpolate2D angle[%d]=%g  depth=%g\n",Ip,angle,depth);

        // Find the isigmap, isigmam and sigmafrac 
        //  U[iP] = U[isigmap]*sigmafrac +U[isigmam]*(1.0-sigmafrac)
        // do three times and use timefrac to produce final VAR[3] at corners
        // iMM = 0U  1V  2W   
        // Only works for UVW.  In future add special cases 3T 4S iMM=iMM-1 or -2 
        if (Ip==IpTest && itime==0) 
          printf("move3d before Interpolatesigma itime= %d, time_now= %fs %gh\n"
               ,itime,time_now,time_now/3600.);
        iMM=0; Interpolatesigma(Ip, iMM, PP, DD, MM, depth,time_now); 
        float Up=PP[Ip].answer[0];
        iMM=1; Interpolatesigma(Ip, iMM, PP, DD, MM, depth,time_now); 
        float Vp=PP[Ip].answer[0];
        //Up = 1.0; 
        //Vp = 0.0; 
        float cosa = cos(angle);
        float sina = sin(angle);
        float Upnow = cosa*Up -sina*Vp;
        float Vpnow = sina*Up +cosa*Vp;
        iMM=2; Interpolatesigma(Ip, iMM, PP, DD, MM, depth,time_now); 
        float Wpnow=PP[Ip].answer[0];
        //Wpnow = 0.0;
        if (Ip==IpTest && itime==0) printf("move3d after sigma UVp[%d]= %g %g UVWpnow= %g, %g, %g  angle=%g  depth=%g\n"
           ,Ip,Up,Vp,Upnow, Vpnow, Wpnow,angle,depth);

        
        /*  Now have time and space interpolates of U,V,W for particle */
        /* Apply them to the particle coordinates and done! 
        (unless temporal runge kutta is needed. 
            Running goofy small time steps)*/
            
            PP[Ip].x_present += dt_sec*(Upnow*1.) ; 
            PP[Ip].y_present += dt_sec*(Vpnow*1.); 
            PP[Ip].z_present += dt_sec*Wpnow*1.0;
            PP[Ip].z_present = min(PP[Ip].z_present, -0.01);       // if z_p is above -0.01
            PP[Ip].z_present = max(PP[Ip].z_present, -depth);    // if z_p is below -depth

            // End of Particle loop on Ip
        }
        
        // End of a time step, increment to next  time_now += dt_sec;
        // if time_frac >1, then it will fall out of the loop and not increment PP.timenow
        //if (cudaindex==2) printf(" 1=%lf",time_now);

        time_now+=dt_sec;    
        //if (cudaindex==2 && itime==0) printf("move3d end timeloop itime= %d, time_now= %lfs %lfh\n",itime,time_now,time_now/3600.);
        //if (cudaindex==2 && itime==(CUDA_STEPS-1)) printf("move3d end timeloop itime= %d, time_now= %lfs %lfh\n",itime,time_now,time_now/3600.);
        //if (cudaindex==2) printf(" 2=%lf",time_now);
    }
    // Update the VBO  pos[]
    //unsigned int maxGLnum = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
    //maxGLnum = MAX_GLPARTICLES;
    float scale = SCALE_GL;
    for(int Ip = cudaindex; Ip <NUM_PARTICLES; Ip += stride){
        int Ipx = Ip%MAX_GLPARTICLES;
        pos[Ipx] = make_float4(scale*PP[Ip].x_present,1000.*scale*PP[Ip].z_present,-scale*PP[Ip].y_present,  1.0f);
    }
    
    if (cudaindex==-1){
        printf("end of move,  pos[0][0]=%g\n",pos[cudaindex]);
    }
    // end of move()
    if ( cudaindex==0) DD[0].time_now = time_now;   // Only update dev_DD[] once
    //  Hopefully the other cudas have gotten started by now and don't need to read dev_DD[0].time_now
}

/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
// update which triangle the iP is in for the three meshes   iMM 012 UVW  
//           note the iMM=2 also gives iele and factors for ANGLE and depth
// Sets PP[Ip].i_ele4[iMM] and the factors  PP[Ip].factor4[iMM][0:2] 
// for (in iMM=0; iMM<3; iMM++)
__device__ void findielefake(int Ip,int iMM,struct PPart *PP, struct MMesh *MM)
{


    int i_ele, keepgoing, k;
    float xpart, ypart;
    float smallest_value = -0.01000; //  -.01  -0.001; 

    // Find surrounding triangle of Particle
    i_ele = PP[Ip].i_ele4[iMM];
    xpart = PP[Ip].x_present;
    ypart = PP[Ip].y_present;
    //return;
    //if(Ip==0) printf(" start findiele i_ele=%d \n",i_ele);
    //  Check for out of domain/ grounded particle
    //  do work if in-domain  else increment igrounded and skip main part of move
    if (i_ele >= 0) { 
        
        keepgoing = 1; 
        while (keepgoing  > 0 ){
            
            //if(Ip==0) printf(" keepgoing findielefake i_ele=%d ",i_ele);

            //  if any of the f's are negative, walk that way and restart while loop
            k=0;
            PP[Ip].factor4[iMM][k]=MM[iMM].a_frac[i_ele][k]*xpart + 
            MM[iMM].b_frac[i_ele][k]*ypart + MM[iMM].c_frac[i_ele][k];
            if ( PP[Ip].factor4[iMM][k] < smallest_value) { 
                i_ele = MM[iMM].tri_connect[i_ele][0]; 
            }
            else { 
                k=1;
                PP[Ip].factor4[iMM][k]=MM[iMM].a_frac[i_ele][k]*xpart + MM[iMM].b_frac[i_ele][k]*ypart + MM[iMM].c_frac[i_ele][k];
                if ( PP[Ip].factor4[iMM][k] < smallest_value ) { 
                  i_ele = MM[iMM].tri_connect[i_ele][1] ; 
            }
            else { 
                k=2;
                PP[Ip].factor4[iMM][k]=MM[iMM].a_frac[i_ele][k]*xpart + MM[iMM].b_frac[i_ele][k]*ypart + MM[iMM].c_frac[i_ele][k];
                if ( PP[Ip].factor4[iMM][k] < smallest_value ) { 
                i_ele = MM[iMM].tri_connect[i_ele][2] ;
                }
                else {
                   //  Found it, iele,   all f's are positive 
                   keepgoing = 0;
            }
            }
         }
         if (i_ele < 0) {    // newly grounded particle, zero him out.
               PP[Ip].factor4[iMM][0]=0.0; PP[Ip].factor4[iMM][1]=0.0; PP[Ip].factor4[iMM][2]=0.0;
               PP[Ip].i_ele4[iMM] = i_ele;
               keepgoing = 0;
         }
         if (keepgoing>0) keepgoing++;
         if (keepgoing > 500) { //printf(" k%d  ",Ip);
         i_ele=-1;
         PP[Ip].i_ele4[iMM] = -1;
         PP[Ip].x_present=0.0;
         PP[Ip].y_present=0.0;
         PP[Ip].z_present=0.0;

         keepgoing=0;}
       }   
       
       //return;
       if (i_ele>=0){     // good particle still in the mesh
        PP[Ip].i_ele4[iMM]=i_ele;}
       // end of while keepgoing 
      

}      
//if(Ip==0) printf(" end findielefake i_ele=%d \n",i_ele);
 




    return;
}
__device__ void findiele(int Ip,int iMM,struct PPart *PP, struct MMesh *MM)
{
    int i_ele, keepgoing, k;
    float xpart, ypart;
    float smallest_value = -0.01000; // -0.001; 

    // Find surrounding triangle of Particle
    i_ele = PP[Ip].i_ele4[iMM];
    xpart = PP[Ip].x_present;
    ypart = PP[Ip].y_present;
    //if(Ip==0) printf(" start findiele i_ele=%d \n",i_ele);
    //  Check for out of domain/ grounded particle
    //  do work if in-domain  else increment igrounded and skip main part of move
    if (i_ele >= 0) { 
        
        keepgoing = 1; 
        while (keepgoing  == 1){
            
            //if(Ip==0) printf(" keepgoing findiele i_ele=%d \n",i_ele);

            //  if any of the f's are negative, walk that way and restart while loop
            k=0;
            PP[Ip].factor4[iMM][k]=MM[iMM].a_frac[i_ele][k]*xpart + 
            MM[iMM].b_frac[i_ele][k]*ypart + MM[iMM].c_frac[i_ele][k];
            if ( PP[Ip].factor4[iMM][k] < smallest_value) { 
                i_ele = MM[iMM].tri_connect[i_ele][0]; 
            }
            else { 
                k=1;
                PP[Ip].factor4[iMM][k]=MM[iMM].a_frac[i_ele][k]*xpart + MM[iMM].b_frac[i_ele][k]*ypart + MM[iMM].c_frac[i_ele][k];
                if ( PP[Ip].factor4[iMM][k] < smallest_value ) { 
                  i_ele = MM[iMM].tri_connect[i_ele][1] ; 
            }
            else { 
                k=2;
                PP[Ip].factor4[iMM][k]=MM[iMM].a_frac[i_ele][k]*xpart + MM[iMM].b_frac[i_ele][k]*ypart + MM[iMM].c_frac[i_ele][k];
                if ( PP[Ip].factor4[iMM][k] < smallest_value ) { 
                i_ele = MM[iMM].tri_connect[i_ele][2] ;
                }
                else {
                   //  Found it, iele,   all f's are positive 
                   keepgoing = 0;
            }
            }
         }
         if (i_ele < 0) {    // newly grounded particle, zero him out.
               PP[Ip].factor4[iMM][0]=0.0; PP[Ip].factor4[iMM][1]=0.0; PP[Ip].factor4[iMM][2]=0.0;
               PP[Ip].i_ele = i_ele;
               keepgoing = 0;
         }
       }   // end of while keepgoing 
       
       PP[Ip].i_ele4[iMM]=i_ele;

}      
//if(Ip==0) printf(" end findiele i_ele=%d \n",i_ele);

}


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
// find 2d interpolated ANGLE(icase==0), depth(icase==1)
//  input is X,Y, A of i_ele points along with factor4
// MM[iMM].X[i_ele4[0:2]] MM[iMM].Y[i_ele4[0:2]] MM[iMM].ANGLE[i_ele4[0:2]] 
// PP[Ip].factor4[iMM][i_ele[0:]]    
// 2Dinterpolate(Ip,iMM,PP,MM,icase);  // icase = 0U, 1V, 2W, 3ANGLE, 4depth  
//    maybe do  VAR[3] = MM[iMM].ANGLE[i_ele4[0:2]]  instead of icase 
//    That way we can feed it the vertical interpolates of UVW[3]
//float VAR[3];
//iMM=3; for (int i=0; i<3; i++) VAR[i]=MM[iMM].angle[PP[Ip].iele4[iMM][i]];
//float angle = 2Dinterpolate(Ip,iMM,PP,MM,VAR);
//iMM=4; for (int i=0; i<3; i++) VAR[i]=MM[iMM].depth[PP[Ip].iele4[iMM][i]];
//float depth = 2Dinterpolate(Ip,iMM,PP,MM,VAR); 

__device__ void Interpolate2D(int Ip, int iMM, struct PPart *PP, float *VAR)
{
    /*
    iMM=3; 
    for (int i=0; i<3; i++) 
    { // i_ele is the element, ele[i_ele[0:2] are the nodes at corners of triangle i_ele
        long elei = MM[iMM].ele[PP[Ip].iele4[iMM][i]];
        VAR[i]=MM[iMM].angle[elei];
    }
    float angle = 2Dinterpolate(Ip,iMM,PP,MM,VAR);
    */

    float factor0=PP[Ip].factor4[iMM][0];
    float factor1=PP[Ip].factor4[iMM][1];
    float factor2=PP[Ip].factor4[iMM][2];
    
   PP[Ip].answer[0] = factor0*VAR[0]+factor1*VAR[1]+factor2*VAR[2];

}


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

__device__ void Interpolatesigma(int Ip, int iMM, 
    struct PPart *PP, struct DData *DD, struct MMesh *MM, float depth, float time_now )
{
    // Find the isigmap, isigmam and sigmafrac 
    //  U[iP] = U[isigmap]*sigmafrac +U[isigmam]*(1.0-sigmafrac)
    // do three times and use timefrac to produce final VAR[3] at corners
    // iMM = 0U  1V  2W   
    // Only works for UVW.  In future add special cases 3T 4S iMM=iMM-1 or -2     int DDT0, DDT2;
    int IpTest = -250;

    int i_ele = PP[Ip].i_ele4[iMM];   
    float vart[3];
    float var[3]; 
    int sp, sm;

    float sigIp = PP[Ip].z_present / depth ;    // 0 to -1.0
    //  count up in sp to walk down in depth
    sp=1;
    while(MM[iMM].sigma[sp]< sigIp) sp++;  // increment if sp is still above sigIp
    sm = sp-1;                         // sp is below sigIp,  sm is above
    float sigfrac = (sigIp-MM[iMM].sigma[sp])/(MM[iMM].sigma[sm]- MM[iMM].sigma[sp]);
    
    // Pick out the three DD's to interpolate in time
    int DD3[3];
    DD3[0]=DD[0].DD3[0];
    DD3[1]=DD[0].DD3[1];
    DD3[2]=DD[0].DD3[2];
    int DDT0=DD3[0];
    //DDT1=DD3[1];
    int DDT2=DD3[2];
    
    if (Ip==IpTest  ) printf(" start of interpretsigma iMM=%d  z_present= %g /depth=%g =sigIP = %g \n  sm,sp sigma[%d]=%g sigma[%d]=%g sigIP %g sigfrac %g\n"
    ,iMM,PP[Ip].z_present, depth,sigIp,sm,MM[iMM].sigma[sm],sp,MM[iMM].sigma[sp],sigIp,sigfrac);
    // loop on time DD3[i]
    // loop on three corners ele[i_ele][j]
    // average sm and sp at the corner
    
    for (int it=0; it<3; it++){  // time loop for DD3[it]
        for (int j=0; j<3; j++){     // loop around corners to get sigma averaged variable
            long ele0=MM[iMM].ele[i_ele][j];
            if      (iMM==0){ // U
                var[j] = DD[DD3[it]].U[sm][ele0]*sigfrac  + DD[DD3[it]].U[sp][ele0]* (1.0 - sigfrac);
            }
            else if (iMM==1){ // V    
                var[j] = DD[DD3[it]].V[sm][ele0]*sigfrac  + DD[DD3[it]].V[sp][ele0]* (1.0 - sigfrac);
            }
            else if (iMM==2){ // W    
                var[j] = DD[DD3[it]].W[sm][ele0]*sigfrac  + DD[DD3[it]].W[sp][ele0]* (1.0 - sigfrac);
            }
            else { printf(" \n\n Bad iMM in Interpolatesigma %d\n\n",iMM); }
        }
        // Have sigma average var[0:2] at the three corners
        //interpolate to center, to get three time increments vart[0:2] 
        //float factor0=PP[Ip].factor4[iMM][0];
        //float factor1=PP[Ip].factor4[iMM][1];
        //float factor2=PP[Ip].factor4[iMM][2];
        vart[it]= PP[Ip].factor4[iMM][0]*var[0] 
        + PP[Ip].factor4[iMM][1]*var[1]
        + PP[Ip].factor4[iMM][2]*var[2];
        
        if (Ip==IpTest ) printf("  intersig DD3=%d var=%g %g %g vart=%g\n "
             ,DD3[it],var[0],var[1],var[2],vart[it]);
    }
    // Finally interpolate in time to get final answer for U, V, W to mover PP[Ip]
    // float time_now = DD[0].time_now;    // Will use dev_DD after the first pass with new DD
    float time_frac=(time_now - DD[DDT0].time)/(DD[DDT2].time - DD[DDT0].time);
    
    //float a =  2.*vart[2] -4.*vart[1] +2.*vart[0];
    //float b = -   vart[2] +4.*vart[1] +   vart[0];
    //float c =                       vart[0];
    //float Upnow = a*time_frac*time_frac + b*time_frac + c;
float Upnow = ( 2.*vart[2] -4.*vart[1] +2.*vart[0])*time_frac*time_frac 
             +(-   vart[2] +4.*vart[1] +   vart[0])*time_frac 
             +(                      vart[0]);                



/*  Now have time sigma and space interpolates of U,V,W for particle */
PP[Ip].answer[0] = Upnow;
        if (Ip==IpTest ) printf("  intersigend timenow=%fs timefrac=%g Upnow=%g\n ",time_now,time_frac,Upnow);


}