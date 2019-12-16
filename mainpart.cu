/*-------------------------------------------------------------------

on line 166 of a loop over all particles ip:
 			P[ip].move(time_now, &mesh, &data);
The Particle::move looks pretty independent of details of mesh and data
so need to replace generators of mesh and data   mesh.cpp,  data.cpp
That should simplify it and get away from netcdf.


Delete all the chaff and get down to read mesh, read data loop on p
data:  U,V,W,S,T,K  U_pp, U_past, U_future   2D arrays, [node][sigma]
mesh: needs elements ele[ele_index][corner_index]
      make a fake mesh by specifying an array of nodes.
      instead of reading a netcdf file.
//
// C++ Interface: main
//
// Description: mainpart.cpp, main.h
//   The C++ program, mainpart.cpp will read the starting lon, lat, start_time,
//   end_time and the inputfilename. The geometry will be read off the netcdf file
//   and mesh parameters calculated and stored into RAM.  The main line program will
//   loop over the time variable.  The velocity fields will be read off the NetCDF
//   file only when necessary to update the time stepping. An inner loop will go over
//   all particles.  Parallelization of the run can be done at this point by simply
//   calculating groups of particles on multiple processors.  
//
//
// Author: Tom Gross 
// 	   Hong Lin , (C) 2004
//
// Copyright: See COPYING file that comes with this distribution
//
//
---------------------------------------------------------------------*/

#include "main.h"
/* done in main.h
*/
#include "simpleGL.h"

#ifndef SIMP_GLCU
#define SIMP_GLCU
#include "simpleGL.cu"
#endif

//#include "struct.h"
// try to add a global struct for reference to internal routines
// Stuct these at top of simpleGL.cu

//PPart *host_P;
//PPart *dev_P;
//MMesh *dev_MM;
//DData *dev_DD;


double dt_sec;



    int num_P = NUM_PARTICLES;
    int node = NODE; 
    int nsigma = NSIGMA;  
    float time_now;

int  main( void ) {

 


        
        /*----------------------------------------------------------------------
        // Read in all the time independent data and put them into MM struct.
        ----------------------------------------------------------------------*/  
        MMesh *MM;
        MM =  (MMesh *)malloc(1*sizeof(MMesh));
        std::vector<std::string> filename={
            "/home/tom/code/NOSfiles/nos.cbofs.regulargrid.20190529.t00z.n001.nc",
            "/home/tom/code/NOSfiles/nos.cbofs.regulargrid.20190529.t00z.n002.nc",
            "/home/tom/code/NOSfiles/nos.cbofs.regulargrid.20190529.t00z.n003.nc",
            "/home/tom/code/NOSfiles/nos.cbofs.regulargrid.20190529.t00z.n004.nc"
        };
        int year = 2019, month=12, day=7, hour= 18;  //, minute=5;  // 19/5/12/1/5
        tm today = {}; 
        today.tm_year =year-1900 ; 
        today.tm_mon = month-1;
        today.tm_mday = day;
        today.tm_hour = hour;
        time_t ToDay = mktime(&today);
        MM[0].ToDay = ToDay;     
        string newername = NetCDFfiledate(MM);
        ReadMesh(newername,MM);

        //ReadMesh(filename[0],MM);
        node = MM[0].node;
        nsigma = MM[0].nsigma;
        printf("After ReadMesh  MM[0].node = %d  node =%d \n",MM[0].node,node);
        
        /*----------------------------------------------------------- 
        // Create object mesh.
        // Initialize the object, then copy MM into mesh. 
        // mesh is used only to interface with triangulate i.e.  mesh.ele_func_tripart();
        -----------------------------------------------------------*/
        Mesh mesh;
        mesh.meshinit(node,nsigma);
        node = mesh.node;
        nsigma = mesh.nsigma;
         printf("Main mesh.node %ld \n",mesh.node);
        mesh.set_Mesh_MMESH(MM);
         printf("main  after set_Mesh_MMESH  mesh.node %ld  node %d  MM[0].node %d\n"
            ,mesh.node, node, MM[0].node);
   /*-------------------------------------------------------------
   // An important function to set up tri-connection in the mesh.
   // Include triangle.h and triangle.c 
   // from Jonathan Richard Shewchuk's Triangle Versions 1.3 and 1.4   
   // Thanks Jonathan Richard Shewchuk, ect. 
   // After calculation, save result in tri-connect and ele array.
   // Now the performance of code is very good.
   -------------------------------------------------------------*/
        mesh.ele_func_tripart();
   

   printf("main   finished triangulate\n");
   if (mesh.node>NODE || mesh.nele > NELE) printf("\n\n\n\nWARNING NODE OR NELE ARE TOO SMALL\n\n\n\n");
   printf("main  after triangulate  mesh.node %ld  node %d  MM[0].node %d \n",mesh.node, node, MM[0].node);

   printf("Copy the derived mesh.X,Y to  the MMesh Struct MM.X,Y   nele = %ld\n",mesh.nele);

   for (int i=0; i<MM[0].node; i++){
       MM[0].Lon[i]=mesh.Lon[i];
       MM[0].Lat[i]=mesh.Lat[i];	
       MM[0].X[i]=mesh.X[i];
       MM[0].Y[i]=mesh.Y[i];
       MM[0].depth[i]=mesh.depth[i];  
       if (i%(MM[0].node/15)==0) {
           printf(" MM[0].XY[%d] %g %g  MM[0].LonLat[%d]= %g %g \n",
           i, MM[0].X[i],MM[0].Y[i],i,MM[0].Lon[i],MM[0].Lat[i] );
       }
   }
   for(int i=(MM[0].node-8); i<MM[0].node; i++)
         printf(" MM[0].XY[%d] %g %g  MM[0].LonLat[%d]= %g %g \n",
           i, MM[0].X[i],MM[0].Y[i],i,MM[0].Lon[i],MM[0].Lat[i] );

   for (int i=0; i<MM[0].nsigma; i++){ MM[0].sigma[i]=mesh.sigma[i];}

   for (int j=0; j<3; j++){
       MM[0].factor[j] = mesh.factor[j];
       for (int i=0; i<mesh.nele; i++){
           MM[0].a_frac[i][j] = mesh.a_frac[i][j] ;
           MM[0].b_frac[i][j] = mesh.b_frac[i][j] ;
           MM[0].c_frac[i][j] = mesh.c_frac[i][j] ;
           MM[0].tri_connect[i][j] = mesh.tri_connect[i][j];
           MM[0].ele[i][j] = mesh.ele[i][j];
                   }
    }    
    
    MM[0].factor[0]=0.123;
    MM[0].factor[1]=0.246;
    MM[0].factor[2]=0.369;
    
    //MMesh *dev_MM;
    size_t MMSizeGeneral = sizeof(MMesh);
    cudaMalloc((void**)&dev_MM,MMSizeGeneral);
    cudaMemcpy(dev_MM,MM,MMSizeGeneral,cudaMemcpyHostToDevice);
    
    printf(" mainpart after MM[0]. =mesh. MM[0].node = %d \n",MM[0].node);
    
   //Hide time interval information from user. 
   //We guess 30 second may be good enough after the numeric testing.
   
    
    // all time units in seconds ;
    //time_end = time_start + dt_sec * num_time;  // number of time steps
    
    // Initialize Particles set size of arrays



/*----------------------------------------------------    
    // Start to calculate particle move.
    ----------------------------------------------------*/   
	



/*      
Section builds struct from the class definitions.  Easier to pass to cuda<<<>>>
PPart, MMesh, DData   structures contain the essentials after the above routines build them.
DData should be rebuilt into arrays   DD[0:2]  like the PPart PP[0:num_p]
*/
//printf("sizeof Particle P %ld \n",sizeof(Particle));
printf("sizeof Mesh mesh  %ld \n",sizeof(mesh));
printf("sizeof Data data  %ld \n",sizeof(int));
printf("sizeof PPart      %ld \n",sizeof(PPart));
printf("sizeof MMesh      %ld \n",sizeof(MMesh));
printf("sizeof DData      %ld \n",sizeof(DData));


printf("initialize the PPart Structs \n");

size_t PPSizeGeneral ;
PPSizeGeneral = sizeof(PPart)*num_P;
printf(" sizeof(PPart)*%d = %lu\n",num_P,PPSizeGeneral);

host_P = (PPart *)malloc(PPSizeGeneral);


PPartInit(host_P,MM,num_P);

cudaMemcpy(dev_MM,MM,MMSizeGeneral,cudaMemcpyHostToDevice);

//PPart *dev_P;
//size_t PPSizeGeneral = (sizeof(PPart)*num_P);
cudaMalloc((void**)&dev_P,PPSizeGeneral);
cudaMemcpy(dev_P,host_P,PPSizeGeneral,cudaMemcpyHostToDevice);
printf("after cudaMalloc for dev_P, PPSizeGeneral=%ld\n",PPSizeGeneral);


// No need to initialize DD.  ReadData will do that later followed by cudaMemcpy
printf("Four separate DData's for past present future and reading\n");
DData *DD;
DD =  (DData *)malloc(4*sizeof(DData));

//DData *dev_DD;
size_t DDSizeGeneral = sizeof(DData)*4;
cudaMalloc((void**)&dev_DD,DDSizeGeneral);
//cudaMemcpy(dev_DD,DD,DDSizeGeneral,cudaMemcpyHostToDevice);

//checkGpuMem();
float free_m,total_m,used_m;
size_t free_t,total_t;
cudaMemGetInfo(&free_t,&total_t);
free_m =(uint)free_t/1048576.0 ;
total_m=(uint)total_t/1048576.0;
used_m=total_m-free_m;
printf ( "  mem free %ld .... %f MB \n  mem total %ld....%f MB mem used %f MB\n"
,free_t,free_m,total_t,total_m,used_m);
/* 
*  end of cuda memory / structure  setups
*
*
*/


 //  TEST of move.cpp
 //MMesh *MMd ; 
 //MMd = (MMesh *)malloc(sizeof(MMesh)); 
 //MMd[0] = MM[0];
 DD[0].ToDay =MM[0].ToDay;
 // Test the NetCDF reading of the file
 cout<<endl;
 for(int ifour = 0; ifour<4; ifour++){
     // Zero out all Velocities to initialize
     for (int i=0; i<MM[0].node; i++){
        for(int isig=0; isig<MM[0].nsigma; isig++){
         DD[ifour].U[isig][i]=0.0;
         DD[ifour].V[isig][i]=0.0;
         DD[ifour].W[isig][i]=0.0;
        }}
    DD[0].ToDay +=3600;  // for hourly files
    string newername = NetCDFfiledate(DD);
    ReadDataRegNetCDF(newername,ifour,DD,MM);
    //ReadDataRegNetCDF(filename[ifour],ifour,DD,MM);
    printf("ReadDataRegNetCDF finished,  DD[%d].time=%g sec time=%g hr \n",ifour,DD[ifour].time,DD[ifour].time/3600.);
    int id = 50;
    printf("ReadDataRegNetCDF DD[%d].time %gs %ghr \n  DD[%d].UV[0][%d] %g %g \n  MM[0].X[10]=%g\n\n",
      ifour, DD[ifour].time,DD[ifour].time/3600,ifour,id,DD[ifour].U[0][id],DD[ifour].V[0][id],MM[0].X[10]);

    }
cout<<endl;
time_now = (DD[0].time + DD[1].time)/2.;   // timefrac = .25
for (int i=0; i<4; i++) DD[i].time_now = time_now;

for (int ip=0; ip<num_P; ip++) host_P[ip].time_now=time_now;

for (int i=0; i<4; i++) DD[0].DD3[i]=i;
cudaMemcpy(dev_DD,DD,DDSizeGeneral,cudaMemcpyHostToDevice);

 
int ip=NUM_PARTICLES / 2;
printf("Host_Move %d time_now=%ghr pid=%d  x,y,z= %g,%g, %g \n\n",
ip, host_P[ip].time_now/3600.,host_P[ip].p_id,host_P[ip].x_present,host_P[ip].y_present,host_P[ip].z_present);


printf("Try to launch mainsimpleGL from mainpart.cu\n");

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif
printf("mainpart.cu skipping mainsimpleGL%s starting...\n", sSDKsample);

GLmoveparticle(host_P,MM,DD);


 
 /*----------------------------------------------------
    // End of particle movement calculation.
    ----------------------------------------------------*/   
		printf("\n mainpart.cp END \n");
    cudaFree(dev_DD);
    cudaFree(dev_MM);
    cudaFree(dev_P);

    /*
    // output P arrays  rewrite for ascii
    Dump dump;
	dump.set_time(&P[0]);
	
	for(int i=0; i<num_P; i++) {
		dump.set_position(&P[i]);
	}
    dump.output(output_name);	 
    */      
    
    return 0;
} // end of main

