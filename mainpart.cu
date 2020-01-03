/*-------------------------------------------------------------------
Dec. 19, 2019  after refactor/  next objective is to read ROMS

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

 

    printf("mainpart.cu  Cuda based particle mover \n");
        
        /*----------------------------------------------------------------------
        // Read in all the time independent data and put them into MM struct.
        ----------------------------------------------------------------------*/  
        MMesh *MM;
        MM =  (MMesh *)malloc(4*sizeof(MMesh));
        
        /*filetemplates  examples:
            "/media/tom/MyBookAllLinux/NOSnetcdf/201912/nos.cbofs.regulargrid.20191207.t18dz.n006.nc",
            "/media/tom/MyBookAllLinux/NOSnetcdf/%d%02d/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc",
        */

        int year = 2019, month=12, day=7, hour= 18;  //, minute=5;  // 19/5/12/1/5
        tm today = {}; 
        today.tm_year =year-1900 ; 
        today.tm_mon = month-1;
        today.tm_mday = day;
        today.tm_hour = hour;
        time_t ToDay = mktime(&today);
        MM[0].ToDay = ToDay;     
        MM[0].filetemplate = 
        "/media/tom/MyBookAllLinux/NOSnetcdf/%d%02d/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc";
        MM[0].filetemplate = 
        "/media/tom/MyBookAllLinux/NOSnetcdf/%d%02d/nos.cbofs.fields.%d%02d%02d.t%02dz.n%03d.nc";
        string newername = NetCDFfiledate(MM[0].filetemplate,MM);
        int icase = 0;


        /*----------------------------------------------------------- 
        // Create object mesh.
        // Initialize the object, then copy MM into mesh. 
        // mesh is used only to interface with triangulate i.e.  mesh.ele_func_tripart();
        -----------------------------------------------------------*/

        Mesh mesh;

        //   icase-1 is used to include Regular call during testing.  Fix later...
        for (icase=1; icase<4; icase++){
            printf(" \n\n Read and Set MM[%d]\n",icase-1);
            ReadMeshField(newername,icase,MM);
            node = MM[icase-1].node;
            nsigma = MM[icase-1].nsigma;
            mesh.meshinit(node,nsigma);

            mesh.set_Mesh_MMESH(icase,MM);
            
            mesh.ele_func_tripart();
            
            mesh.move_meshtoMMesh(icase, MM);
        }
        printf("\nmain  after set_Mesh_MMESH  mesh.node %ld  node %d  MM[0].node %d\n"
        ,mesh.node, node, MM[0].node);
        int iMM=0; int i_ele = 50; 
        for (iMM=0; iMM<4; iMM++){for (int iP=0; iP<MM[iMM].node; iP+=5000)
            printf(" main Mesh set MM[%d].depth[%d] = %g  MM.ANGLE = %g\n",
            iMM,iP, MM[iMM].depth[iP], MM[iMM].ANGLE[iP]);
        }
        //MMesh *dev_MM;    // no need to do this. Space is cudaMalloc'd and call is to (struct MMesh dev_MM)
        size_t MMSizeGeneral = 4*sizeof(MMesh);
        cudaMalloc((void**)&dev_MM,MMSizeGeneral);
        cudaMemcpy(dev_MM,MM,MMSizeGeneral,cudaMemcpyHostToDevice);
                
        
        
        /*      
        Build the Particle Struct PPart host_P
        */
        
        printf("Initialize the PPart Structs \n");
        
        size_t PPSizeGeneral ;
        PPSizeGeneral = sizeof(PPart)*num_P;
        printf(" sizeof(PPart)*%d = %lu\n",num_P,PPSizeGeneral);
        
        host_P = (PPart *)malloc(PPSizeGeneral);
        
        
        //  Elaborate this routine for different inital conditions of particles
        //  Default, for now, is to duplicate the mesh with some random jiggle
        PPartInit(host_P,MM,num_P);
        
        //cudaMemcpy(dev_MM,MM,MMSizeGeneral,cudaMemcpyHostToDevice);
        
        //PPart *dev_P;
        //size_t PPSizeGeneral = (sizeof(PPart)*num_P);
        cudaMalloc((void**)&dev_P,PPSizeGeneral);
        cudaMemcpy(dev_P,host_P,PPSizeGeneral,cudaMemcpyHostToDevice);
        printf("after cudaMalloc for dev_P, PPSizeGeneral=%ld\n",PPSizeGeneral);
        
        
        // No need to initialize DD here.  ReadData will do that later followed by cudaMemcpy
        printf("Four separate DData's for past present future and reading\n");
        DData *DD;
        DD =  (DData *)malloc(4*sizeof(DData));
        DD[0].filetemplate = MM[0].filetemplate;
        
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
         */
        
        // Initialize some DD data, 
        // Zero out all Velocities to initialize
        //  Makes the coastline capture grounded particles, as it is not read as a velocity
        DD[0].ToDay =MM[0].ToDay;
        for(int ifour = 0; ifour<4; ifour++){
            for (int i=0; i<NODE; i++){
                for(int isig=0; isig<NSIGMA; isig++){
                    DD[ifour].U[isig][i]=0.0;
                    DD[ifour].V[isig][i]=0.0;
                    DD[ifour].W[isig][i]=0.0;
                }
            }
            DD[0].ToDay +=3600;  // for hourly files
            string newername = NetCDFfiledate(DD[0].filetemplate,DD);
            //ReadDataRegNetCDF(newername,ifour,DD,MM);
            ReadFieldNetCDF(newername,ifour, DD, MM);
            printf("ReadData finished,  DD[%d].time=%f sec time=%g hr \n",ifour,DD[ifour].time,DD[ifour].time/3600.);
            int id = 50; int isig=2;
            printf("ReadData DD[%d].time %fs %ghr \n  DD[%d].UVW[%d][%d] %g %g %g \n  MM[0].XY[%d]= %g  %g\n\n",
              ifour, DD[ifour].time,DD[ifour].time/3600,
              ifour,isig, id,DD[ifour].U[isig][id],DD[ifour].V[isig][id],DD[ifour].W[isig][id],
              id,MM[0].X[id],MM[0].Y[id]);
        }
    cout<<endl;
    time_now = (DD[0].time + DD[1].time)/2.;   // timefrac = .25
    for (int i=0; i<4; i++) DD[i].time_now = time_now;
    printf(" mainpart  DD[0].time_now = %f DD[0].time = %f \n",DD[0].time_now,DD[0].time);

    for (int ip=0; ip<num_P; ip++) host_P[ip].time_now=time_now;

    for (int i=0; i<4; i++) DD[0].DD3[i]=i;

    cudaMemcpy(dev_DD,DD,DDSizeGeneral,cudaMemcpyHostToDevice);


printf("/n/nLaunch GLmoveparticle from mainpart.cu\n");

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

GLmoveparticle(host_P,MM,DD);


 
 /*----------------------------------------------------
    // End of particle movement calculation.
    ----------------------------------------------------*/   
	printf("\n mainpart.cp END \n");
    cudaFree(dev_DD);
    cudaFree(dev_MM);
    cudaFree(dev_P);

    
    return 0;
} // end of main

