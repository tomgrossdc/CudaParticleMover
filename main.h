/***************************************************************************
 *   Copyright (C) 2004 by                                       	   *
 *  Tom Gross                                                              *
 *  Hong Lin                                                               *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

//Test how to check out.

#ifndef MAIN_H
#define MAIN_H

//  WARNING Change these do make clean
#define NODE 110000     // 16384 g 8150 8050 8000  x 8100 8200
#define NROWS 128      //  g 32
#define NSIGMA 15
#define NELE 220000    //  g 15800
#define NUM_PARTICLES 150000   //250000 65000dec4  1500625 g1625 1600 x 1650 1700 1800 2000
#define MAX_GLPARTICLES 150000 //250000 65000
#define DT_SEC 1.5
#define CUDA_STEPS 60
#define FILE_DTIME 3600.
#define SCALE_GL 0.00003  // .000015(full bay) scale PP.x_ to pos[Ipx][4]  


#include <iostream>
using namespace std;
#include <fstream>
#include <string.h>
#include <math.h>
#include <iomanip>
#include <time.h>
#include <thread> 

#include <netcdf>
using namespace netCDF;
using namespace netCDF::exceptions;


//#include "netcdfcpp.h"
//#include "netcdf.h"#include "date.h" 

#include "mesh.h"
#include "data.h"
#include "particle.h"
//#include "move.h"

//#include "dump_netcdf.h"

//#include "date.h"

//#include "Bugs.h"

#define PI 3.14159265358979

// The original code of triangle.h and triangle.c is written by C.
extern "C" {
	#include "triangle.h"
}

struct PPart { 
    int p_id;
    int i_ele; 
    float x_present; 
    float y_present; 
    float z_present; 
    float time_now;
    float factor[3];
    int num_P;};

struct MMesh {
    float Lon[NODE];
    float Lat[NODE];	
    float X[NODE];
    float Y[NODE];
    float Xbox[4];
	float depth[NODE];
	float sigma[NSIGMA];
	float factor[3];	
	float a_frac[NELE][3];	
	float b_frac[NELE][3];
	float c_frac[NELE][3];	
 	long tri_connect[NELE][3];	
	long ele[NELE][3];
    int node;
    int nsigma;
    float Mask[352737];
    time_t ToDay;
    int firstnodeborder=99827;
};

struct DData {
    double time;
    float U[NSIGMA][NODE];
    float V[NSIGMA][NODE];
    float W[NSIGMA][NODE];
    float temp[NSIGMA][NODE];
    float salt[NSIGMA][NODE];
    float Kh[NSIGMA][NODE];
    int DD3[4];
    time_t ToDay;
    float time_now;

};
void Cuda_Move();
void ReadMesh(string& filename, struct MMesh *MM);
string NetCDFfiledate(struct MMesh *MM);
string NetCDFfiledate(struct DData *MM);





#endif
