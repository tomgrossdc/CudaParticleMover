//test.
#ifndef MESH_H
#define MESH_H

#include "main.h"
#include "string.h"

void ReadMesh(string& filename, struct MMesh *MM);
void AddOutsideLonLat(int iMM, struct MMesh *MM);

class Mesh {
public:

/*--------------------------------------------------------------------
    // Function      : default Mesh Constructor
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : none
    //
    // Description   : Constructs a new Mesh object initialized to 0s
-----------------------------------------------------------------------*/
Mesh();
void meshinit(int node, int nsigma);
/*--------------------------------------------------------------------
    // Function      : Mesh Constructor with two parameters
    /
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : const char *file --- a file name of a netcdf file
    //                 FileMode mode --- indicate file accessability.
    //
    // Return values : none
    //
    // Description   : Open the Netcdf file, read all dimensions and allocate momory for arrays.
-----------------------------------------------------------------------*/

//void set_Mesh(int node, int nsigma);
void set_Mesh_MMESH(int icase, struct MMesh *MM);
void move_meshtoMMesh(int icase, struct MMesh *MM);




/*--------------------------------------------------------------------
    // Function      : Set_Center
    // Description   : Determine lat_lon_key0 and lat_lon_key1
----------------------------------------------------------------------*/
void Set_Center();

/*--------------------------------------------------------------------
    // Function      : Deg2Meter
    // Description   : Translate lat lon from degree to meter for calculate
----------------------------------------------------------------------*/
void Deg2Meter();

/*--------------------------------------------------------------------
    // Function      : Meter2Deg
    // Description   : Translate lat lon from meter to degree for storage
----------------------------------------------------------------------*/
void Meter2Deg();

/*--------------------------------------------------------------------
    // Function      : ele_func_tripart
        ------------------------------------------------------------
        // An important function to set up tri-connection in the mesh.
        // Include triangle.h and triangle.c 
        // from Jonathan Richard Shewchuk's Triangle Versions 1.3 and 1.4   
        // Thanks Jonathan Richard Shewchuk, ect. 
        // After calculation, save result in tri-connect and ele array.
        // Now the performance of code is very good.
        -------------------------------------------------------------
    // Description   : Use triangle class to set uo tri-connection in mesh.
----------------------------------------------------------------------*/
void ele_func_tripart();

/*--------------------------------------------------------------------
    // Function      : ele_func
    // Description   : Calculate a b c factors
----------------------------------------------------------------------*/
void ele_func();

/*--------------------------------------------------------------------
    // Function      : find_ele
    // Description   : Find ele by its initial position
----------------------------------------------------------------------*/
int find_ele(float x, float y, int i_ele);

/*--------------------------------------------------------------------
    // Function      : get_factor
    // Description   : Get current factors of the triangle.
----------------------------------------------------------------------*/
double get_factor(int index);

/*--------------------------------------------------------------------
    // Function      : get_ele
    // Return values : int
    // Description   : Get current ele of the triangle.
----------------------------------------------------------------------*/
int get_ele(int ele_index, int corner_index);

/*--------------------------------------------------------------------
    // Function      : nc_error
    // Description   : Output Netcdf error messages to standard I/O.
----------------------------------------------------------------------*/
void nc_error(int err_id);

/*--------------------------------------------------------------------
    // Function      : destructor
    // Description   : Free the data object memory for further usage.
----------------------------------------------------------------------*/
~Mesh();

public: 

	const char* path;		//Netcdf file name.
	
	int ncid; 			//NetCDF file ID.	
    	int status;			//Netcdf related error id.
    	long int nele;			//Number of ele
    	long int nbnd;			//Number of bnd
    	long int nface;			//Number of faces
    	long int nbi;			//Number of bi
 
    	long **ele, **bnd;		
    	long **tri_connect;		
    	float *X;			//X position (meter)
	float *Y;			//Y position (meter)
	float *Lat;			//Latitude array(degree)
	float *Lon;			//Longitude array(degree)
	float *depth;			//Depth array (degree)
    	float **a_frac;				
	float **b_frac;				
	float **c_frac;				
	float *factor;			//factor array.
    	double LON_LAT_KEY_0, LON_LAT_KEY_1;	//Center of latitude and longitude.
    	
	//Accessible to data.
    	long int nsigma;		//Number of sigma.
	long int node;			//Number of node.
    	float *sigma;			//sigma array.
};


#endif
