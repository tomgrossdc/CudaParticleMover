//test.
#ifndef MESH_H
#define MESH_H

#include "main.h"
#include "string.h"

void ReadMesh(string& filename, struct MMesh *MM);
void AddOutsideLonLat(struct MMesh *MM);

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
Mesh(const char* file);

/*--------------------------------------------------------------------
    // Function      : set_Mesh
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : Fill up all empty arrays by reading data from netCDF file.
    //                 Only fills the time independent arrays, longitude, latitude, bathy, eles etc...
-----------------------------------------------------------------------*/
void set_Mesh(int node, int nsigma);
void set_Mesh_MMESH(struct MMesh *MM);
/*--------------------------------------------------------------------
    // Function      : set_float_field
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : Fill up time dependent members by reading netCDF file.
-----------------------------------------------------------------------*/
void set_float_field();

/*--------------------------------------------------------------------
    // Function      : read_floatfield
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : float time_now
    //
    // Return values : void
    //
    // Description   : Read in the time variable arrays U,V,W,T,S etc.
    //                 Toggle U_pp, U_past, U_future so they surround Time
----------------------------------------------------------------------*/
void read_floatfield(float time_now);

/*--------------------------------------------------------------------
    // Function      : find_2D_array
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : int time_index, char t_name
    //
    // Return values : 2D array (U, V)
    //
    // Description   : Read in the U V according to time slice.
----------------------------------------------------------------------*/
float** find_2D_array(int time_index, char t_name);

/*--------------------------------------------------------------------
    // Function      : test_report
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : build a report as testing purpose.
----------------------------------------------------------------------*/
void test_report();

/*--------------------------------------------------------------------
    // Function      : Set_Center
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : Determine lat_lon_key0 and lat_lon_key1
----------------------------------------------------------------------*/
void Set_Center();

/*--------------------------------------------------------------------
    // Function      : Deg2Meter
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : Translate lat lon from degree to meter for calculate
----------------------------------------------------------------------*/
void Deg2Meter();

/*--------------------------------------------------------------------
    // Function      : Deg2Meter
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : Translate lat lon from meter to degree for storage
----------------------------------------------------------------------*/
void Meter2Deg();

/*--------------------------------------------------------------------
    // Function      : printout_Mesh
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : Dump to screen diagnostic output of fixed arrays
----------------------------------------------------------------------*/
void printout_Mesh();

/*--------------------------------------------------------------------
    // Function      : print_floatfield
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : Dump to screen diagnostic output of Time dependent arrays
----------------------------------------------------------------------*/
void print_floatfield();

/*--------------------------------------------------------------------
    // Function      : ele_func_tripart
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : Use triangle class to set uo tri-connection in mesh.
----------------------------------------------------------------------*/
void ele_func_tripart();

/*--------------------------------------------------------------------
    // Function      : ele_func
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : void
    //
    // Description   : Calculate a b c factors
----------------------------------------------------------------------*/
void ele_func();

/*--------------------------------------------------------------------
    // Function      : find_ele
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : float x, float y, int i_ele
    //
    // Return values : int (iele)
    //
    // Description   : Find ele by its initial position
----------------------------------------------------------------------*/
int find_ele(float x, float y, int i_ele);

/*--------------------------------------------------------------------
    // Function      : get_factor
    //
    // Author        : Hong Lin
    //
    // Creation date : Nov 04, 2004
    //
    // Parameters    : int
    //
    // Return values : double
    //
    // Description   : Get current factors of the triangle.
----------------------------------------------------------------------*/
double get_factor(int index);

/*--------------------------------------------------------------------
    // Function      : get_ele
    //
    // Author        : Hong Lin
    //
    // Creation date : Nov 04, 2004
    //
    // Parameters    : int ele_index, int corner_index
    //
    // Return values : int
    //
    // Description   : Get current ele of the triangle.
----------------------------------------------------------------------*/
int get_ele(int ele_index, int corner_index);

/*--------------------------------------------------------------------
    // Function      : nc_error
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : int err_id
    //
    // Return values : void
    //
    // Description   : Output Netcdf error messages to standard I/O.
----------------------------------------------------------------------*/
void nc_error(int err_id);

/*--------------------------------------------------------------------
    // Function      : destructor
    //
    // Author        : Hong Lin
    //
    // Creation date : July 01, 2004
    //
    // Parameters    : none
    //
    // Return values : none
    //
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
