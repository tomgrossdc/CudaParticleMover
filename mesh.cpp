/*   Alter this by eliminating netcdf from set_Mesh 
    include reading a simple text file for lon, lat, depth, sigma
*/


#include "mesh.h"

/*--------------------------------------------------------------
    Class : Mesh
    Author : Tom Gross, Hong Lin
    Contact info : tom.gross@noaa.gov  hong.lin@noaa.gov
    Superclass : none
    Subclass : none    
    Required files : mesh.h
    Description : Read in the mesh data
       elefunc
       Create the triangle connectivity array  triconnect
       and create the a,b,c for the linear interpolations
       Transform from meters to degrees and back
--------------------------------------------------------------*/    

/*--------------------------------------------------------------------
   All temporary variables will start with t_, such as t_ele.
   All temporary index will start with i_, such as i_time_future. 
----------------------------------------------------------------------*/

Mesh::Mesh()
    {
    // in the future add a passed file to be read for the mesh data
//printf("mesh.cpp  node=%ld  nsigma=%ld\n",node,nsigma);
		
		node = NODE;  // number of grid points
		nsigma = NSIGMA;  // number of sigma points
	
	
	
	/*
	cout<<"Mesh() number of grid nodes      node= "<<node<<endl;
	cout<<"Mesh() number of sigma levels  nsigma= "<<nsigma<<endl;


  	LON_LAT_KEY_0=0.0;
 	LON_LAT_KEY_1=0.0;
	
	//Allocate dynamic momory space. 
	//Free all the momery in the destructor.
	Lon = new float[node];	
	Lat = new float [node];
	X = new float[node];
	Y = new float [node];
	depth = new float[node];
	sigma = new float[nsigma];	
	
	factor = new float[3];
	*/			

}  

//  Primary constructor of lower case mesh structure er, class
void Mesh::meshinit(int nodel, int nsigmal) {
    // in the future add a passed file to be read for the mesh data
printf("mesh::meshinit(int,int) old values: node=%ld  nsigma=%ld \n",node,nsigma);
		
		//node = 101;  // number of grid points
		//nsigma = 11;  // number of sigma points
	
	node = nodel;
	nsigma = nsigmal;
	
	cout<<"REally new number of grid nodes      node= "<<node<<endl;
	cout<<"number of new sigma levels  nsigma= "<<nsigma<<endl;


  	LON_LAT_KEY_0=0.0;
 	LON_LAT_KEY_1=0.0;
	
	//Allocate dynamic momory space. 
	//Free all the momery in the destructor.
	Lon = new float[node];	
	Lat = new float [node];
	X = new float[node];
	Y = new float [node];
	depth = new float[node];
	sigma = new float[nsigma];	
	
	factor = new float[3];
		
}

//Create a box for the mesh
void Mesh::set_Mesh(int node, int nsigma){
// Set variables  Lon, Lat, depth, sigma
printf("mesh.cpp set_Mesh node=%d  nsigma=%d\n",node,nsigma);

// The arrays were initialized in Mesh() as empty arrays length[node]
	int ix,iy;
   float x,y; 
	float xmin=-75.;
	float xmax=-70.;
	float ymin = 36.;
	float ymax = 38.;
   int isdegrees=1;

	xmin = -60000.37;
	xmax = 60000.89;
	ymin = -60000.44;
	ymax = 60000.98;
	isdegrees = 0;
	

    float dx = (xmax-xmin)/32.;
	float dy = (ymax-ymin)/32.;
	int nrows = 32;

	for (int i=0; i<node; i++) {
		iy = i/(nrows);
		ix = i-iy*nrows;
		Lon[i] =  ix*dx +xmin; 
		Lat[i] =  iy*dy +ymin;
		depth[i]= 10.;
      //printf(" Lon[%d] %f %f %f \n",i,Lon[i],Lat[i],depth[i]);

	}
	for (int i=0; i<nsigma; i++){
		sigma[i]=float(i)/float(nsigma-1);
	}
	
	
   if (isdegrees==1){
	Set_Center();
	//After read in Lon[i] and Lat[0:node-1], change them from degree to meter and store in X,Y array.
	Deg2Meter();	
	}
   else {
      for (int i=0;i<node;i++) {
   	   X[i] = Lon[i];
  	   Y[i] = Lat[i];
      }
   }
   
   return;		
}

//Read lon, lat, depth and sigma from struct MMesh read from the Netcdf file.
void Mesh::set_Mesh_MMESH(struct MMesh *MM){
// Set variables  mesh.Lon, Lat, depth, sigma from MM.Lon etc.
cout<<" Mesh::set_Mesh_MMESH MM.node="<<MM[0].node<<endl;
printf("mesh.cpp set_Mesh_MMesh MM.node=%d  MM.nsigma=%d\n",MM[0].node,MM[0].nsigma);

// The arrays were initialized in Mesh() as empty arrays length[MM.node]

	int isdegrees = 1;
	
	for (int i=0; i<MM[0].node; i++) {

		Lon[i] =  MM[0].Lon[i]; 
		Lat[i] =  MM[0].Lat[i];
		depth[i]= MM[0].depth[i];

	}
	for (int i=0; i<MM[0].nsigma; i++){
		sigma[i]=MM[0].sigma[i];
	}
	
	
   if (isdegrees==1){
	Set_Center();
	//After read in Lon[i] and Lat[0:node-1], change them from degree to meter and store in X,Y array.
	Deg2Meter();	
	}
   else {
      for (int i=0;i<MM[0].node;i++) {
   	   X[i] = Lon[i];
  	   Y[i] = Lat[i];
      }
   }
   
   return;		
}

void Mesh::Set_Center() {
  printf(" Set_Center %ld \n",node);

 for (int i=0;i<node;i++) {
 	LON_LAT_KEY_0 =LON_LAT_KEY_0+ Lon[i]/node;
 	LON_LAT_KEY_1 =LON_LAT_KEY_1+ Lat[i]/node;   
 }
   printf("LONLATKEY %g %g \n",LON_LAT_KEY_0,LON_LAT_KEY_1);

 return;
} //End of Set_Center().


//  Deg2Meter converts longitude latitude to Meters for full mesh array.
void Mesh::Deg2Meter() {
 	  printf(" Deg2Meter %ld \n",node);
float DEG_PER_METER= 90./(10000*1000);
 for (int i=0;i<node;i++) {
  	X[i] = (( Lon[i]-LON_LAT_KEY_0) /DEG_PER_METER )*cos(LON_LAT_KEY_1 * PI/180.);
  	Y[i] =  ( Lat[i]-LON_LAT_KEY_1) /DEG_PER_METER;
 }
   return;
}   //End of Deg2Meter().


void Mesh::ele_func_tripart() {

   struct triangulateio in, mid;
   
   in.numberofpoints = node;
   in.numberofpointattributes = 0;
   in.pointlist = (REAL *) malloc(in.numberofpoints * 2 * sizeof(REAL));
   for(int i=0; i<node; i=i+1) {
      	in.pointlist[2*i] = X[i];
   	in.pointlist[2*i+1] = Y[i];
   }
   in.pointmarkerlist = (int *) NULL;
   
   in.numberofsegments = 0;
   in.numberofholes = 0;
   in.numberofregions = 1;
   in.regionlist = (REAL *) NULL;   
   
   mid.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of point attributes is zero: */
   mid.pointattributelist = (REAL *) NULL;
   mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
   mid.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
   mid.triangleattributelist = (REAL *) NULL;
   mid.neighborlist = (int *) NULL;         /* Needed only if -n switch used. */  
    
   cout << "just before triangulate "<<endl; 
   //char *znstr;
   //char *znstr = "zn";
   // = "zn";
   char znstr[]="zn";
   triangulate(znstr, &in, &mid, NULL); 
   cout << "just after triangulate"<<endl;

   tri_connect = new long* [mid.numberoftriangles];
   ele = new long* [mid.numberoftriangles];
   for(int i=0; i<mid.numberoftriangles; i++) {
	tri_connect[i] = new long [nface];
	ele[i] = new long [nface];
   }   
   
   nele=mid.numberoftriangles;
   a_frac = new float* [nele];
   b_frac = new float* [nele];	
   c_frac = new float* [nele];	
   for(int i=0; i<nele; i++) {
	a_frac[i] = new float[nface];
	b_frac[i] = new float[nface];		
	c_frac[i] = new float[nface];		
   }
   cout<<"numberoftriangles = "<<mid.numberoftriangles<<endl;	
   
   for(int i=0; i<mid.numberoftriangles; i++) {
   	for(int j=0; j<3; j++) {
   		tri_connect[i][j]=mid.neighborlist[i*3+j];
		ele[i][j]=mid.trianglelist[i*3+j];
	}
   }
   
     /*-----------------------------------------------------------
    // Calculate a, b, c factors
    -----------------------------------------------------------*/	
   ele_func();	
   return;
}

//inputs are ele, X, Y; Outputs are triconnect,a,b,c
void Mesh::ele_func() {
  int i,k; 
  float x3[3], y3[3], xo ,yo;
  float d;
//printf("start ele_func\n");

  for (i=0; i<nele; i++) {  
   	for (k=0;k<3;k++) {
      		x3[k] = X[ele[i][k]] ;
      		y3[k] = Y[ele[i][k]] ;
   	}

	// Determinate [x3 y3 ones(3,1)] 
  	d= x3[0]*y3[1] +x3[1]*y3[2] +x3[2]*y3[0] -x3[0]*y3[2] -x3[1]*y3[0] -x3[2]*y3[1] ; 
	
	//  Calculate a,b,c for every triangle
    	for( k=0; k<3; k++) {
         	a_frac[i][k] = (y3[1]-y3[2])/d;
        	b_frac[i][k] = (x3[2]-x3[1])/d;
        	c_frac[i][k] = (x3[1]*y3[2]-x3[2]*y3[1])/d;
		//   rotate the x,y for next function
	 	xo=x3[0]; 
		x3[0]=x3[1]; 
		x3[1]=x3[2]; 
		x3[2]=xo;
	 	yo=y3[0]; 
		y3[0]=y3[1]; 
		y3[1]=y3[2]; 
		y3[2]=yo;
    	} 
  }
  printf("mesh.cpp finished ele_func\n");

   return;
} 
//   end of ele_func


/*-------------------------------------------------------------
//  find_ele   Locates the element which holds xpart,ypart
//  Finds the ele where is ax+by+c >0.0 for all three corners
//  also returns the factor[3] array of weights
------------------------------------------------------------*/ 
int Mesh::find_ele(float xpart, float ypart, int i_ele) {  

  int keepgoing=1;
  int k;
  float smallest_value = -0.001; 
  
     //  Check for out of domain particle
     if (i_ele < 0) { 
     	factor[0]=0.0; factor[1]=0.0; factor[2]=0.0;
      	return i_ele;
     }

 while (keepgoing  == 1){
//  if any of the f's are negative, walk that way and restart while loop
   k=0;
   factor[k]=a_frac[i_ele][k]*xpart + b_frac[i_ele][k]*ypart + c_frac[i_ele][k];
   if ( factor[k] < smallest_value) { 
   	i_ele = tri_connect[i_ele][0]; 
   }
   else { 
      k=1;
      factor[k]=a_frac[i_ele][k]*xpart + b_frac[i_ele][k]*ypart + c_frac[i_ele][k];
      if ( factor[k] < smallest_value ) { 
      	  i_ele = tri_connect[i_ele][1] ; 
      }
      else { 
          k=2;
          factor[k]=a_frac[i_ele][k]*xpart + b_frac[i_ele][k]*ypart + c_frac[i_ele][k];
          if ( factor[k] < smallest_value ) { 
	  	i_ele = tri_connect[i_ele][2] ;
          }
          else {
             //  Found it, iele,   all f's are positive 
             keepgoing = 0;
	  }
      }
   }
   if (i_ele < 0) { 
     	factor[0]=0.0; factor[1]=0.0; factor[2]=0.0;
     	keepgoing = 0;
   }
 }
 
 return i_ele;
}

//Read out 3 factors one by one.
double Mesh::get_factor(int index) {

    return factor[index];
}

//Read out current ele.
int Mesh::get_ele(int ele_index, int corner_index) {

    return ele[ele_index][corner_index];
}


// Destructor, free momery.
Mesh::~Mesh() {
	
	/*delete Lon;	
	delete X;
	delete depth;
	delete sigma;
	delete factor;	
	delete Y;			
	delete Lat;				
	delete [] a_frac;	
	delete [] b_frac;
	delete [] c_frac;	
 	delete [] tri_connect;	
	delete [] ele;
	*/
//cout<<"mesh.cpp Mesh Destructor called"<<endl;	
}



void ReadMesh(string& filename, struct MMesh *MM)
{


NcDim dim;
long ij;
int nx,ny;
int node,nodemore;

cout<<"ReadMesh: " << filename << endl;
NcFile dataFile(filename, NcFile::read);

   NcVar data=dataFile.getVar("mask");
   if(data.isNull()) printf(" data.isNull Mask/n");
for (int i=0; i<2; i++){
      dim = data.getDim(i);
      size_t dimi = dim.getSize();
      if (i==0) ny = dimi;
      if (i==1) nx = dimi;

      cout<<"ReadMesh  i= "<<i<<" dimi= "<<dimi<<endl;
   }
double Mask[ny][nx];
double LLL[ny][nx];
//double LLat[ny][nx];
//std::vector<std::vector<double> > LLL( ny , std::vector<double> (nx));  
//std::vector<std::vector<double> > Mask( ny , std::vector<double> (nx));  


printf("sizeof LLL  %ld\n",sizeof(LLL));

printf("sizeof Mask %ld\n",sizeof(Mask));

   data.getVar(Mask);
   ij=0;
   for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
    { MM[0].Mask[ij] = Mask[i][j]; ij++; }  }

   data=dataFile.getVar("Longitude");
   if(data.isNull()) printf(" data.isNull Longitude/n");
   data.getVar(LLL);
    ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[0].Lon[ij] = LLL[i][j]; ij++;} }  }


 node = ij;
 int summask=0;
   for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++){
       if((i-1)>=0) {
                        summask += Mask[i-1][j];
         if((j+1)<node) summask += Mask[i-1][j+1];
         if((j-1)>=0  ) summask += Mask[i-1][j-1];
       }
       if((i+1)<node) {
                        summask += Mask[i+1][j];
         if((j+1)<node) summask += Mask[i+1][j+1];
         if((j-1)>=0  ) summask += Mask[i+1][j-1];
       }
       if((j+1)<node)   summask += Mask[i][j+1];
       if((j-1)>=0  )   summask += Mask[i][j-1];
      
       if (Mask[i][j]==0 && summask > 0 ){
 //         MM[0].Lat[node] = LLat[i][j]; 
          MM[0].Lon[node] = LLL[i][j];
          MM[0].depth[node]=0.0;
          node++;}  
          summask=0; 
      } }  // i,j loop

  data=dataFile.getVar("Latitude");
  if(data.isNull()) printf(" data.isNull Latitude/n");
  data.getVar(LLL);
    ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[0].Lat[ij] = LLL[i][j]; ij++;} }  }

 node = ij;
 MM[0].firstnodeborder=node;  // initialize the first border node

 summask=0;
   for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++){
       if((i-1)>=0) {
                        summask += Mask[i-1][j];
         if((j+1)<node) summask += Mask[i-1][j+1];
         if((j-1)>=0  ) summask += Mask[i-1][j-1];
       }
       if((i+1)<node) {
                        summask += Mask[i+1][j];
         if((j+1)<node) summask += Mask[i+1][j+1];
         if((j-1)>=0  ) summask += Mask[i+1][j-1];
       }
       if((j+1)<node)   summask += Mask[i][j+1];
       if((j-1)>=0  )   summask += Mask[i][j-1];
      
       if (Mask[i][j]==0 && summask > 0 ){
 //         MM[0].Lat[node] = LLat[i][j]; 
          MM[0].Lat[node] = LLL[i][j];
          MM[0].depth[node]=0.0;
          node++;}  
          summask=0; 
      } }  // i,j loop


   data=dataFile.getVar("h");
   if(data.isNull()) printf(" data.isNull h/n");
   data.getVar(LLL);
    ij=0; for (int i=0; i<(ny); i++){for (int j=0; j<(nx); j++)
       {if (Mask[i][j]>.5){MM[0].depth[ij] = LLL[i][j]; ij++;} }  }


    MM[0].node = node;
    printf(" masked Lon, Lat num ij = %ld\n",ij);


   /* Although labeled sigma, in Regulargrid this is Depth
   Apparently the regular grid gets rid of sigma variations
   and just uses fixed depths for the 3d
   */
   data=dataFile.getVar("Depth");
   if(data.isNull()) printf(" data.isNull Depth/n");
   dim = data.getDim(0);
   size_t dimi = dim.getSize();
   MM[0].nsigma=dimi;
   cout<<" ReadMesh Depth dimi="<<dimi <<endl;
   double sigma[dimi];
   data.getVar(sigma);
   for (int i=0; i<dimi; i++) MM[0].sigma[i]=sigma[i];



// not needed as LL Mask will be deleted on exit from this routine
//delete [] LLL;
//delete [] Mask;

dataFile.close();
printf(" ReadMesh                  node = %d\n", MM[0].node);
AddOutsideLonLat(MM);
printf(" ReadMesh after AddOutside node = %d\n", MM[0].node);

}

void AddOutsideLonLat(struct MMesh *MM){
int nodemore;

// Xbox will be changed to meters in Particle initialize
MM[0].Xbox[0]= -78.;    // Lon min
MM[0].Xbox[1]=  -73.;    //Lon max
MM[0].Xbox[2]= 36.;    // Lat min
MM[0].Xbox[3]=  41.;    // Lat max

nodemore = MM[0].node;

for (int i=0; i<2; i++){
   for (int j=0; j<2; j++){
      MM[0].Lon[nodemore] = MM[0].Xbox[i];
      MM[0].Lat[nodemore] = MM[0].Xbox[j+2];
      MM[0].depth[nodemore]= 5.;
      nodemore++;
   }
}
   MM[0].Lon[nodemore] = (MM[0].Xbox[0]+MM[0].Xbox[1] )/2.;  // 01 12 23 34
   MM[0].Lat[nodemore] = MM[0].Xbox[2];  // 01 12 23 34
   MM[0].depth[nodemore]= 5.;
   nodemore++;
   MM[0].Lon[nodemore] = (MM[0].Xbox[0]+MM[0].Xbox[1] )/2.;  // 01 12 23 34
   MM[0].Lat[nodemore] = MM[0].Xbox[3];  // 01 12 23 34
   MM[0].depth[nodemore]= 5.;
   nodemore++;
   MM[0].Lon[nodemore] = MM[0].Xbox[0];  // 01 12 23 34
   MM[0].Lat[nodemore] = (MM[0].Xbox[2]+MM[0].Xbox[3] )/2.;  // 01 12 23 34
   MM[0].depth[nodemore]= 5.;
   nodemore++;
   MM[0].Lon[nodemore] = MM[0].Xbox[1];  // 01 12 23 34
   MM[0].Lat[nodemore] = (MM[0].Xbox[2]+MM[0].Xbox[3] )/2.;  // 01 12 23 34
   MM[0].depth[nodemore]= 5.;
   nodemore++;

MM[0].node = nodemore;


}

string NetCDFfiledate(struct MMesh *MM){
   // Same data is in DData[0].ToDay and MMesh[0].ToDay  
   // But they need separate calls to initialize
// create updated filename based on given root name Regular
// and the time seconds ToDay 
// Call from elsewhere after doing this in mainline:
/*
int year = 2019, month=5, day=12, hour= 0, minute=5;
tm today = {}; 
today.tm_year =year-1900 ; 
today.tm_mon = month-1;
today.tm_mday = day;
today.tm_hour = hour;
time_t ToDay = mktime(&today);
MM[0].ToDay = ToDay; 

// Then anywhere do:
    MM[0].ToDay +=3600;  // for hourly files
    string newername = NetCDFfiledate(MM);
*/
char buffer [125];
int n;

struct tm * tday = gmtime(&MM[0].ToDay);

    //ToDay +=3600;
    tday = gmtime(&MM[0].ToDay);
    //n = sprintf(buffer
    // ,"/home/tom/code/NOSfiles/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc"
    // ,tday->tm_year +1900, tday->tm_mon +1, tday->tm_mday,6*int(tday->tm_hour/6),1+(tday->tm_hour%6));
    n = sprintf(buffer
     ,"/media/tom/MyBookAllLinux/NOSnetcdf/%d%02d/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc"
     ,tday->tm_year +1900, tday->tm_mon +1
     ,tday->tm_year +1900, tday->tm_mon +1 
     ,tday->tm_mday,6*int(tday->tm_hour/6),1+(tday->tm_hour%6));

    string newname;
    for (int i=0; i<n; i++) newname=newname+buffer[i]; 

    return newname;
}

string NetCDFfiledate(struct DData *MM){
   // Same data is in DData[0].ToDay and MMesh[0].ToDay  
   // But they need separate calls to initialize
// create updated filename based on given root name Regular
// and the time seconds ToDay 
// Call from elsewhere after doing this in mainline:
/*
int year = 2019, month=5, day=12, hour= 0, minute=5;
tm today = {}; 
today.tm_year =year-1900 ; 
today.tm_mon = month-1;
today.tm_mday = day;
today.tm_hour = hour;
time_t ToDay = mktime(&today);
DD[0].ToDay = ToDay; 

// Then anywhere do:
    MM[0].ToDay +=3600;  // for hourly files
    string newername = NetCDFfiledate(MM);
*/
char buffer [125];
int n;

struct tm * tday = gmtime(&MM[0].ToDay);
//     ,"/home/tom/code/NOSfiles/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc"

    //ToDay +=3600;
    tday = gmtime(&MM[0].ToDay);
    n = sprintf(buffer
     ,"/media/tom/MyBookAllLinux/NOSnetcdf/%d%02d/nos.cbofs.regulargrid.%d%02d%02d.t%02dz.n%03d.nc"
     ,tday->tm_year +1900, tday->tm_mon +1
     ,tday->tm_year +1900, tday->tm_mon +1 
     ,tday->tm_mday,6*int(tday->tm_hour/6),1+(tday->tm_hour%6));

    string newname;
    for (int i=0; i<n; i++) newname=newname+buffer[i]; 

    return newname;
}