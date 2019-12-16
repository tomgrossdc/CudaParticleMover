#include "particle.h"

// Initialize PP* by declaring and malloc'ing in mainline before calling this:
void PPartInit(struct PPart *host_P, struct MMesh *MM, int num_P)
{


/*  box stuff  no longer needed with netcdf
int numrows=50;
int numcols = 50;
float dy = 20000./numrows;  // spacing between rows
float dx = 20000./numcols;  // spacing between cols
float dxip = 20000./(num_P/numcols); // spacing along a line
float dyip = 20000./(num_P/numrows); // spacing along a column
*/

float maxminXY[4];
maxminXY[0]= 100000000.;  // min X
maxminXY[1]=-100000000.;  // max X
maxminXY[2]= 100000000.;  // min Y
maxminXY[3]=-100000000.;  // max Y

int imesh = 0;
for (int ip=0; ip<num_P; ip++)
{   
        host_P[ip].num_P = num_P;
        host_P[ip].p_id = ip;
        host_P[ip].i_ele = 50;
        host_P[ip].x_present = MM[0].X[imesh]/1. + 0.00;  //2.1  1.0625 goes 6 hr
        host_P[ip].y_present = MM[0].Y[imesh]/1. + 0.00;  // 0.00001
        host_P[ip].z_present = MM[0].depth[imesh]*500.*0.0;
        //host_P[ip].z_present = MM[0].Y[imesh];
        host_P[ip].time_now= 0.0;   // time_start;   probably can get rid of this
        if (ip%(NUM_PARTICLES/10) == 0) 
        printf(" host_P[%d] %g %g %g\n", ip,host_P[ip].x_present,host_P[ip].y_present,host_P[ip].z_present);

        imesh+=59;
        imesh=imesh%MM[0].node;
        maxminXY[0]=min(maxminXY[0],MM[0].X[imesh]);
        maxminXY[1]=max(maxminXY[1],MM[0].X[imesh]);
        maxminXY[2]=min(maxminXY[2],MM[0].Y[imesh]);
        maxminXY[3]=max(maxminXY[3],MM[0].Y[imesh]);
};
//  This is a bad place to set Xbox.  Needs to be in Mesh
printf(" MINMAX  X %g %g    Y %g %g \n",maxminXY[0],maxminXY[1],maxminXY[2],maxminXY[3]);
   MM[0].Xbox[0]= maxminXY[0]+1000.; // lon,x min  moved a bit inside mins and max
   MM[0].Xbox[1]= maxminXY[1]-1000.;  // lon,x max
   MM[0].Xbox[2]= maxminXY[2]+1000.;  // lat,y min
   MM[0].Xbox[3]= maxminXY[3]-1000.;  // lat,y max

/* the problem is probably not on out of bounds points. 
  Move any particles near boundary inside a bit
*/
/*float shrinkage = 1.05;    // 1.5 works
for (int ip=0; ip<num_P; ip++){
        if (host_P[ip].x_present < MM[0].Xbox[0]/shrinkage) host_P[ip].x_present = MM[0].Xbox[0]/shrinkage;
        if (host_P[ip].x_present > MM[0].Xbox[1]/shrinkage) host_P[ip].x_present = MM[0].Xbox[1]/shrinkage;
        if (host_P[ip].y_present < MM[0].Xbox[2]/shrinkage) host_P[ip].y_present = MM[0].Xbox[2]/shrinkage;
        if (host_P[ip].y_present > MM[0].Xbox[3]/shrinkage) host_P[ip].y_present = MM[0].Xbox[3]/shrinkage;
}
*/
   /* 
for (int ip=0; ip<num_P; ip+=2)
{   
        host_P[ip].num_P = num_P;
        host_P[ip].p_id = ip;
        host_P[ip].i_ele = NELE/2;
        host_P[ip].x_present = dxip*(ip%(num_P/numrows))  -10000.;
        host_P[ip].y_present = dy*float(ip/(num_P/numcols))  -10000.;
        host_P[ip].z_present = 5.*(rand()/(float)RAND_MAX)  -2.5;

        host_P[ip].time_now=time_start;
        if (ip%(NUM_PARTICLES/10) == 0) 
        printf(" host_P[%d] %g %g %g\n", ip,host_P[ip].x_present,host_P[ip].y_present,host_P[ip].z_present);
}
for (int ip=1; ip<num_P; ip+=2)
{   
        host_P[ip].num_P = num_P;
        host_P[ip].p_id = ip;
        host_P[ip].i_ele = NELE/2;
        host_P[ip].y_present = dyip*(ip%(num_P/numcols))  -10000.;
        host_P[ip].x_present = dx*float(ip/(num_P/numrows))  -10000.;
        host_P[ip].z_present = 5.*(rand()/(float)RAND_MAX)  -2.5;

        host_P[ip].time_now=time_start;
        if (ip%(NUM_PARTICLES/10) == 0) 
        printf(" host_P[%d] %g %g %g\n", ip,host_P[ip].x_present,host_P[ip].y_present,host_P[ip].z_present);
}
*/
printf(" host_P10 %g %g %g\n", 
host_P[10].x_present,host_P[10].y_present,host_P[10].z_present);
printf(" host_P11 %g %g %g\n", 
host_P[11].x_present,host_P[11].y_present,host_P[11].z_present);



}