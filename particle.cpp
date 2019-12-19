#include "particle.h"

// Initialize PP* by declaring and malloc'ing in mainline before calling this:
void PPartInit(struct PPart *host_P, struct MMesh *MM, int num_P)
{

float yrow;
int jrow;
int numrows=100;
for (int ip=0; ip<num_P; ip++)
{   
        jrow = ip/(num_P/numrows) ;
        yrow = jrow * 50000./numrows;
        host_P[ip].num_P = num_P;
        host_P[ip].p_id = ip;
        host_P[ip].i_ele = NELE/2;
        host_P[ip].y_present = yrow;
        host_P[ip].x_present = 10000.*(rand()/(float)RAND_MAX)  -35000.;
        host_P[ip].z_present = 5.*(rand()/(float)RAND_MAX)  -2.5;

        host_P[ip].time_now=0.0;
        if (ip%(NUM_PARTICLES/10) == 0) 
        printf(" host_P[%d] %g %g %g\n", ip,host_P[ip].x_present,host_P[ip].y_present,host_P[ip].z_present);
}

int imesh =0;     //  99827  108049;
for (int ip=0; ip<num_P; ip++) {
        host_P[ip].x_present = MM[0].X[imesh] + 1000.*(rand()/(float)RAND_MAX - 0.5);  
        host_P[ip].y_present = MM[0].Y[imesh] + 1000.*(rand()/(float)RAND_MAX - 0.5);  
        host_P[ip].z_present = MM[0].depth[imesh]*500.*0.0;
imesh+=1; imesh=imesh%MM[0].node;
}


printf(" host_P10 %g %g %g\n", 
host_P[10].x_present,host_P[10].y_present,host_P[10].z_present);
printf(" host_P11 %g %g %g\n", 
host_P[11].x_present,host_P[11].y_present,host_P[11].z_present);
}




