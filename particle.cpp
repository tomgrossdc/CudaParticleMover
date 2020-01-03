#include "particle.h"

// Initialize PP* by declaring and malloc'ing in mainline before calling this:
void PPartInit(struct PPart *host_P, struct MMesh *MM, int num_P)
{
enum class Layouts {BoxRows, FillBay, SewerPipe };
Layouts layout = Layouts::FillBay;

//case   box
switch(layout){
    case Layouts::BoxRows:{
        float yrow;
        int jrow;
        int numrows=100;
        for (int ip=0; ip<num_P; ip++)
        {   
        jrow = ip/(num_P/numrows) ;
        yrow = jrow * 150000./numrows;
        host_P[ip].num_P = num_P;
        host_P[ip].p_id = ip;
        host_P[ip].i_ele = NELE/2;
        host_P[ip].y_present = yrow + 1000.*(rand()/(float)RAND_MAX -.5);
        host_P[ip].x_present = 50000.*(rand()/(float)RAND_MAX -.5)  -25000.;
        host_P[ip].z_present = 500.*(rand()/(float)RAND_MAX)  -2.5;

        host_P[ip].time_now=0.0;
        if (ip%(NUM_PARTICLES/10) == 0) 
        printf(" host_P[%d] %g %g %g\n", ip,host_P[ip].x_present,host_P[ip].y_present,host_P[ip].z_present);
        }
    } break;
    case Layouts::FillBay:{
        //case  fill bay with a jiggle at nodal points
        printf(" MM[0].firstnodeborder = %d \n", MM[0].firstnodeborder);
        //MM[0].firstnodeborder=99827;
        int imesh =0;     //  99827  108049;
        for (int ip=0; ip<num_P; ip++) {
            host_P[ip].num_P = num_P;
            host_P[ip].p_id = ip;
            host_P[ip].i_ele = 55;
            for (int i=0; i<4;i++) host_P[ip].i_ele4[i] = 55;
                host_P[ip].x_present = MM[0].X[imesh] + 1000.*(rand()/(float)RAND_MAX - 0.5);  
                host_P[ip].y_present = MM[0].Y[imesh] + 1000.*(rand()/(float)RAND_MAX - 0.5);  
                host_P[ip].z_present = -2.0;    //MM[0].depth[imesh]/2.;
                //host_P[ip].z_present = MM[0].Y[imesh]/5.;
                imesh+=1; 
                imesh=imesh%MM[0].firstnodeborder;
        }
    } break;
    default: { printf("error particle case");}
}  // switch closure

//  Add the coastal border to last particle positions
int imesh =MM[0].firstnodeborder;     //  99827  108049;
for (int ip=num_P-(MM[0].node-MM[0].firstnodeborder); ip<num_P; ip++) {
        host_P[ip].x_present = MM[0].X[imesh] ;  
        host_P[ip].y_present = MM[0].Y[imesh] ;  
        host_P[ip].z_present = 0.0;
imesh+=1; 
if (imesh>=MM[0].node) imesh = MM[0].firstnodeborder;
}

printf(" host_P10 %g %g %g\n", 
host_P[10].x_present,host_P[10].y_present,host_P[10].z_present);
printf(" host_P11 %g %g %g\n", 
host_P[11].x_present,host_P[11].y_present,host_P[11].z_present);
}




