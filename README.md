# CudaParticleMover
Particle tracker using Nvidia Cuda cores and NetCDF files from NOAA/NOS

Building the code:
Use a computer with an Nvidia graphics card. 
Install Netcdf libraries
Install Nvidia libraries
Tailor Makefile to use those libraries
Tailor Cronjob.sh to download NetCDF files and write them to your own directory system. 

3D branch.  Add the vertical data to DData and Read

four meshes
Rework cuda move to do single variable at a time. Could work with Reg if it just uses the same MM[0] for all.
Of course 3d adds complexity...
