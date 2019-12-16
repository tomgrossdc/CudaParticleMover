#!/bin/sh
# CronJob.sh
export WGETWORKDIR=/media/tom/MyBookAllLinux/NOSnetcdf/CRONTAB
echo " CRONJob.sh with wget_regular_test.py WGETWORKDIR = ", $WGETWORKDIR
cd /media/tom/MyBookAllLinux/NOSnetcdf/CRONTAB
python3 wget_cbofs_fields_tomMy.py >> /media/tom/MyBookAllLinux/NOSnetcdf/CRONTAB/junkF.log
date
echo "THE END"

