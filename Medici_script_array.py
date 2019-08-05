from subprocess import call
import time
import glob
import os
import sys

fnames_all =[]
folder=str(sys.argv[1])
fps=str(sys.argv[2])
comp=str(sys.argv[3])
base_folder='/QRISdata/Q0291/ForAnalysis/'+folder # folder containing the demo files
for file in glob.glob(os.path.join(base_folder,'*.tif')):
    if file.endswith(".tif") and not file.endswith("_mean.tif"):
        fnames_all .append(file)
fnames_all.sort()

#Create the job array

with open('ArrayCaiman.pbs','w') as the_file:
	the_file.write('#!/bin/bash \n')
	the_file.write('#PBS -A UQ-SCI-SBMS \n')
	the_file.write('#PBS -l select=1:ncpus=6:mem=110GB:vmem=110GB \n')
	the_file.write('#PBS -l walltime=6:00:00 \n')
	the_file.write('#PBS -j oe \n')
	job_string = """#PBS -t 1-%s \n""" % (str(len(fnames_all)))
	the_file.write(job_string)
	job_string = job_string = 'filename=`ls -1 '+base_folder+'/*0.tif | tail -n +\${PBS_ARRAY_INDEX} | head -1` \n'
	the_file.write(job_string)
	job_string = 'python ~/CaImAn/CaImAn_Euramoo_singleThread_flow_2.py $filename \n' 
	the_file.write(job_string)

#Now do some things
os.environ["FPS"]=fps
os.environ["COMP"]=comp
job_string = """qsub -v FPS=%s,COMP=%s ArrayCaiman.pbs""" % (fps, comp)
print job_string
call([job_string],shell=True)
