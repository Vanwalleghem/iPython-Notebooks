from subprocess import call
import time
import glob
import os
import sys

fnames_all =[]
folder=str(sys.argv[1])
fps=str(sys.argv[2])
comp=str(sys.argv[3])
base_folder='/30days/uqgvanwa/'+folder # folder containing the demo files
for file in glob.glob(os.path.join(base_folder,'*.tif')):
    if file.endswith(".tif") and not file.endswith("_mean.tif"):
        fnames_all .append(file)
fnames_all.sort()

# Loop over your jobs

for i,name in enumerate(fnames_all):
        job_string = """qsub -v DATA=%s,FPS=%s,COMP=%s Caiman.pbs""" % (name, fps, comp)
	os.environ["FPS"]=fps	
	os.environ["COMP"]=comp
        call([job_string],shell=True)
	print job_string


