import sys
import os
import numpy as np
from scipy.io import savemat
import scipy.stats as scistat
import caiman as cm
import tifffile as tif
from caiman.components_evaluation import evaluate_components
import caiman.source_extraction.cnmf as cnmf
from caiman.source_extraction.cnmf.utilities import extract_DF_F
fnames=[sys.argv[1]]	#name of the movie
final_frate=int(os.environ["FPS"]) # frame rate in Hz
K=int(os.environ["COMP"]) # number of neurons expected per patch, that seems to work well
n_processes = 5 # if using the intel nodes
single_thread=True   
dview=None
gSig=[2,2] # expected half size of neurons, works for nuclear GCaMP
merge_thresh=0.95 # merging threshold, max correlation allowed
p=2 #order of the autoregressive system
downsample_factor=1 # use .2 or .1 if file is large and you want a quick answer
final_frate=final_frate*downsample_factor
idx_xy=None
base_name=os.environ["TMPDIR"]+"/"

fname_new=cm.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy )
fname_new=cm.save_memmap_join(fname_new,base_name=base_name+'Yr', n_chunks=n_processes, dview=dview)
Yr,dims,T=cm.load_memmap(fname_new)
Y=np.reshape(Yr,dims+(T,),order='F')



####################Harry's changed code here- Frame drop detection##################
#Load called file
m_orig = cm.load_movie_chain(fnames)

#Get averages and standard deviations for top-left and bottom-right quarters of image (asymmetrical necessary for unknown reason)
averages=[]
averages2=[]
stds=[]
stds2=[]
for i in range(Y.shape[2]):
    toaverage=(Y[Y.shape[0]/2:,Y.shape[1]/2:,i]-Y[Y.shape[0]/2:,Y.shape[1]/2:,i-1])
    toaverage2=(Y[:Y.shape[0]/2,:Y.shape[1]/2,i]-Y[:Y.shape[0]/2,:Y.shape[1]/2,i-1])
    averages.append(np.mean(toaverage).tolist())
    averages2.append(np.mean(toaverage2).tolist())
    stds.append(np.std(toaverage).tolist())
    stds2.append(np.std(toaverage2).tolist())

#Get product and z-score to try and prevent false-positives in isolated segments	
asymm=[a*b for a,b in zip(averages,averages2)]
astds=[a*b for a,b in zip(stds,stds2)]
asymmzs=scistat.zscore(asymm)
astdzs=scistat.zscore(astds)

#Define sigma maximum acceptable z-score for avg-intensity or avg-stdev
sigma=3
droppedframes=np.argwhere(((astdzs <= -sigma) | (astdzs >= sigma)) & ((asymmzs<=-sigma) | (asymmzs>=sigma)))
droppedframes=np.array(droppedframes).reshape(-1,).tolist()

#Take all frames exceeding this threshold, and their adjacent frames, as a set to be culled
dropsp1=[i+1 for i in droppedframes]
dropsm1=[i-1 for i in droppedframes]
dropsmerge=droppedframes+dropsp1+dropsm1
cullframes= [i for i in list(set(sorted(dropsmerge))) if i>=1]

#Set all culled frames to zeros (deprecated, and redundant from following code, but left in in case removing it breaks something...)
m_orig=m_orig.astype('float')
for i in range(m_orig.shape[0]):
    if np.in1d(i,cullframes):
        m_orig[i,:,:]=np.zeros(((m_orig.shape[1]),(m_orig.shape[2])))

#Get neighbouring frames to culled frames		
ranges=[]
from operator import itemgetter
from itertools import groupby
for k, g in groupby(enumerate(cullframes), lambda (i,x):i-x):
    group = map(itemgetter(1), g)
    ranges.append((group[0], group[-1]))

#Average frames bordering culled frames, then overwrite culled frames with average of bordering frames	
for i in range(len(ranges)):
    avgborders=(m_orig[np.amax(ranges[i])+1,:,:]+m_orig[np.amin(ranges[i])-1,:,:])/2
    for j in range((ranges[i])[0],(ranges[i])[1]+1):
        m_orig[j,:,:]=avgborders

#Reset the datatype which was changed in the deprecated code, then overwrite original savefile		
m_orig=m_orig.astype('uint16')
tif.imsave(sys.argv[1],m_orig)
################################End code changes###########################


fname_new=cm.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy )
fname_new=cm.save_memmap_join(fname_new,base_name=base_name+'Yr', n_chunks=n_processes, dview=dview)
Yr,dims,T=cm.load_memmap(fname_new)
Y=np.reshape(Yr,dims+(T,),order='F')
nb_back=1
options = cnmf.utilities.CNMFSetParms(Y,n_processes,p=p,gSig=gSig,K=K,ssub=1,tsub=1,nb=nb_back)
#options['preprocess_params']['max_num_samples_fft']=10000
Cn = cm.local_correlations(Y)
Yr,sn,g,psx = cnmf.pre_processing.preprocess_data(Yr,dview=dview,**options['preprocess_params'])
Ain,Cin, b_in, f_in, center=cnmf.initialization.initialize_components(Y, **options['init_params'])
Ain,b_in,Cin, f_in = cnmf.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, dview=dview,**options['spatial_params'])
if Cin.size > 0:
	options = cnmf.utilities.CNMFSetParms(Y,n_processes,p=p,gSig=gSig,K=K,tsub=1)
	options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
	Cin,Ain,b_in,f_in,S,bl,c1,neurons_sn,g,YrA = cnmf.temporal.update_temporal_components(Yr,Ain,b_in,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])		
	Ain,Cin,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cnmf.merging.merge_components(Yr,Ain,b_in,Cin,f_in,S,sn,options['temporal_params'], options['spatial_params'],dview=dview, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=merge_thresh,mx=1000, fast_merge = True)	
	Ain,b_in,Cin, f_in = cnmf.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, dview=dview,**options['spatial_params'])	
	options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution	
	Cin,Ain,b_in,f_in,S,bl,c1,neurons_sn,g,YrA = cnmf.temporal.update_temporal_components(Yr,Ain,b_in,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])		
	traces=Cin+YrA
	tB = np.minimum(-2,np.floor(-5./30*final_frate))
	tA = np.maximum(5,np.ceil(25./30*final_frate))
	fitness_raw, fitness_delta, erfc_raw,erfc_delta,r_values,num_significant_samples = evaluate_components(Y,traces,Ain,Cin,bl,f_in, final_frate, remove_baseline = True, N = 5, robust_std = False, Athresh = 0.1, Npeaks = 5, thresh_C = 0.2)	
    
	idx_components_r=np.where(r_values>=.6)[0]
	idx_components_raw=np.where(fitness_raw<-40)[0]        
	idx_components_delta=np.where(fitness_delta<-20)[0]
	idx_components=np.union1d(idx_components_r,idx_components_raw)
	C_dff = extract_DF_F(Yr, Ain.tocsc()[:, idx_components], Cin[idx_components, :], bl[idx_components], quantileMin = 8, frames_window = 200, dview = dview)
	idx_components=np.union1d(idx_components,idx_components_delta)  	
	idx_components_bad=np.setdiff1d(range(len(traces)),idx_components)	
	savemat(fnames[0][:-4]+'_output_analysis_matlab.mat',mdict={'ROIs':Ain,'DenoisedTraces':Cin,'Baseline':bl, 'Noise':YrA, 'Spikes': S,'DFF':C_dff , 'idx_components':idx_components, 'Correlation_image':Cn})
