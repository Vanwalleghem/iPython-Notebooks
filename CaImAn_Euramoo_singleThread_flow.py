import sys
import os
import numpy as np
from scipy.io import savemat
import caiman as cm
#from caiman.motion_correction import motion_correct_oneP_rigid
from caiman.components_evaluation import evaluate_components
import caiman.source_extraction.cnmf as cnmf
#from caiman.source_extraction.cnmf.utilities import extract_DF_F
import tifffile
fnames=[sys.argv[1]]	#name of the movie
fnames_orig=fnames
final_frate=int(os.environ["FPS"]) # frame rate in Hz
K=int(os.environ["COMP"]) # number of neurons expected per patch, that seems to work well
n_processes = 5 # if using the intel nodes
single_thread=True   
dview=None
gSig=[2,2] # expected half size of neurons, works for nuclear GCaMP
merge_thresh=0.8 # merging threshold, max correlation allowed
p=1 #order of the autoregressive system
downsample_factor=1 # use .2 or .1 if file is large and you want a quick answers
final_frate=final_frate*downsample_factor
spatial_factor=1
idx_xy=None
base_name=os.environ["TMPDIR"]+"/"
fname_new=cm.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(spatial_factor, spatial_factor, downsample_factor), remove_init=0,idx_xy=idx_xy )
fname_new=cm.save_memmap_join(fname_new,base_name=base_name+'Yr', n_chunks=n_processes, dview=dview)
#mc = motion_correct_oneP_rigid(fnames[0],gSig_filt = gSig,max_shifts = [5,5],dview=dview,splits_rig = 5,save_movie = True)    
#fname_new = cm.save_memmap([ mc.fname_tot_rig], base_name='memmap_', order = 'C')
#fname_new = cm.save_memmap(fnames, base_name='memmap_', order = 'C')
Yr,dims,T=cm.load_memmap(fname_new)
Y=np.reshape(Yr,dims+(T,),order='F')
tifffile.imsave(fnames[0][:-4]+'_mean.tif',np.mean(Y,axis=2))
#tifffile.imsave(base_name+'temp.tif',Y.swapaxes(0,2).swapaxes(1,2))a
#fnames=[base_name+'temp.tif']
#fname_new=cm.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy )
#fname_new=cm.save_memmap_join(fname_new,base_name=base_name+'Yr', n_chunks=n_processes, dview=dview)
nb_back=1
options = cnmf.utilities.CNMFSetParms(Y,n_processes,p=p,gSig=gSig,K=K,ssub=1,tsub=1,nb=nb_back,n_pixels_per_process=1000,block_size=1000, remove_very_bad_comps=True)
#options['preprocess_params']['max_num_samples_fft']=1000
Cn = cm.local_correlations(Y)
#savemat(fnames_orig[0][:-4]+'_output_correlation.mat',mdict={'Correlation_image':Cn})
Yr,sn,g,psx = cnmf.pre_processing.preprocess_data(Yr,dview=dview,n_pixels_per_process=1000, noise_range = [0.25,0.5],noise_method = 'logmexp', compute_g=False,  p = 1,lags = 5, include_noise = False, pixels = None,max_num_samples_fft=3000, check_nan = True)
options['init_params']['use_hals']=True
Ain,Cin, b_in, f_in, center=cnmf.initialization.initialize_components(Y, **options['init_params'])
Ain,b_in,Cin, f_in = cnmf.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, dview=dview,**options['spatial_params'])
options['temporal_params']['n_pixels_per_process']=1000
options['spatial_params']['block_size']=1000
if Cin.size > 0:
	#options = cnmf.utilities.CNMFSetParms(Y,n_processes,p=p,gSig=gSig,K=K,tsub=1)
	options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
	Cin,Ain,b_in,f_in,S,bl,c1,neurons_sn,g,YrA,lam = cnmf.temporal.update_temporal_components(Yr,Ain,b_in,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])		
	Ain,Cin,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cnmf.merging.merge_components(Yr,Ain,b_in,Cin,f_in,S,sn,options['temporal_params'], options['spatial_params'],dview=dview, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=merge_thresh,mx=1000, fast_merge = True)	
	Ain,b_in,Cin, f_in = cnmf.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, dview=dview,**options['spatial_params'])	
	options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution	
	#options['temporal_params']['method'] = 'cvxpy'
	Cin,Ain,b_in,f_in,S,bl,c1,neurons_sn,g,YrA,lam = cnmf.temporal.update_temporal_components(Yr,Ain,b_in,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])		
	traces=Cin+YrA
	tB = np.minimum(-2,np.floor(-5./30*final_frate))
	tA = np.maximum(5,np.ceil(25./30*final_frate))
	fitness_raw, fitness_delta, erfc_raw,erfc_delta,r_values,num_significant_samples = evaluate_components(Y,traces,Ain,Cin,bl,f_in, final_frate, remove_baseline = True, N = 5, robust_std = False, Athresh = 0.1, Npeaks = 5, thresh_C = 0.2)	
    
	idx_components_r=np.where(r_values>=.6)[0]
	idx_components_raw=np.where(fitness_raw<-40)[0]        
	idx_components_delta=np.where(fitness_delta<-20)[0]
	idx_components=np.union1d(idx_components_r,idx_components_raw)
	#C_dff = extract_DF_F(Yr, Ain.tocsc()[:, idx_components], Cin[idx_components, :], bl[idx_components], quantileMin = 8, frames_window = 200, dview = dview)
	idx_components=np.union1d(idx_components,idx_components_delta)  	
	idx_components_bad=np.setdiff1d(range(len(traces)),idx_components)	
	#savemat(fnames[0][:-4]+'_output_analysis_matlab.mat',mdict={'ROIs':Ain,'DenoisedTraces':Cin,'Baseline':bl, 'Noise':YrA, 'Spikes': S,'DFF':C_dff , 'idx_components':idx_components, 'Correlation_image':Cn})
	savemat(fnames_orig[0][:-4]+'_output_analysis_matlab.mat',mdict={'ROIs':Ain,'DenoisedTraces':Cin,'Baseline':bl, 'Noise':YrA, 'Spikes': S,'idx_components':idx_components})
	savemat(fnames_orig[0][:-4]+'_output_correlation.mat',mdict={'Correlation_image':Cn})
	#Cdf = extract_DF_F(Yr=Yr, A=Ain, C=Cin, bl=bl, dview = dview)
	#savemat(fnames_orig[0][:-4]+'_output_DF_matlab.mat',mdict={'DF':Cdf})
