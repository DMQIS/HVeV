# modified version of `nexus_processing_alt/Nexus_utils.py`

import ast
import h5py
#import joblib
import os
#import pickle
from glob import glob

#from pylab import *
import numpy as np
#import pandas as pd
import scipy
from scipy.optimize import curve_fit

from scipy.stats import skew
from scdmsPyTools.Cuts.General import removeOutliers
from scdmsPyTools.Traces.Stats import slope
#from scdmsPyTools.Traces.Noise import *
# import Noise_mod  as Noise # in case the scdmsPyTools is not the modified version
import scdmsPyTools.Traces.Noise as Noise  
#from scdmsPyTools.TES.Templates import *
#from scdmsPyTools.Traces.Filtering import *
try:
	import scdmsPyTools.BatTools.IO as io
except ImportError:
	import rawio.IO as io
	pass

# put these after the scdmsPyTools imports
from datetime import datetime


# basic functions
def Gauss(x, a, x0, sigma):
	return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def Pulse2(t, tau1, tau2, t0=0, normalized=True):
	pulse = np.zeros(len(t))
	dt = t - t0
	m = (dt>0)
	pulse[m] = (np.exp(-dt[m]/tau1)-np.exp(-dt[m]/tau2))
	if normalized:
		# normalized to integrate to 1
		#norm = 1/(tau1-tau2)
		# normalized to peak at 1
		norm = 1/((tau2/tau1)**(tau2/(tau1-tau2)) - (tau2/tau1)**(tau1/(tau1-tau2)))
		pulse *= norm
	return pulse


class loader:
	'''
	Class for loading data. rawIO wrapper, primarily
	'''
	
	def __init__(self,filename_pattern,data_type='NEXUS',file_range=None,keylist=None):
		self.data_type=data_type
		self.current_file=None
		self.current_file_idx=-1
		self.total_events_in_current_file=0
		self.current_event_idx_in_current_file=0
		self.total_events=0
		self.current_event_idx=0
		self.current_event=None
		self.is_end_of_series=False
		
		self.filename_pattern = filename_pattern
		self.file_list = self.sortByExt(glob(filename_pattern),self.data_type)
		if file_range is not None:
			self.file_list = self.file_list[file_range[0]-1:file_range[1]]
		
		# Specify which key to read. If none, will read all keys
		self.keylist = keylist
	

	def fridgeTemp_Cali(self, fridgeTemp):
		def valmap(x, in_min, in_max, out_min, out_max):
			return ((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)
		# -10V to +10 V is mapped to 16000 Ohms to 11000 Ohms.
		mapedRes = valmap(np.array(fridgeTemp), -10, 10, 16000, 11000) 
		tempMap=np.array([[2000, 13],[2091.76, 8.75472],[2111.74, 8.24688],[2131.96, 7.79824],[2153.45, 7.3307],[2172.28, 6.98209],[2193.39, 6.60556],[2214.54, 6.24911],[2234.71, 5.92383],[2255.85, 5.63267],[2277.37, 5.33618],[2297.98, 5.0742],[2321.09, 4.80528],[2342.69, 4.56953],[2365.53, 4.33958],[2389.61, 4.1171],[2413.41, 3.91027],[2437.55, 3.71767],[2462.4, 3.53223],[2488.58, 3.35425],[2514.47, 3.19003],[2541.36, 3.03094],[2569.76, 2.87789],[2597.76, 2.73573],[2627.01, 2.59996],[2657.08, 2.47007],[2688.21, 2.34516],[2720.79, 2.22729],[2753.39, 2.11709],[2787.75, 2.01079],[2822.16, 1.91048],[2857.59, 1.81714],[2894.9, 1.72238],[2931.82, 1.63456],[2972.24, 1.55525],[2975.99, 1.54451],[3017.9, 1.46584],[3062.09, 1.39122],[3104.63, 1.32172],[3149.67, 1.25639],[3196.38, 1.19266],[3244.4, 1.13302],[3293.54, 1.07694],[3344.54, 1.02287],[3398.43, 0.97098],[3452.02, 0.92316],[3509.21, 0.87715],[3567.86, 0.83287],[3628.67, 0.79087],[3690.2, 0.75181],[3755.35, 0.71391],[3822.63, 0.67768],[3891.05, 0.64406],[3965.85, 0.61161],[4039.32, 0.58115],[4115.71, 0.55192],[4195.41, 0.52452],[4277.2, 0.49818],[4362.43, 0.47318],[4450.33, 0.44946],[4541.67, 0.42689],[4635.74, 0.40566],[4734.35, 0.38541],[4834.75, 0.36619],[4939.2, 0.34774],[5047.99, 0.33053],[5161.99, 0.31388],[5278.46, 0.29816],[5399.95, 0.28323],[5525.92, 0.26919],[5656.34, 0.25577],[5792.96, 0.2429],[5933.94, 0.23056],[6079.5, 0.21917],[6231.7, 0.20819],[6389.26, 0.19781],[6553.22, 0.18794],[6725.12, 0.17855],[6902.83, 0.16959],[7087.59, 0.16118],[7280.69, 0.1531],[7482.65, 0.14544],[7690.85, 0.13817],[7907.85, 0.13125],[8135.97, 0.12472],[8373.12, 0.1185],[8615.15, 0.11272],[8874.08, 0.10706],[9144.32, 0.10174],[9431.32, 0.09667],[9726.49, 0.0918],[10042.53, 0.08724],[10369.37, 0.08286],[10707.65, 0.0787],[11064.44, 0.07478],[11442, 0.07102],[11462.52, 0.07088],[11862.16, 0.06733],[12280.39, 0.06395],[12716.85, 0.06076],[13177.93, 0.05772],[13663.93, 0.05485],[14185.57, 0.05208],[14725.22, 0.04944],[15298.93, 0.04697],[15905.6, 0.0446],[16547.63, 0.04234],[17223.91, 0.04024],[17944.17, 0.03818],[18706.85, 0.03624],[19520.36, 0.03443],[20379.67, 0.03266],[21303.21, 0.03103],[21322.02, 0.03099],[21335.68, 0.03097],[22282.49, 0.02947],[22303.95, 0.02941],[22305.42, 0.02939],[22311.6, 0.0294],[23334.01, 0.02792],[23348.78, 0.02793],[23362.07, 0.02791],[23377.9, 0.02793],[24473.06, 0.0265],[24487.71, 0.02649],[24492.17, 0.02647],[25701.99, 0.0251],[25710.92, 0.02505],[25741.29, 0.02499],[26980.53, 0.02381],[27022.84, 0.02376],[27042.04, 0.02373],[28428.92, 0.02251],[28443, 0.02246],[28512.87, 0.02237],[29891.37, 0.02136],[30021.06, 0.02121],[30101.15, 0.02119],[30119.28, 0.02116],[31573.57, 0.02016],[31746.06, 0.02002],[31793.93, 0.01998],[31908.61, 0.01989],[33423.11, 0.01905],[33593.63, 0.01885],[33635.96, 0.01886],[35473.63, 0.018],[35618.27, 0.0178],[35668.68, 0.01781],[38132.33, 0.01669],[38258.71, 0.01665],])   
		# Use this lookup table to convert resistance to temperature
		f=scipy.interpolate.interp1d(tempMap[:,0],tempMap[:,1], bounds_error=False, fill_value=-999.)
		return f(mapedRes)
	
	def readHDFFile(self,filename):
		fdata = h5py.File(filename,'r')

		metadata = ast.literal_eval(fdata['metadata']['metadata'][0])
		data=fdata['traces']['traces']
		DIexist=False
		if ('DIenabled' in metadata.keys()):
			if (metadata['DIenabled']):
				DIdata=fdata['traces']['DItraces']
				DIexist=True

		for k,v in metadata.items():
			if(k == 'chan_ind'):
				continue
			if(type(v) == list):
				metadata[k]=np.array(metadata[k],dtype=np.float)

		sample_rate = float(metadata['sample_rate_measured'])
		number_channels = len(metadata['Rb'])

		Rfb = metadata['Rfb']
		TR = metadata['TR']
		Amp = metadata['Amp']
		SF = metadata['SF']
		Gains = Amp * SF
		if 'ForI' in metadata:
			if (metadata['ForI'] == 'I'):
				Gains = Gains / metadata['dynamic_range'] * 32768.
		dVdI = Rfb * TR * Gains  # Volts to Amps
		dIdV = 1.0 / dVdI # Amps to Volts
		chan_ind = metadata['chan_ind']

		number_samples=int(len(data)/number_channels)
		data=np.reshape(data,(number_channels,number_samples))# * dIdV[:,None]
		# data*=dIdV[:,None]

		res=dict()
		chan_names=list()
		for ich in range(4):
			if (chan_ind[ich] >= 0):
				ch_str = 'CH%d'%(ich+1)
				chan_names.append(ch_str)
				res[ch_str] = [data[chan_ind[ich]]]
		relweight=1.0
		try:
			res['Total'] = [res['CH2'][0]+res['CH3'][0]*relweight]
		except:
			res['Total'] = [res['CH2']]
		if DIexist:
			res['CHD'] = np.array(DIdata)
			chan_names.append('CHD')
			chan_names.append('CHD0')
			chan_names.append('CHD1')
			chan_names.append('CHD2')
			unpackedDI = np.unpackbits(res['CHD'])
			res['CHD0'] = [unpackedDI[7::8]]
			res['CHD1'] = [unpackedDI[6::8]]
			res['CHD2'] = [unpackedDI[5::8]]

		res['chan_names']=np.append(chan_names,'Total')

		res['dVdI']=dVdI
		res['Fs']=sample_rate
		res['number_samples']=number_samples
		res['prop']=metadata
		res['filenum']=1
		res['trigpt'] = None
		res['filename'] = filename.split('/')[-1].split('.')[0]

		if('crystal_bias_srs' in metadata.keys()):
			res['crystal_V'] = metadata['crystal_bias_srs']
			res['crystal_I'] = metadata['HV_current_srs']
			res['fridgeTemp'] = metadata['fridge_temp_srs']
			res['fridgeTempCali'] = self.fridgeTemp_Cali(res['fridgeTemp'])
			res['now'] = metadata['now']

		fdata.close()

		return res
	
	def fileIO(self,filename,data_type):
		if data_type == 'SLAC':
			data = io.getRawEvents("",filename,channelList=self.keylist)["Z1"]
		
			# Convert to int16 to save RAM
			# When doing processing, try to keep it as int16
			self.data=dict()
			for ch in data:
				self.data[ch] = np.array([(-d+32768).astype(np.int16) for d in data[ch]])
			event_idx = np.transpose(data.index.tolist())[1]%10000
			event_info = io.getEventInfo('', filename)

			# Series start time
			start_time_second_str = "20"+str(np.transpose(data.index.tolist())[0][0])[2:]
			datetime_object = datetime.strptime(start_time_second_str, '%Y%m%d%H%M%S')
			epoch = datetime.utcfromtimestamp(0)
			series_start_time_epoch = (datetime_object - epoch).total_seconds()

			self.data["event_info"] = event_info
			self.data["event_idx"] = event_idx
			self.data["filename"] = os.path.basename(filename[0])
			self.data["series_start_time_epoch"]=series_start_time_epoch
			self.data['ADC2A'] = 1/2**16 *8/5e3 /2.4/4
		elif data_type == 'NEXUS':
			data = io.getRawEvents("",filename,channelList=self.keylist)["Z1"]
		
			# Convert to int16 to save RAM
			# When doing processing, try to keep it as int16
			self.data=dict()
			for ch in data:
				self.data[ch] = np.array([(-d+32768).astype(np.int16) for d in data[ch]])
			event_idx = np.transpose(data.index.tolist())[1]%10000
			event_info = io.getEventInfo('', filename)

			# Series start time
			start_time_second_str = "20"+str(np.transpose(data.index.tolist())[0][0])[2:]
			datetime_object = datetime.strptime(start_time_second_str, '%Y%m%d%H%M%S')
			epoch = datetime.utcfromtimestamp(0)
			series_start_time_epoch = (datetime_object - epoch).total_seconds()

			self.data["event_info"] = event_info
			self.data["event_idx"] = event_idx
			self.data["filename"] = os.path.basename(filename[0])
			self.data["series_start_time_epoch"]=series_start_time_epoch
			self.data['ADC2A'] = 1/2**16 *8.192 / 4/2100/14
		
		elif data_type == 'Animal':
			if len(filename)>1:
				raise Exception("One file at a time for Animal")
			self.data = self.readHDFFile(filename[0])
		return self.data
		
	def sortByExt(self,files,data_type):
		numbers=list()
		if data_type == 'SLAC' or data_type == 'NEXUS':
			for f in files: # RUN00099_DUMP0000.mid.gz
				numbers.append(int(f.split('_')[-1].split('.')[0][-4:]))
		elif data_type == 'Animal':
			for f in files:
				numbers.append(int(f.split('_')[-1].split('.')[0]))
		return np.array(files)[np.argsort(numbers)].tolist()
	
	def load_next(self):
		if self.is_end_of_series is False:
			if (self.current_file is None) or (self.current_event_idx_in_current_file==self.total_events_in_current_file):
				self.current_file_idx+=1
				self.current_file = self.file_list[self.current_file_idx]
				self.data = self.fileIO(self.current_file,self.data_type)
				if self.keylist is None:
					self.keylist=list(self.data.keys())

				self.total_events_in_current_file=len(self.data[self.keylist[0]])
				self.current_event_idx_in_current_file=0
				self.total_events+=self.current_event_idx_in_current_file


			self.current_event = dict()
			for key in self.keylist:
				self.current_event[key]=self.data[key][self.current_event_idx_in_current_file]
			self.current_event_idx_in_current_file+=1 # Move to nexe event
			self.current_event_idx+=1 # Add one to the counter
			
			# If at the end of one file, move to the next
			
			# If at the end of all files
			if (self.current_file_idx == len(self.file_list)-1) and (self.current_event_idx_in_current_file==self.total_events_in_current_file):
				self.is_end_of_series=True
			
			return self.current_event
		else:
			return -1
		
	def load_next_n(self, nevents):
		
		# Initialize a dict to hold the result
		# If there is nothing yet, read one event
		if self.current_event is None:
			self.load_next()
			nevents-=1
		self.current_N_events=dict()
		for key in self.keylist:
			self.current_N_events[key]=[self.current_event[key]]
			
		# Loop until reaches number of events required
		while (nevents>0) and (self.is_end_of_series==False):
			self.load_next()
			for key in self.keylist:
				self.current_N_events[key].append(self.current_event[key])
			nevents-=1
		if self.is_end_of_series==True:
			print(f"Not enough data, could not get the last {nevents} events")
			
		for key in self.keylist:
				self.current_N_events[key]=np.array(self.current_N_events[key])
				
		return self.current_N_events

	def load_next_nfiles(self, nfiles):
		# First, decide the indices of next n files
		# If starting with nothing:
		if (self.current_file_idx==-1):
			filename_start_idx = 0
			self.current_file_idx=0
		else:
			filename_start_idx = self.current_file_idx
			
		# If requested is within total file number
		if (len(self.file_list))>(self.current_file_idx+nfiles):
			filename_end_idx = self.current_file_idx+nfiles
		else:
			filename_end_idx=len(self.file_list)
		self.next_nfilenames = self.file_list[filename_start_idx:filename_end_idx]
		
		self.current_file_idx=filename_end_idx
		
		# rawIO already have built-in method to do this
		return self.fileIO(self.next_nfilenames,data_type=self.data_type)

	
	
class trigger:
	def __init__(self, data, channel_config, trigger_channels, trigger_threshold,
				 deactivation_threshold=None, TTL_THRESHOLD=300, INRUN_RANDOM=2,
				 WINDOW_TEMPLATE=False, USE_GAUS_KERNEL=False, gauss_sigma=20, BYPASS_HIGH_ENERGY_IN_FILTER = False,
				 trigger_type=0, align_max = True, filter_kernels=None, PSD=None, OF_LPF=-1,
				 pre_trig=4096, post_trig=4096, pre_trig_kernel=4096, post_trig_kernel=4096, fs=625000, 
				 keep_metadata=True, verbose=True, use_ttl_falling_edge=False, remove_trigger_offset=False,
				 data_type=None, ADC2A=1.0):
		'''
		Class for simulated trigger. Finds pulses and cuts them out full trace

		data: dict. from `loader`
				Each channel should have traces in the unit of Amps.
		channel_config: dict
				channel information
		trigger_channels: list
				channels to trigger on. If a new channel needs to be made, use syntax new_channel_name=chA_name+chB_name+..., e.g. "Total_NFC = PES2+PFS2"
		trigger_threshold: list of float number, or list of string
				float list  - absolute trigger threshold for each channel, ADC unit
				string list - trigger threshold in the unit of sigmas. Will automatically calculate the trigger threshold
		TTL_THRESHOLD: int
				TTL trigger threshold, ADC unit
		INRUN_RANDOM: int
				Random trigger per trace. Default is 2 (4Hz for 0.5-second trace)
		WINDOW_TEMPLATE: bool
				Apply a window function to the template
		USE_GAUS_KERNEL: bool
				overwrite the given filter_kernels with default Gaus-derivitive filter
		gauss_sigma : int
				width of the gauss in the unit of samples for the gaussian filter (if the USE_GAUS_KERNEL is True)
		BYPASS_HIGH_ENERGY_IN_FILTER: bool
				bypass the high energy trace with gaus filter. Only works when USE_GAUS_KERNEL is False.
		trigger_type: int
				0 - threshold trigger
				1 - threshold trigger after filtering the trace with shaping filter
				2 - random trigger
		trigger_kernels: list 
				Default: None. Will generate a pulse shape with hardcoded rise and fall time.
				trigger kernels for each channel. Number of kernels should follow the order of trigger channel. 
				If PSD is given, it will be weighted by PSD
		PSD: list
				Default: None.
				PSDs for each channel. Number of PSDs should follow the order of the trigger channel.
		align_max: boolean
				Align the trigger point to the maximum after the threshold-crossing
		OF_LPF: list or int
				The bin number of the low-pass filter frequency cutoff in the OF template. Default is -1 - no low-pass filter
		use_ttl_falling_edge : bool
				If true, using the falling edge of the TTL signal for the TTL triggering
		remove_trigger_offset : bool
				calculate expected trigger location offsets by convolving the filter kernel with the templates
		'''
		
		# parse arguments
		self.data = data
		self.channel_config = list(channel_config) # copy channel list
		self.trigger_channels = list(trigger_channels)
		self.Fs = fs # Sampling frequency
		self.align_max = align_max
		self.pre_trig,self.post_trig = pre_trig,post_trig
		self.verbose = verbose
		self.use_ttl_falling_edge = use_ttl_falling_edge
		self.remove_trigger_offset = remove_trigger_offset
		if data_type is not None:
			self.data_type = data_type
		elif 'event_info' in data.keys():
			self.event_info = data['event_info']
			self.data_type = 'NEXUS'
		elif 'dVdI' in data.keys():
			self.dVdI = data['dVdI']
			self.data_type = 'ANIMAL'
		else:
			print('ERROR: Unrecognized data format')
			return
		self.ADC2A = ADC2A
		self.INRUN_RANDOM = INRUN_RANDOM
		self.WINDOW_TEMPLATE = WINDOW_TEMPLATE
		# trigger type
		if type(trigger_type) is list and len(trigger_type) == len(trigger_channels):
			self.trigger_type = trigger_type
		elif type(trigger_type) is int:
			self.trigger_type = [trigger_type]*len(trigger_channels)
		else:
			print('ERROR: Bad trigger_type')
			return -1
		# force TTL channel to use unfiltered level trigger
		for i in range(len(self.trigger_channels)):
			if self.trigger_channels[i] == 'TTL': 
				self.trigger_type[i] = 0
		self.TTL_THRESHOLD = TTL_THRESHOLD
		
		# 1. Data processing
		# 1.1. Add additional trigger channels
		self._add_channels(self.data, channel_config)
		# 1.2. Filtering
		if filter_kernels is None or USE_GAUS_KERNEL:
			self.filter_kernels = [self._make_filter_kernel(pre_trig_kernel, post_trig_kernel, USE_GAUS_KERNEL=USE_GAUS_KERNEL, gaus_sigma=gauss_sigma)\
								   for i in range(len(self.trigger_channels))]
			if filter_kernels is not None:
				# Normalize gaus kernels with templates, if templates are given:
				for i, ch in enumerate(self.trigger_channels):
					if self.trigger_type[i]!=1 or ch=="TTL":
						continue
					else:
						self.filter_kernels[i] = self.filter_kernels[i]/np.max(np.correlate(self.filter_kernels[i],filter_kernels[i]))
		else:
			if PSD is None:
				# this is the MF case, where we use the templates as the filter kernels.
				# the next 4 lines are to normalize the kernel in the same way it is normalized for the gaussian filter
				self.filter_kernels = filter_kernels
				for i, ch in enumerate(self.trigger_channels):
					if self.trigger_type[i]!=1 or ch=="TTL":
						continue
					self.filter_kernels[i] = filter_kernels[i]/np.max(np.correlate(filter_kernels[i],filter_kernels[i]))
					self.filter_kernels[i] -= np.mean(self.filter_kernels[i])
				
			else:
				self.filter_kernels = self._make_filter_kernel_OF(filter_kernels,PSD,OF_LPF)
		# Although the function is called, channels that don't have trigger_type==1 will not be filtered.
		self.data_filtered = self._apply_filter(self.data, self.trigger_channels, self.trigger_type, self.filter_kernels)
		

		# 2. Set trigger threshold
		self.set_threshold(self.trigger_channels,trigger_threshold)
		self.set_deactivation_threshold(self.trigger_channels,deactivation_threshold)
			
			
		# 3. Calculating trigger offsets
		self.trigger_offsets = []
		if self.remove_trigger_offset:
			for i,ch in enumerate(self.trigger_channels):
				if filter_kernels is None or not USE_GAUS_KERNEL or self.trigger_type[i]!=1 or ch=='TTL':
					offset = 0
				else:
					filtered_template = scipy.signal.correlate(filter_kernels[i], self.filter_kernels[i], mode="same")
					threshold = np.max(filtered_template)*0.8
					trigger_points, _, _ = self._threshold_trigger(filtered_template, threshold,
																   rising_edge=True, align_max=self.align_max,
																   deactivate_th=None, peak_search_window_limit=self.post_trig)
					if len(trigger_points)<1:
						offset = 0
					else:
						offset = self.pre_trig - trigger_points[0]
				self.trigger_offsets.append(offset)
			print(f'trigger offsets: {self.trigger_channels} - {self.trigger_offsets}')
		
		if BYPASS_HIGH_ENERGY_IN_FILTER and not USE_GAUS_KERNEL:
			gaus_kernels = [self._make_filter_kernel(pre_trig_kernel, post_trig_kernel, USE_GAUS_KERNEL=True, gaus_sigma=gauss_sigma) for i in range(len(self.trigger_channels))]
			# Normalize gaus kernels with templates:
			for i, ch in enumerate(self.trigger_channels):
				if self.trigger_type[i]!=1 or ch=="TTL":
					continue
				else:
					gaus_kernels[i] = gaus_kernels[i]/np.max(np.correlate(gaus_kernels[i],filter_kernels[i]))
			self.data_filtered_gaus = self._apply_filter(self.data, self.trigger_channels, self.trigger_type, gaus_kernels)
			self.data_filtered = self._combine_gaus_filter(self.data_filtered, self.data_filtered_gaus, self.trigger_channels,
														   self.trigger_type, self.pre_trig, self.post_trig, self.trigger_threshold, n_threshold = [8])
		
	#-------------------------------------------------------------------------------------------
	# Functions
	#-------------------------------------------------------------------------------------------
	
	def _add_channels(self, data, channel_config):
		self.channel_list = list(channel_config.keys())
		self.raw_trace_length = len(self.data[self.channel_list[0]][0])		
		# process data to make some new channels if needed
		for ch in channel_config:
			# Handle TTL channel separately. If there is DCRC data on TTL channel, remove from trigger list
			if ch=="TTL":
				try: # in case there is no TTL data
					data[ch]=np.array(data[channel_config[ch]["sub_channels"][0]]).astype(np.int16)
					data[ch]=-data[ch]
				except:
					self.trigger_channels.remove("TTL")
					self.channel_list.remove("TTL")
					continue
				if len(data[ch][0])==0:
					try:
						self.trigger_channels.remove("TTL")
						self.channel_list.remove("TTL")
					except:
						pass
				continue
			if channel_config[ch]["sub_channels"] is not None:
				# Make new channels
				data[ch]=np.zeros_like(data[list(data.keys())[0]]).astype(np.float32)
				for i_subch,sub_ch in enumerate(channel_config[ch]["sub_channels"]):
					data[ch]+=data[sub_ch]*channel_config[ch]["sub_channels_weights"][i_subch]
		
		
	
	def _make_filter_kernel(self,pre_trig, post_trig,tau1=10e-6,tau2=30e-6,AC_coupled=True,USE_GAUS_KERNEL=False,gaus_sigma=20,gaus_truncate=3):
		# JR - rewrite this
		if not USE_GAUS_KERNEL:
			x = np.arange(0,(pre_trig+post_trig))/self.Fs
			x0 = pre_trig/self.Fs
			dx=(x-x0)
			dx*=np.heaviside(dx,1)
			kernel = (np.exp(-dx/tau1)-np.exp(-dx/tau2))/(tau1-tau2)*np.heaviside(dx,1)
			kernel_normed = kernel/(np.dot(kernel,kernel/max(kernel)))

			if AC_coupled:
				kernel_normed-=np.mean(kernel_normed)

			self.kernel_normed = kernel_normed
		else:	 
			x=np.arange(-gaus_sigma*gaus_truncate,gaus_sigma*gaus_truncate+1,1)
			kernel = np.diff(Gauss(x,1,0,gaus_sigma))
			step_function = np.heaviside(x[:-1],1)
			kernel_normed = kernel/(np.dot(kernel,step_function))
			self.kernel_normed = kernel_normed
		return kernel_normed
	
	def _make_filter_kernel_OF(self, filter_kernels, PSD, OF_LPF=-1):
		# JR - combine with similar function in Nexus_RQ.py
		if PSD is None:
			return filter_kernels
		
		filter_kernels_normed = []
		OF_LPF = np.repeat(OF_LPF,len(filter_kernels)) if type(OF_LPF) is int else OF_LPF
		if type(PSD) is list:
			for i, (PSD_template,trigTemplate) in enumerate(zip(PSD,filter_kernels)):
				if len(PSD_template)==0:
					continue
				trigTemplate_fft = np.fft.rfft(trigTemplate)
				if OF_LPF[i]>=0:
					trigTemplate_fft[OF_LPF[i]:]=0
				fft_norm = 1/(len(trigTemplate)/2)  
				J=PSD_template**2
				if sum(J)==0:
					J=np.ones(len(J))
				J[0]=np.inf
				OF = np.conjugate(trigTemplate_fft)/J
				OF_norm = np.real(OF.dot(trigTemplate_fft))*fft_norm
				OF = OF/OF_norm
				trigTemplate_OF = np.fft.irfft(OF.conjugate())
				if self.WINDOW_TEMPLATE:
					trigTemplate_OF*=np.hamming(len(trigTemplate_OF)) # apply a window function
					trigTemplate_OF-=np.mean(trigTemplate_OF)
				filter_kernels_normed.append(trigTemplate_OF)
		return filter_kernels_normed
	
	def _apply_filter(self, data, trigger_channel_list, trigger_type, filter_kernel):
		data_filtered = dict()
		for i, ch in enumerate(trigger_channel_list):
			if trigger_type[i] == 1 and ch != 'TTL':
				data_filtered[ch] = [scipy.signal.correlate(trace, filter_kernel[i],mode="same").astype(np.float32) for trace in data[ch]]
			else: # or not
				data_filtered[ch] = data[ch]
		return data_filtered
	
	def _combine_gaus_filter(self, data_filtered, data_filtered_gaus, trigger_channel_list, trigger_type, pre_trig, post_trig, trigger_threshold, n_threshold=[8]):
		"""
		Combine the OF filtered trace with the gaus-filtered trace, with the threshold of n_threshold times the trigger_threshold
		"""
		data_filtered_combined = dict()
		n_threshold = np.repeat(n_threshold[0], len(trigger_threshold)) if len(n_threshold)==1 else n_threshold
		for i, ch in enumerate(trigger_channel_list):
			if trigger_type[i]!=1 or ch=="TTL":
				data_filtered_combined[ch] = data_filtered[ch]
			else:
				data_filtered_combined[ch] = data_filtered[ch]
				
				# Trigger with the gaus-filtered trace at a relative high threshold
				trig_th = trigger_threshold[i]*n_threshold[i]
				for i_trace,trace in enumerate(data_filtered_gaus[ch]):
					trigger_points, trigger_amps, trigger_width = self._threshold_trigger(trace, trig_th, rising_edge=True,
						   align_max=True, deactivate_th=None, peak_search_window_limit=self.post_trig)
					# If an event is triggered, replace the OF filtered trace with the Gaus filtered trace.
					if len(trigger_points)>0:
						for trigpt in trigger_points:
							data_filtered_combined[ch][i_trace][trigpt-pre_trig:trigpt+post_trig] = data_filtered_gaus[ch][i_trace][trigpt-pre_trig:trigpt+post_trig]
				
		return data_filtered_combined
	
	def _gaus_fit(self, data, drange=None,bins=400):
		if drange is None:
			drange=[min(data),max(data)]
		y,ibins=np.histogram(data,range=drange,bins=bins)
		x = 0.5*(ibins[1:]+ibins[:-1])
		# weighted arithmetic mean (corrected - check the section below)
		mean = sum(x * y) / sum(y)
		sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
		popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
		
		mean,sigma= popt[1],popt[2]
		#print(popt[1],popt[2])
		popt,pcov = curve_fit(Gauss, x[x<mean+sigma], y[x<mean+sigma], p0=[max(y), mean, sigma])
		#print(popt[1],popt[2])
		
		return popt,pcov,x,y

	def set_threshold(self, trigger_channels, trigger_threshold):
		# trigger_channels : list of str
		# trigger_threshold : int or list of ints or str or list of str

		# Handle thresholds, optionally in sigmas
		thresholds = [0 for ch in trigger_channels]
		for i in range(len(trigger_channels)):
			ch = trigger_channels[i]
			# handle flexible `trigger_threshold` format
			if type(trigger_threshold) is list:
				threshi = trigger_threshold[i]
			else:
				threshi = trigger_threshold
			# sigma?
			if type(threshi) is str:
				nsigma = float(trigger_threshold[i].split('sigma')[0])
				# Median is a rough estimation of baseline
				y = data_filtered[ch][0] #- np.median(data[ch][0])
				try:
					popt,pcov,xx,yy = self._gaus_fit(y[self.pre_trig:-self.post_trig],bins=400)
					sigma = np.abs(popt[2])
				except:
					sigma = np.std(y)
					print("WARNING: Failed to fit channel",ch)
				threshi = nsigma * sigma
				print(f'  Using sigma for {ch}: {nsigma} * {sigma} ADCu')
			thresholds[i] = threshi
			print(f'  Setting threshold for {ch}: {threshi} ADCu')
		self.trigger_threshold = thresholds
		return 0

	def set_deactivation_threshold(self,trigger_channels,trigger_threshold=None):
		if trigger_threshold is None:
			self.deactivation_threshold = self.trigger_threshold
			return
		thresholds = [0 for ch in trigger_channels]
		if trigger_threshold is not None:
			for i in range(len(trigger_channels)):
				if type(trigger_threshold) is list:
					threshi = trigger_threshold[i]
				else:
					threshi = trigger_threshold
				thresholds[i] = threshi
		self.deactivation_threshold = thresholds

		

	#-------------------------------------------------------------------------------------------
	# Trigger function		
	def _threshold_trigger(self, trace, trig_th, rising_edge=True, align_max=False,
						   deactivate_th=None, peak_search_window_limit=4096, trigger_offset=0):
		if rising_edge:
			trigger_points=np.flatnonzero((trace[0:-1] < trig_th) & (trace[1:] >= trig_th))+1
		else:
			trigger_points=np.flatnonzero((trace[0:-1] > trig_th) & (trace[1:] <= trig_th))+1
		
		# Align to the maximum point after the trigger cross-over, if desired
		if align_max is False:
			return trigger_points+trigger_offset, np.zeros(len(trigger_points)),np.zeros(len(trigger_points))
		else: 
			if deactivate_th==None:
				deactivate_th=trig_th
			trigger_points_aligned = []
			trigger_amp_aligned = []
			trigger_widths = []
			for trigpt in trigger_points:
				if trigpt<(len(trace)-peak_search_window_limit):
					# Set the peak search window, from the trigger point to where the trace goes below deactivate_th
					search_window = np.argmax(trace[trigpt:trigpt+peak_search_window_limit]<deactivate_th)
					if search_window==0:
						search_window=peak_search_window_limit
					# Adjust trigger point to maximum
					argmax_offset = np.argmax(trace[trigpt:trigpt+search_window])
					trigger_points_aligned.append(trigpt + argmax_offset)
					trigger_amp_aligned.append(trace[trigger_points_aligned[-1]])
					trigger_widths.append(search_window)
			# Remove possible duplications:
			trigger_points_aligned,mask = np.unique(trigger_points_aligned,return_index=True)
			trigger_points_aligned = (trigger_points_aligned + trigger_offset).astype(int)
			trigger_amp_aligned = np.array(trigger_amp_aligned)[mask]
			trigger_widths = np.array(trigger_widths).astype(int)[mask]
			return trigger_points_aligned, trigger_amp_aligned, trigger_widths
	
	def _random_trigger(self, N_random_per_file=None):
		if N_random_per_file is None:
			N_random_per_file = self.raw_trace_length//(self.pre_trig+self.post_trig)-1
		trigger_points = np.random.randint(self.pre_trig+1,self.raw_trace_length-self.post_trig,size=(N_random_per_file))
		return trigger_points, np.zeros(len(trigger_points)), np.zeros(len(trigger_points))
		
	def run_trigger(self, N_random_per_file=None):
		'''
		description
		'''
		# Trigger info
		trigger_result = dict()
		for key in ["trig_ch", "trig_traceidx", "trig_loc", "trig_amp", "trig_width", 'trig_pileups',
					'SeriesNumber', 'EventNumber', 'EventTime', 'TriggerTime',"timestamp","filename_idx"]:
			trigger_result[key] = []
		if self.data_type == 'ANIMAL':
			for key in ["crystal_V","filename","now","fridgeTemp","fridgeTempCali"]:
				trigger_result[key] = []
		# The amplitudes of all channels when trigger fires
		for ch in self.trigger_channels:
			trigger_result[f"trig_amp_{ch}"] = []
		self.trigger_result = trigger_result
		
		OFP_WINDOW = 250
		for i_trace in range(len(self.data_filtered[self.trigger_channels[0]])):
			for i,ch in enumerate(self.trigger_channels):
				trace   = self.data_filtered[ch][i_trace]
				trig_th = self.trigger_threshold[i]
				deac_th = self.deactivation_threshold[i]
				align_max_temp = self.align_max if ch != "TTL" else False # Force TTL channel to use edge trigger only.
				trigger_offset = 0 if len(self.trigger_offsets)<len(self.trigger_channels) else self.trigger_offsets[i]
				trigger_points = []
				
				# If Level trigger -- Trigger type 0,1
				if self.trigger_type[i] in [0,1]:
					rising_edge = True
					if ch == 'TTL' and self.use_ttl_falling_edge:
						rising_edge = False
					trigger_points, trigger_amps, trigger_width = self._threshold_trigger(trace, trig_th, rising_edge=rising_edge,
						   align_max=align_max_temp, deactivate_th=deac_th, peak_search_window_limit=self.post_trig, trigger_offset=trigger_offset)
					# Remove trigger points too close to the edge
					mask = (trigger_points>self.pre_trig)&(trigger_points<self.raw_trace_length-self.post_trig)
					trigger_points = trigger_points[mask]
					trigger_amps   = trigger_amps[mask]
					trig_width	 = trigger_width[mask]
					# Detect trigger pileup within the OFP window
					# JR what is this
					if len(trigger_points)>=1:
						trigger_dt = np.diff(trigger_points)
						if len(trigger_dt)>=2:
							trig_pileups = np.concatenate(([trigger_dt[0]<OFP_WINDOW],(trigger_dt[:-1]<OFP_WINDOW)|(trigger_dt[1:]<OFP_WINDOW),[trigger_dt[-1]<OFP_WINDOW]))
						elif len(trigger_dt)==1:
							trig_pileups = np.concatenate(([trigger_dt[0]<OFP_WINDOW],[trigger_dt[0]<OFP_WINDOW]))
						elif len(trigger_dt)==0:
							trig_pileups = [False]
					else:
						trig_pileups=[]
					
					# Write down trigger info
					trigger_result["trig_ch"]=np.append(trigger_result["trig_ch"], [ch]*len(trigger_points))
					trigger_result["trig_traceidx"]=np.append(trigger_result["trig_traceidx"],[i_trace]*len(trigger_points))
					trigger_result["trig_loc"]=np.append(trigger_result["trig_loc"],trigger_points)
					trigger_result["trig_amp"]=np.append(trigger_result["trig_amp"],trigger_amps)
					trigger_result["trig_width"]=np.append(trigger_result["trig_width"],trig_width)
					trigger_result["trig_pileups"]=np.append(trigger_result["trig_pileups"],trig_pileups)
					
					# Write down the amplitude of all channels when trigger fires
					for j,ch_2 in enumerate(self.trigger_channels):
						trigger_result[f"trig_amp_{ch_2}"]=np.append(trigger_result[f"trig_amp_{ch_2}"],self.data_filtered[ch_2][i_trace][trigger_points])
						
				# If Random trigger -- Trigger type 2
				addrandoms = (i == 0 and len(trigger_points) < self.INRUN_RANDOM)
				if self.trigger_type[i] == 2 or addrandoms:
					# 2 in-run random per 0.5 second trace, if self.INRUN_RANDOM is True
					
					N_random = self.INRUN_RANDOM if (self.INRUN_RANDOM and (self.trigger_type[i] !=2)) else N_random_per_file
					trigger_points_rand, trigger_amps_rand ,trigger_width_rand = self._random_trigger(N_random)
					trig_pileups_rand = np.repeat(False, len(trigger_points_rand))
					trigger_result["trig_ch"]=np.append(trigger_result["trig_ch"], ["Random"]*len(trigger_points_rand))
					trigger_result["trig_traceidx"]=np.append(trigger_result["trig_traceidx"],[i_trace]*len(trigger_points_rand))
					trigger_result["trig_loc"]=np.append(trigger_result["trig_loc"],trigger_points_rand)   
					trigger_result["trig_amp"]=np.append(trigger_result["trig_amp"],trigger_amps_rand)
					trigger_result["trig_width"]=np.append(trigger_result["trig_width"],trigger_width_rand)
					trigger_result["trig_pileups"]=np.append(trigger_result["trig_pileups"],trig_pileups_rand)
					
					if "trigger_points" in locals():
						trigger_points = np.append(trigger_points,trigger_points_rand)
					else:
						trigger_points=trigger_points_rand
						
					# Write down the amplitude of all channels when trigger fires
					for j,ch_2 in enumerate(self.trigger_channels):
						trigger_result[f"trig_amp_{ch_2}"]=np.append(trigger_result[f"trig_amp_{ch_2}"],self.data_filtered[ch_2][i_trace][trigger_points_rand])						
					
				# Information from MIDAS				   
				n_triggers = len(trigger_points)
				if self.data_type == 'NEXUS' or self.data_type == 'SLAC':
					event_info_index = self.data["event_idx"][i_trace]
					for info in self.event_info:
						if info["event"]["EventNumber"]%10000!=event_info_index:
							continue
						else:
							for key in ['SeriesNumber', 'EventNumber', 'EventTime', 'TriggerTime']:
								if key=='TriggerTime':
									trigger_result[key] = np.append(trigger_result[key], [info['trigger'][key],]*n_triggers)
								else:
									trigger_result[key] = np.append(trigger_result[key], [info['event'][key],]*n_triggers)
							info_thisevent=info
							break
					
					trigger_result["filename_idx"] = np.append(trigger_result["filename_idx"],[int(self.data["filename"].split("_")[-1].split(".")[0][-4:])]*n_triggers)
					# This timestamp has steps in it, corresponding to 
					trigger_result["timestamp"] = np.append(trigger_result["timestamp"], info_thisevent['trigger']["TriggerTime"]+1/self.Fs*np.array(trigger_points))
				# Information from Animal
				elif self.data_type == 'ANIMAL':
					trigger_result["SeriesNumber"] = np.append(trigger_result["SeriesNumber"],[int(self.data["filename"].split("_")[0])]*n_triggers)
					trigger_result["filename_idx"] = np.append(trigger_result["filename_idx"],[int(self.data["filename"].split("_")[-1].split(".")[0])]*n_triggers)
					trigger_result["crystal_V"]	= np.append(trigger_result["crystal_V"],[self.data["crystal_V"]]*n_triggers)
					trigger_result["fridgeTemp"]   = np.append(trigger_result["fridgeTemp"],[self.data["fridgeTemp"]]*n_triggers)
					trigger_result["fridgeTempCali"] = np.append(trigger_result["fridgeTempCali"],[self.data["fridgeTempCali"]]*n_triggers)
					trigger_result["now"]		  = np.append(trigger_result["now"],[self.data["now"]]*n_triggers)
					trigger_result["timestamp"]	= np.append(trigger_result["timestamp"],trigger_result["filename_idx"][-1]*1500000/1515151 + 1/1515151*np.array(trigger_points))
					
			
				# If Random trigger -- Trigger type 2
				if self.trigger_type[i] == 2:			
					break # Only ran random on one channel

		trigger_result["trig_traceidx"] = trigger_result["trig_traceidx"].astype(np.int16)
		trigger_result["trig_loc"] = trigger_result["trig_loc"].astype(np.int32)
		trigger_result["trig_amp"] = trigger_result["trig_amp"].astype(np.float32)
		trigger_result["trig_width"] = trigger_result["trig_width"].astype(np.int16)
		trigger_result["filename_idx"] = trigger_result["filename_idx"].astype(np.int16)
		if self.data_type == 'NEXUS' or self.data_type == 'SLAC':
			trigger_result["SeriesNumber"] = trigger_result["SeriesNumber"].astype(np.int64)
			trigger_result["EventNumber"] = trigger_result["EventNumber"].astype(np.int32)
		
		
		# Dictionary that holds triggered traces
		Traces = {}
		for ch in self.channel_list:
			Traces[ch] = []
		# trace window around trigger and append
		for i_event,i_trace in enumerate(trigger_result["trig_traceidx"]):
			trigpt = trigger_result["trig_loc"][i_event]
			for ch in self.channel_list:
				Traces[ch].append(self.data[ch][i_trace][trigpt-self.pre_trig: trigpt+self.post_trig])
		# convert to numpy array
		for ch in self.channel_list:
			Traces[ch] = np.array(Traces[ch],dtype=float)
			if self.ADC2A != 1.0:
				Traces[ch] *= self.ADC2A
#		 if "TTL" in self.channel_list:
#			 # Remove TTL channel from returned traces
#			 Traces.pop("TTL")
			
		self.Traces = Traces
		return Traces, trigger_result






# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def apply_autocuts(traces, filter_kernel=None, n_iterations=2, thresholds=None):
	# threshold - an additional cut. Filtered traces that are above the threshold are removed
	good_traces = {}
	acceptances = {}
	for ch in traces:
		t = traces[ch]
		if filter_kernel is None:
			filtered_traces = np.copy(t)
		else:
			# mode=valid: we don't use zero padding here, because it messes with the range parameter in the autocuts
			filtered_traces = np.apply_along_axis(lambda m: scipy.signal.correlate(m, filter_kernel, mode='valid'), axis=1, arr=t)
			
		if thresholds is not None and ch in thresholds:
			mask = filtered_traces.max(axis=1)<thresholds[ch]
			filtered_traces = filtered_traces[mask]
			selected_traces = t[mask]
		else:
			selected_traces = np.copy(t)
			
		xvals = np.arange(len(filtered_traces[0]))

		ranges = np.max(filtered_traces, axis=1) - np.min(filtered_traces, axis=1)
		slopes = np.array([slope(xvals, trace) for trace in filtered_traces])
		means = np.mean(filtered_traces, axis=1)
		flattened_traces = filtered_traces - slopes[:, np.newaxis]*xvals[np.newaxis, :]
		skewnesses = np.abs(skew(flattened_traces, axis=1))
		
		for i in range(n_iterations):
			meanIn = removeOutliers(means)
			rangeIn = removeOutliers(ranges)
			slopeIn = removeOutliers(slopes)
			skewIn = removeOutliers(skewnesses)

			inds = meanIn & rangeIn & slopeIn & skewIn
			means=means[inds]
			ranges=ranges[inds]
			slopes=slopes[inds]
			skewnesses=skewnesses[inds]
			selected_traces=selected_traces[inds]
		
		good_traces[ch] = selected_traces
		acceptances[ch] = len(selected_traces)/len(t)
	
	return good_traces, acceptances


def getPSD(Traces, Fs=625000, weight=1, channels=None, autocuts=True):
	'''
	Returns dict of dict of PSDs. Uses scipy.signal.periodogram via Noise.processNoise

	Traces: dict, default unit is uA
	Fs=625000: sampling freq
	weight: scale traces
	'''
	PSDs=dict()
	PSDs["PSD"]=dict()
	Channels = list(Traces.keys()) if channels is None else channels
	for ch in Channels:
		if ch =="TTL":
			continue
		print(ch)
		noiseResults = Noise.processNoise(Traces[ch]*weight, traceGain = 1., makePlots=False, autoCut=autocuts, fs=Fs, fileStr='_', saveResults=False, useCorrectAveraging=True)
		PSDs["f"] = noiseResults["f"]
		PSDs["PSD"][ch] = noiseResults["psdMean"]
		#print(noiseResults["psdMean"][:20])
		if np.isnan((noiseResults["psdMean"])[0]):
			print("--maybe a prebias series. Use default PSD for this file. STOP PSD making.")
			print("Default PSD: ", fallback_PSD_fname)
			PSDs = joblib.load(fallback_PSD_fname)
			break
	return PSDs

def getTemplate(Traces,trigger_result, trigger_channel_map, trigBin=4096, endBin=4096+300, amplitude_ranges=None, verbose=True):
	"""
	Make average pulse for channels with specified trigger channel
		Traces: dict
		trigger_result: trigger result from trigger object
		trigger_channel_map: {"channel_to_make_average_pulse":"corresponding_trigger_channel"}
	"""
	templates=dict()
	trigger_templates=dict()
	for ch in trigger_channel_map:
		trig_ch = trigger_channel_map[ch]
		mask = (trigger_result["trig_ch"]==trig_ch)
		if type(amplitude_ranges) is list:
			mask = mask& (trigger_result["trig_amp"]>amplitude_ranges[0])& (trigger_result["trig_amp"]<amplitude_ranges[1])
		elif type(amplitude_ranges) is dict:
			mask = mask& (trigger_result["trig_amp"]>amplitude_ranges[ch][0])& (trigger_result["trig_amp"]<amplitude_ranges[ch][1])  
			
		if sum(mask)>0:
			mean_trace = processTemplate_mod(Traces[ch][mask],autocut=True,cutIters=2,trigBin=trigBin,endBin=endBin,verbose=verbose)
			mean_trace-=mean(mean_trace[:trigBin])
			mean_trace/=max(mean_trace)
		else:
			mean_trace=np.zeros(len(Traces[ch][0]))
		templates[ch]=mean_trace
		if ch==trig_ch:
			trigger_templates[ch]=mean_trace
	return templates,trigger_templates

def getTemplate_single_trig(Traces,trigger_result, trigger_channel_map, trigBin=4096, endBin=4096+300, amplitude_ranges=None, verbose=True):
	"""
	Make average pulse for all channels based on a single trigger channel
		Traces: dict
		trigger_result: trigger result from trigger object
		trigger_channel_map: {"trigger_channel":"trigger_channel"}
	"""
	templates=dict()
	trigger_templates=dict()
	for ch in trigger_channel_map:
		trig_ch = trigger_channel_map[ch]
		mask = (trigger_result["trig_ch"]==trig_ch)
		if type(amplitude_ranges) is list or amplitude_ranges is None:
			#mask = mask& (trigger_result["trig_amp"]>amplitude_ranges[0])& (trigger_result["trig_amp"]<amplitude_ranges[1])
			amplitude_range = amplitude_ranges
		#elif type(amplitude_ranges) is dict:
			#mask = mask& (trigger_result["trig_amp"]>amplitude_ranges[ch][0])& (trigger_result["trig_amp"]<amplitude_ranges[ch][1])  
	
	for ch in Traces:
		if sum(mask)>0:
			if type(amplitude_ranges) is dict:
				amplitude_range = amplitude_ranges[ch]
			mean_trace = processTemplate_mod(Traces[ch][mask],autocut=True,cutIters=2,trigBin=trigBin,endBin=endBin,amplitude_range=amplitude_range,verbose=verbose)
			mean_trace-=mean(mean_trace[:trigBin])
			#mean_trace/=max(mean_trace)
		else:
			mean_trace=np.zeros(len(Traces[ch][0]))
		templates[ch]=mean_trace
		if ch==trig_ch:
			trigger_templates[ch]=mean_trace
	return templates,trigger_templates

def getOptimalFilter(self, s, rtJ):
	# Noah formalism
	# s : trigger template
	# rtJ : sqrt(J) = PSD = output of np.fft.rfft
	J = np.abs(rtJ)**2
	J[0]=np.inf
	sf = np.fft.rfft(s) # s, Fourier-transformed
	phi = np.conjugate(sf)/J # optimal filter
	denom = np.real(phi.dot(sf))
	phi_prime = phi/denom

	#A = phi_prime.dot(vf)

def append_dicts(dict1,dict2):
	dict_combined=dict()
	for key in dict1:
		dict_combined[key]=np.append(dict1[key],dict2[key])
	return dict_combined

def processTemplate_mod(rawTraces,autocut=True,cutIters=2,trigBin=450,endBin=2000,amplitude_range=None,verbose=False):

	x=np.arange(0,len(rawTraces[0]))

	nonTrace = np.logical_or(x < trigBin, x > endBin)
	xNT = x[nonTrace]
	xSlope= xNT - np.mean(xNT)

	#for storing results
	skewnesses=list()
	means=list()
	ranges=list()
	slopes=list()
	traces=list()
	amps=list()

	for trace in rawTraces:
		tt=trace[nonTrace]#*1e6 #current in microAmps
		ranges.append(max(tt)-min(tt))

		ntMean=np.mean(tt)
		means.append(ntMean)

		ntSlope = slope(xSlope,tt-ntMean,removeMeans=False)
		slopes.append(ntSlope)

		skewness=skew(tt)
		skewnesses.append(abs(skewness))
		
		amps.append(max(trace)-ntMean)

		traces.append(trace)

	#convert to np structures
	traces=np.array(traces)
	means=np.array(means)
	skewnesses=np.array(skewnesses)
	slopes=np.array(slopes)
	ranges=np.array(ranges)
	amps=np.array(amps)

	if(autocut):
		slopes_all=slopes
		means_all=means
		skews_all=skewnesses
		ranges_all=ranges
		for i in range(0,cutIters):
			#make cuts
			meanIn = removeOutliers(means)
			rangeIn = removeOutliers(ranges)
			slopeIn = removeOutliers(slopes)
			skewIn = removeOutliers(skewnesses)
			ampsIn = (amps>amplitude_range[0])&(amps<amplitude_range[1]) if amplitude_range is not None else np.repeat(True, len(meanIn))

			inds = meanIn & rangeIn & slopeIn & skewIn & ampsIn
			amps=amps[inds]
			means=means[inds]
			ranges=ranges[inds]
			slopes=slopes[inds]
			skewnesses=skewnesses[inds]
			traces=traces[inds]

	tmean=np.mean(means)
	template=np.mean(traces,axis=0)
	if verbose:
		print('Acceptance',float(len(traces))/float(len(rawTraces)))

	return template

'''	
def selectFromList(data_list_filename, conditions, precise = True, remove_unprocessed=False, processed_dir = '/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/AnimalData/AR70/processing_auto/'):
	data_list = pd.read_csv(data_list_filename, skipinitialspace=1,skiprows=1)
	data_list_length = len(data_list[data_list.keys()[0]])
	#date = [str(data_list["Series"][i])[:8] for i in range(data_list_length)]
	#data_list["Date"] = date	
	data_list_mask = pd.DataFrame()
	data_list_length = len(data_list["Series"])
	mask = np.ones(data_list_length)
	if conditions=="all":
		pass
	else:
		for entry in conditions:
			if precise:
				data_list_mask[entry]=[data_list[entry][i]==conditions[entry] for i in range(data_list_length)]
			else:
				if type(conditions[entry])==str:
					data_list_mask[entry]=[(conditions[entry] in data_list[entry][i]) for i in range(data_list_length)]			
				else:
					data_list_mask[entry]=[data_list[entry][i]==conditions[entry] for i in range(data_list_length)]			
			mask = mask * data_list_mask[entry]
	maskIndex = np.flatnonzero(mask)			
#	 return maskIndex, data_list

	# let's find out what hasn't been processed either
	unprocessed_series = []
	
	conditioned_series_list = data_list.loc[maskIndex].loc[:,"Series"].values
	for series in conditioned_series_list:
		if not os.path.exists(processed_dir+'OFResults_'+str(int(series))+'.pkl'):
			unprocessed_series.append(str(int(series)))
	
	
	return maskIndex, data_list, unprocessed_series		


def fitGaus_from_hist(n,ibins,fitrange=None,poissonerror=False,interp_points=400,p0=None,make_plot=False):
	bincenters=np.array(0.5*(ibins[1:]+ibins[:-1]))
	if fitrange is None:
		fitrange_idx=[0,-1]
	else:
		fitrange_idx=[np.argmax(bincenters>fitrange[0]),np.argmax(bincenters>fitrange[1])]
	fit_x=bincenters[fitrange_idx[0]:fitrange_idx[1]]
	fit_y=n[fitrange_idx[0]:fitrange_idx[1]]
	fit_err=np.sqrt(fit_y) if poissonerror else np.ones(len(fit_y))
	
	if p0 is None:
		ymean = sum(fit_x * fit_y) / sum(fit_y)
		sigma = np.sqrt(sum(fit_y * (fit_x - ymean)**2) / sum(fit_y))
		p0=[max(fit_y), ymean, sigma]
	popt,pcov=scipy.optimize.curve_fit(Gauss,fit_x,fit_y,p0=p0)
	
	plot_x=np.linspace(min(fit_x),max(fit_x),interp_points)
	plot_y=Gauss(plot_x,*popt)
	
	if make_plot:
		step(fit_x,fit_y,where="mid")
		plot(plot_x,plot_y)
	
	return popt,pcov,plot_x,plot_y,bincenters,n		


def hl_envelopes_idx(s,dmin=1,dmax=1):
	"""
	s : 1d-array, data signal from which to extract high and low envelopes
	dmin, dmax : int, size of chunks, use this if size of data is too big
	"""

	# locals min	  
	lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
	# locals max
	lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 

	"""
	# the following might help in some case by cutting the signal in "half"
	s_mid = np.mean(s) (0 if s centered or more generally mean of signal)
	# pre-sort of locals min based on sign 
	lmin = lmin[s[lmin]<s_mid]
	# pre-sort of local max based on sign 
	lmax = lmax[s[lmax]>s_mid]
	"""

	# global max of dmax-chunks of locals max 
	lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
	# global min of dmin-chunks of locals min 
	lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]

	return lmin,lmax


def smoothPSD(PSDs,freq_limit=2e3):
	PSDs_new = copy.deepcopy(PSDs)
	for ch in PSDs["PSD"]:
		psd_l_inds,_=hl_envelopes_idx(PSDs["PSD"][ch])
		psd_l_inds=np.append(psd_l_inds,[1,2])
		x,y = PSDs["f"][psd_l_inds],PSDs["PSD"][ch][psd_l_inds]
		l_p = scipy.interpolate.interp1d(x,y,bounds_error = False, fill_value=0.0)
		y_interpreted = l_p(PSDs["f"])
		PSDs_new["PSD"][ch][1:np.argmax(PSDs["f"]>freq_limit)]=y_interpreted[1:np.argmax(PSDs["f"]>freq_limit)]
	return PSDs_new
'''


