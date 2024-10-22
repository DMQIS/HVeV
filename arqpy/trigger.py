'''
Trigger and loading functions

Routines in this module:

...

modified version of `nexus_processing_alt/Nexus_utils.py`
'''

import os

import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.stats import skew

from .util import *

'''
# CDMS-specific
try:
	import cdms
	from scdmsPyTools.Cuts.General import removeOutliers
	from scdmsPyTools.Traces.Stats import slope
	import scdmsPyTools.Traces.Noise as Noise
except ImportError:
	print('WARNING: Missing CDMS library')
'''

class Trigger:
	def __init__(self, mode=1, data=None, chs=None, trigger_chs=None, 
				 threshold=None, deactivation_threshold=None,
				 fsamp=625000, pretrig=4096, posttrig=4096, ADC2A=1.0, detector=1,
				 filters=[], usegaus=False, sigmas=None, usewindow=False,
				 randomrate=0, remove_filter_offset=True):
		#		 BYPASS_HIGH_ENERGY_IN_FILTER = False,
		#		 align_max = True, OF_LPF=-1, verbose=True, data_type=None):
		'''
		Class for simulated trigger. Finds pulses and cuts them out full trace

		data: files list OR numpy array of traces
		channel_config: dict
				channel information
		trigger_channels: list
				channels to trigger on. If a new channel needs to be made, use syntax new_channel_name=chA_name+chB_name+..., e.g. "Total_NFC = PES2+PFS2"
		threshold: list of float number, or list of string
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
		filters: list 
		align_max: boolean
				Align the trigger point to the maximum after the threshold-crossing
		remove_filter_offset: remove time offset induced by filter (mode 1 only)
		'''
		
		# parse arguments
		self.mode = mode
		self.data = data
		self.chs = chs
		self.trigger_chs = trigger_chs
		self.threshold = threshold
		self.deactivation_threshold = deactivation_threshold
		self.fsamp = fsamp
		self.pretrig = pretrig
		self.posttrig = posttrig
		self.ADC2A = ADC2A
		self.detector = detector
		self.randomrate = randomrate
		self.remove_filter_offset = remove_filter_offset
		self.pileup_window = 250
		self.align_max = False
		# more argument handling
		if trigger_chs is None:
			if chs is not None:
				self.trigger_chs = chs
			else:
				print('ERROR: No trigger channels provided')
				return -1
		if mode not in [0,1,2,3]:
			print(f'ERROR: Unrecognized trigger mode {mode}')
			return -1
		# construct filters
		if mode == 1: # filter
			if usegaus:
				self.filters = filters
				# sigmas
				#self.filter_kernels = [self._make_filter_kernel(pre_trig_kernel, post_trig_kernel, USE_GAUS_KERNEL=USE_GAUS_KERNEL, gaus_sigma=gauss_sigma)\
				#					   for i in range(len(self.trigger_channels))]
				# Normalize kernels with templates, if templates are given
				#if filter_kernels is not None:
				#	for i, ch in enumerate(self.trigger_channels):
				#		if self.trigger_type[i] == 1 and ch != 'TTL':
				#			self.filters[i] = self.filters[i]/np.max(np.correlate(self.filters[i],filter_kernels[i]))
			else:
				# enforce one filter per trigger channel
				if len(filters) != len(self.trigger_chs):
					print('ERROR: Must provide 1 filter per trigger ch')
					return -1
				self.filters = filters
			'''
			if usewindow:
				window_filters = []
				for f in filters:
					window_filters.append()
			'''
				
			# trigger offsets
			self.trigger_offsets = {}
			if self.remove_filter_offset:
				for ch in self.trigger_chs:
					filtered_trace = self._apply_filter(self.filters[ch],self.filters[ch])
					'''
					rfothreshold = np.max(filtered_template)*0.8
					trigger_points = self._threshold_trigger(filtered_trace,
						rfothreshold, rising_edge=True, align_max=self.align_max)
					if len(trigger_points)<1:
						offset = 0
					else:
						offset = self.pretrig - trigger_points[0]
					'''
					offset = np.argmax(filtered_trace)
					self.trigger_offsets[ch] = offset
			else:
				for ch in self.trigger_chs:
					self.trigger_offsets[ch] = 0.0

		return

	def _apply_filter(self, trace, kernel):
		filtered_trace = np.correlate(trace, kernel, mode='same')
		return filtered_trace
	
	def _combine_gaus_filter(self, data_filtered, data_filtered_gaus, trigger_channel_list, trigger_type, pre_trig, post_trig, threshold, n_threshold=[8]):
		"""
		Combine the OF filtered trace with the gaus-filtered trace, with the threshold of n_threshold times the threshold
		"""
		data_filtered_combined = dict()
		n_threshold = np.repeat(n_threshold[0], len(threshold)) if len(n_threshold)==1 else n_threshold
		for i, ch in enumerate(trigger_channel_list):
			if trigger_type[i]!=1 or ch=="TTL":
				data_filtered_combined[ch] = data_filtered[ch]
			else:
				data_filtered_combined[ch] = data_filtered[ch]
				
				# Trigger with the gaus-filtered trace at a relative high threshold
				trig_th = threshold[i]*n_threshold[i]
				for i_trace,trace in enumerate(data_filtered_gaus[ch]):
					trigger_points, trigger_amps, trigger_width = self._threshold_trigger(trace, trig_th, rising_edge=True,
						   align_max=True, deactivate_th=None, peak_search_window_limit=self.post_trig)
					# If an event is triggered, replace the OF filtered trace with the Gaus filtered trace.
					if len(trigger_points)>0:
						for trigpt in trigger_points:
							data_filtered_combined[ch][i_trace][trigpt-pre_trig:trigpt+post_trig] = data_filtered_gaus[ch][i_trace][trigpt-pre_trig:trigpt+post_trig]
				
		return data_filtered_combined
		

	# Trigger function		
	def runTrigger(self):
		# load traces into dict
		if type(self.data) is dict:
			traces = self.data
			nevents = len(traces[0])
			tracelen = len(traces[0][0])
			self.tracelen = tracelen
		elif type(self.data) is np.ndarray:
			traces = {}
			chs = self.chs # implicitly requires trigger_chs subset of chs
			for i in range(len(chs)):
				traces[chs[i]] = self.data[i]
			nevents = len(self.data[0])
			tracelen = len(self.data[0][0])
			self.tracelen = tracelen
		elif type(self.data) is list:
			loadtrig = (self.mode == 3)
			chs = list(set(self.chs+self.trigger_chs)) # all chs, no duplicates
			events = loadEvents(files=self.data,detectors=[self.detector],
								chs=chs,ADC2A=self.ADC2A,loadtrig=loadtrig)
			traces = events[self.detector]
			nevents = len(traces[self.trigger_chs[0]]) # ch0 traces
			tracelen= len(traces[self.trigger_chs[0]][0]) #ch0 trace0
			self.tracelen = tracelen

		# get trigger points
		mode = self.mode
		triggered_traces = {}
		for ch in self.chs:
			triggered_traces[ch] = []
		triggered_traces['event_num'] = []
		print(f'Triggering on {nevents} events')
		for i in range(nevents):
			if i % 100 == 0:
				print(i,'/',nevents)
			for ch in self.trigger_chs:
				trace = traces[ch][i]
				if mode == 0: # trigger
					trigger_points = self._threshold_trigger(trace,self.threshold[ch],trigger_offset=self.trigger_offsets[ch])
				elif mode == 1: # filter + trigger
					filtered_trace = self._apply_filter(trace,self.filters[ch])
					trigger_points = self._threshold_trigger(filtered_trace,self.threshold[ch],trigger_offset=self.trigger_offsets[ch])
				elif mode == 2: # randoms
					trigger_points = self._random_trigger()
					break # don't need to repeat for each channel
				elif mode == 3: # external
					trigger_points = np.array(events['triggers']*self.fsamp,dtype=int)
					break # don't need to repeat for each channel
			# remove trigger points too close to the edge
			mask = (trigger_points>self.pretrig)&(trigger_points<tracelen-self.posttrig)
			trigger_points = trigger_points[mask]
			'''# handle pileup
			pileup_window = self.pileup_window
			if reject_pileup:
				# Detect trigger pileup within the OFP window
				if len(trigger_points)>=1:
					trigger_dt = np.diff(trigger_points)
					if len(trigger_dt)>=2:
						trig_pileups = np.concatenate(([trigger_dt[0]<pileup_window],(trigger_dt[:-1]<pileup_window)|(trigger_dt[1:]<pileup_window),[trigger_dt[-1]<pileup_window]))
					elif len(trigger_dt)==1:
						trig_pileups = np.concatenate(([trigger_dt[0]<pileup_window],[trigger_dt[0]<pileup_window]))
					elif len(trigger_dt)==0:
						trig_pileups = [False]
				else:
					trig_pileups=[]
			'''

			# slice + dice
			triggered_traces['event_num'] += [i]*len(trigger_points)
			for trigger_point in trigger_points:
				for ch in self.chs:
					trace = traces[ch][i][trigger_point-self.pretrig:trigger_point+self.posttrig]
					triggered_traces[ch].append(trace)
		print('Found {0}'.format(len(triggered_traces[ch])))	

		return triggered_traces

	def _threshold_trigger(self, trace, threshold, rising_edge=True, 
						   deactivation_threshold=None, align_max=True,
						   peak_search_window_limit=4096, trigger_offset=0):
		# elegant, one-line, constant-threshold discriminator
		if rising_edge:
			trigger_points=np.flatnonzero((trace[0:-1]<threshold)&(trace[1:]>=threshold))+1
		else:
			trigger_points=np.flatnonzero((trace[0:-1]>threshold)&(trace[1:]<=threshold))+1

		# implement deactivation + align_max
		# TODO: assumes rising_edge
		if deactivation_threshold is not None:
			deactivation_window = 0
			new_trigger_points = []
			# for each point, reject points before deactivation
			for tp in trigger_points:
				# if in window, skip. otherwise, add then adjust window
				if tp < deactivation_window:
					continue
				else:
					if align_max:
						tpmax = tp + np.argmax(trace[tp:deactivation_window])
						new_trigger_points.append(tpmax)
					else:
						new_trigger_points.append(tp)
					deactivation_window = np.argmax(trace[tp:]<deactivation_threshold)
					if deactivation_window > peak_search_window_limit:
						deactivation_window = peak_search_window_limit
					deactivation_window += tp
		elif align_max: # similar
			new_trigger_points = []
			for tp in trigger_points:
				tpmax = tp + np.argmax(trace[tp:tp+peak_search_window_limit])
				new_trigger_points.append(tpmax)
			new_trigger_points = np.unique(new_trigger_points)

		# trigger offsets
		trigger_points -= trigger_offset
		
		return trigger_points
	
	def _random_trigger(self):
		randomrate = self.randomrate
		trigger_points = np.random.randint(self.pretrig+1,self.tracelen-self.posttrig,size=randomrate)
		return trigger_points

'''
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


