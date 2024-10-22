'''
Reduced Quantity object, containing filter functions

Routines in this module:

fft(a, n=None, axis=-1, norm="backward")
ifft(a, n=None, axis=-1, norm="backward")
rfft(a, n=None, axis=-1, norm="backward")
irfft(

modified version of `nexus_processing_alt/Nexus_RQ.py`
'''

#__all__ = ['fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'rfftn',
#           'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy

from .util import *
from .trigger import Trigger

#from scdmsPyTools.Traces.Filtering import *
#from scdmsPyTools.Traces.Stats import *


class RQ: 
	def __init__(self, data=None, traces=None, chs=None, ch_names=None, detector=1,
				 fsamp=625000, pretrig=4096, posttrig=4096, ADC2A=1.0,
				 PSDs=None, pulse_templates=None,
				 max_chi2_freq=None, lpfs=None,
				 OFL=True, OFL_max_delay=None,
				 OFP=True, OFP_window=250,
				 baselength=4086, taillength=3096, maxchi2freq=None,
				 PlateauDelay=100, PlateauLength=100,
				 UseFilterForRQs=False, CutoffFrequenciesForRQs=None, FilterOrderForRQs=10,
				 WindowForPulseRQs=100, saturation_amplitude=1.0,
				 trigger_chs=None, randomrate=None,
				 threshold=None, deactivation_threshold=None):
		'''
		Class for RQ calculations
		fsamp: sampling frequency (Hz)
		pretrig/posttrig: bins to use before/after the pulse leading-edge
		ADC2A: scale factor to apply to traces, before processing
		       used to convert ADC units to Amperes
		max_chi2_freq: max frequency to use in chi2 computations (Hz)
		               used to ignore HF noise/cutoffs/etc that would ruin chi2
		LPF_freqs: dict of max freq bins to use, for low-pass filtering
		OFL: option to compute OF with optimized pulse delay
		OFL_max_delay: max pulse delay to consider, in bins
		OFP: option to compute OF with pileup handling
		OFP_window

		baselength: first N trace values to use for baseline RQs
		taillength: last N trace values to use for tail RQs
		maxchi2freq: highest frequency to use for chi2 calculation

		data: dict. from `loader`
		'''
		self.fsamp = fsamp
		self.pretrig = pretrig
		self.posttrig = posttrig
		self.ADC2A = ADC2A
		self.lpfs = None
		self.OFL = OFL
		self.OFL_max_delay = OFL_max_delay
		self.OFP = OFP
		self.OFP_window = OFP_window
		self.chs = chs
		self.ch_names = ch_names
		self.data_type = 'SLAC' # default
		self.detector = detector
		self.baselength = baselength
		self.taillength = taillength
		self.maxchi2freq = maxchi2freq
		# optional trigger config
		self.trigger_chs = trigger_chs
		self.randomrate = randomrate
		self.threshold = threshold
		self.deactivation_threshold = deactivation_threshold

		#  initial data-handling
		self.data = data
		self.setPSD(PSDs,chs)
		self.setTemplates(pulse_templates,chs)

		# trace/PSD math
		self.tracelen = pretrig + posttrig
		self.frequencies = np.fft.rfftfreq(self.tracelen,1/fsamp)
		if maxchi2freq is not None:
			self.maxchi2bin = np.argmax(self.frequencies>=maxchi2freq)
		else:
			self.maxchi2bin = None
		'''
		if LPF_freqs is not None:
			self.LPF_bins = {}
			for ch in LPF_freqs.keys():
				self.LPF_bins[ch] = np.argmax(self.frequencies>=LPF_freqs[ch])
		else:
			self.LPF_bins = None
		'''

		# for stat descript of "baseline" and "tail"
		self.PlateauDelay = PlateauDelay # time between the trigger location and the start of the window, in which PlateauAmplitude is calculated
		self.PlateauLength = PlateauLength # length of the PlateauAmplitude window [samples]
		self.WindowForPulseRQs = WindowForPulseRQs
		if type(saturation_amplitude) in [int,float]:
			self.saturation_amplitude = {}
			for ch in chs:
				self.saturation_amplitude[ch] = saturation_amplitude
		self.MF_threshold = 0.2
		
		# These setting are for filtering raw traces with Butterworth filter
		# before calculating some of the RQs.
		# The filter is not applied for the OF RQs!
		self.UseFilterForRQs = UseFilterForRQs 
		self.CutoffFrequenciesForRQs = CutoffFrequenciesForRQs
		self.FilterOrderForRQs = FilterOrderForRQs

		# storage for the actual RQs!
		self.RQs = {}


	# set PSD dictionary
	def setPSD(self,psds,chs=None):
		# chs: list of channel names
		# psds: list of PSDs
		if type(psds) is dict:
			self.PSDs = psds
		elif chs is not None: # handle array of arrays
			self.PSDs = {}
			for i in range(len(chs)):
				self.PSDs[chs[i]] = psds[i]


	# set pulse template dictionary
	def setTemplates(self,templates,chs=None):
		# chs: list of channel names
		# templates: list of pulse templates
		if type(templates) is dict:
			self.templates = templates
		elif chs is not None: # handle array of arrays
			self.templates = {}
			for i in range(len(chs)):
				self.templates[chs[i]] = templates[i]


	# Compute optimal filter components, and store
	def makeOF(self,chs=None,templates=None,psds=None,lpfs=None):
		# use class variables if not provided as arguments
		if chs is None:
			chs = self.chs
		if templates is None:
			templates = self.templates
		if psds is None:
			psds = self.PSDs
		if lpfs is None:
			lpfs = self.lpfs

		# Golwala-Kurinsky notation. cf. Shutt 1993, Zadeh & Ragazzini 1952
		phis = {} # optimal filters, un-normalized
		norms = {} # OF norms
		phi_primes = {} # OFs
		sfs = {} # Fourier-transformed pulse templates
		Js = {} # noise PSDs
		filter_kernels_varPSD = {} # ?
		filter_kernels_varPSD_limited = {} # ?
		for ch in chs:
			J = np.array(self.PSDs[ch]) # copy before modifying
			J[0] = np.inf # ignore DC component
			if self.lpfs is not None: # low-pass filter, optionally
				max_freq_bin = self.lpfs[ch]
			else:
				max_freq_bin = len(J)
			s = self.templates[ch]
			sf = np.fft.rfft(s) # shortcut for real-valued input
			sf[max_freq_bin:] = 0.0
			phi = np.conjugate(sf)/J
			norm = np.real(np.sum(phi*sf))
			# TODO: allow chi2 max freq bin, to ignore high freq discrepancies
			#if self.max_chi2_freq_bin is not None:
			#	OF_norm_for_chi2_rq = OF_norm 
			#else:
			#	OF_norm_for_chi2_rq = np.real(np.sum((OF*trigTemplate_fft)[:self.MAX_CHISQ_FREQ_BIN]))
			phi_prime = phi/norm

			phis[ch] = phi
			norms[ch] = norm
			phi_primes[ch] = phi_prime
			sfs[ch] = sf
			Js[ch] = J

			# TODO: time-domain matched filter stuff
			# The variance of noise in time domain. Used in time-domain chi2
			filter_kernels_varPSD[ch] = (np.fft.irfft(psds[ch]**2)*self.fsamp)[0]
			if max_freq_bin != len(J):
				filter_kernels_varPSD_limited[ch] = filter_kernels_varPSD[ch]
			else:
				psd_lim = np.copy(psds[ch])
				psd_lim[max_freq_bin:]=0
				filter_kernels_varPSD_limited[ch] = (np.fft.irfft(psd_lim**2)*self.fsamp)[0]
			
		self.phis = phis
		self.norms = norms
		self.phi_primes = phi_primes
		self.sfs = sfs
		self.Js = Js

		self.filter_kernels_varPSD = filter_kernels_varPSD #
		self.filter_kernels_varPSD_limited = filter_kernels_varPSD_limited #
		return

	# get dict of theoretical resolutions
	def getTheoryRes(self):
		theores = {}
		for ch in self.chs:
			sf = self.sfs[ch]
			J = self.Js[ch]
			varA = 0.5/np.sum(np.abs(sf)**2/J)
			sigA = np.sqrt(varA)
			theores[ch] = sigA
		return theores

	# run various OFs on a single trace
	#   implemented this way to reduce repeated operations
	def _runOFsingle(self,trace,norm,sf,phi_prime,J,chi2mask,max_freq_bin=None,OFL=True,OFP=False):
		# setup
		OF_RQs = {}
		N = float(len(trace))
		dt = 1/self.fsamp

		# Fourier-transform trace
		vf = np.fft.rfft(trace)
		OFtrace = phi_prime*vf # the filtered trace (in Fourier space)

		# 1. OF without delay (OF0_*)
		A_nodelay = np.real(np.sum(OFtrace))
		chi0 = np.sum(np.abs(vf[chi2mask])**2/J[chi2mask]) # first part of chi2
		chi2_nodelay = 2 * (chi0-np.abs(A_nodelay)**2*norm)
		OF_RQs['OF0_A'] = A_nodelay
		OF_RQs['OF0_chi2'] = chi2_nodelay
	
		# 2. OF with delay (OF_*), and with limited delay (OFL_*)
		if OFL:
			#  Calculate A(t0) for all t0 in one step:
			As = N/2*np.fft.irfft(OFtrace)
			chi2s = 2 * (chi0-np.abs(As)**2*norm)
			#  then find best t0
			ind  = np.argmin(chi2s)
			t0   = ind * dt
			A    = As[ind] # = np.sum(np.exp(1j*2*np.pi*f*t0)*phi*vf) / norm
			chi2 = chi2s[ind]
			OF_RQs['OF_time'] = t0
			OF_RQs['OF_A'] = A
			OF_RQs['OF_chi2'] = chi2
			# limit max delay
			if self.OFL_max_delay is not None:
				indL = np.argmin(chi2s[:self.OFL_max_delay])
				OF_RQs['OFL_time'] = indL * dt
				OF_RQs['OFL_A'] = As[indL]
				OF_RQs['OFL_chi2'] = chi2s[indL]
		
		# 3. Pile-up OF (OFP_*)
		# TODO: Jamie did not go through what this does yet
		if OFP:
			OFP_window = self.OFP_window
			print('WARNING: OFP not yet implemented')
			'''
			# Limit the search window to +/-250 points around middle of trace
			# Runtime is 0.004 s for 250 points window.
			# I don't understand why 256 is x3 to x4 times slower than 250...

			#half_tracelength=int(Ns//2)
			#T = self.avg_pulses[ch][half_tracelength-pileup_window:half_tracelength+pileup_window]
			#NoisePSD = scipy.signal.resample(self.PSDs[ch],len(T)//2+1)*1e-6 # Turn unit into uA; resample; note the LPF setting also need resample
			#Signal = trace[half_tracelength-pileup_window:half_tracelength+pileup_window]
			#A1s,A2s,t1,t2,Xrp = OptimumFilterAmplitude_Pileup(Signal,T,NoisePSD,self.fsamp,downSample=1,LPF=int(OF_LPF/(half_tracelength/pileup_window)))
			#t1,t2 = int((t1*self.fsamp+pileup_window)%int(2*pileup_window)-pileup_window),int((t2*self.fsamp+pileup_window)%int(2*pileup_window)-pileup_window) # Turn the unit into samples and centered at 0
			# Use the modified version
			NoisePSD = np.copy(self.PSDs[ch])
			Signal = trace
			T = self.avg_pulses[ch]
			A1s,A2s,t1,t2,Xrp,Xrp_for_chi2_rq = self.OptimumFilterAmplitude_PileupMod(Signal,T,NoisePSD,downSample=1,LPF=int(OF_LPF),delayMax=self.pileup_window)
			# Turn the unit into samples and centered at 0
			t1,t2 = np.rint(t1*self.fsamp), np.rint(t2*self.fsamp)
			# store		
			OF_RQs['OFP_A1'] = A1s
			OF_RQs['OFP_A2'] = A2s
			OF_RQs['OFP_time1'] = t1
			OF_RQs['OFP_time2'] = t2
			OF_RQs['OFP_chi2'] = Xrp
			OF_RQs['OFP_chi2_full'] = Xrp_for_chi2_rq
			'''

		return OF_RQs


	# use Trigger class to get short traces (containing pulses, usually) 
	def runTrigger(self,**kwargs):
		# Trigger keywords:
		keys = ['mode','data','chs','trigger_chs','detector','fsamp','ADC2A','pretrig','posttrig',
		        'randomrate','filters','window','usegaus','sigmas','trigger_points',
		        'threshold','deactivation_threshold']
		trig_kwargs = {} # trigger key word arguments
		for key in kwargs:
			trig_kwargs[key] = kwargs[key]
		# inherit trigger settings from RQ class, if not yet defined
		for key in keys:
			if key not in trig_kwargs:
				if hasattr(self,key):
					trig_kwargs[key] = getattr(self,key)
		# if no filter defined, try using OF
		if 'filters' not in trig_kwargs:
			if hasattr(self,'phi_primes'):
				filters = {}
				for ch in self.phi_primes:
					filt = np.fft.irfft(self.phi_primes[ch].conjugate())
					filt *= len(self.phi_primes[ch]) # (re)normalize
					filters[ch] = filt
				trig_kwargs['filters'] = filters
		tg = Trigger(**trig_kwargs)
		traces = tg.runTrigger()
		self.traces = traces
		return
	
	# TODO: what is this
	def matchedFilter(self,S,T,threshold,cut=0):
		A0=np.dot(T,T)
		if type(cut)==np.ndarray:
			mask=(S<threshold)&cut
		else:
			mask=(S<threshold)
		Adot=np.dot(T[mask],S[mask])
		A0dot=np.dot(T[mask],T[mask])
		A_t=Adot/A0dot
		return A_t


	
	# A modified version from scdmsPyTools
	# TODO: what is this
	# uses `timeMatrix` and `timeShiftMatrix`
	'''
	def OptimumFilterAmplitude_PileupMod(self, Signal, Template, NoisePSD, downSample=8, LPF=-1, delayMax=250):

		dt = 1.0/self.fsamp
		Ns = float(len(Signal))
		T = Ns*dt
		dnu = 1.0/T

		#take one-sided fft of Signal and Template
		Sf = np.fft.rfft(Signal)
		Tf = np.fft.rfft(Template)

		#check for compatibility between PSD and fft
		if(len(NoisePSD) != len(Sf)):
			raise ValueError("PSD length incompatible with signal size")

		chi2LPF_SF = 1.
		if (LPF > 0) and (LPF < len(NoisePSD)):
			NoisePSD[LPF:]=np.maximum(NoisePSD[LPF:],NoisePSD[LPF])
			Sf[LPF:] = 0
			Tf[LPF:] = 0
			chi2LPF_SF = float(LPF)/len(NoisePSD)
			
		chiScale=4*dt/(Ns**2) / chi2LPF_SF
		
		if self.MAX_CHISQ_FREQ_BIN>=0 and (self.MAX_CHISQ_FREQ_BIN<LPF or LPF<0):
			chi2LPF_SF = float(self.MAX_CHISQ_FREQ_BIN)/len(NoisePSD)
			
		#this factor is derived from the need to convert the dft to continuous units, and then get a reduced chi-square
		chiScale_for_chi2_rq=4*dt/(Ns**2) / chi2LPF_SF
			
		#take squared noise PSD
		J=NoisePSD**2.0

		#TEMPORARY: NEED TO SWITCH TO FULL FFT
		J[0]=np.inf

		#find optimum filter and norm
		OF = Tf.conjugate()/J
		Norm = np.real(OF.dot(Tf))
		Norm_for_chi2_rq = Norm if self.MAX_CHISQ_FREQ_BIN<0 else np.real(np.sum((OF*Tf)[:self.MAX_CHISQ_FREQ_BIN]))
		OFp = OF/Norm

		#filter template and trace
		Sfilt = OFp*Sf
		Tfilt = OFp*Tf

		#compute OF with delay

		#have to correct for np rfft convention by multiplying by N/2
		At = np.fft.irfft(Sfilt)*Ns/2.0
		Gt = np.real(np.fft.irfft(Tfilt))*Ns/2.0

		#signal part of chi-square
		chi0 = np.real(np.dot(Sf.conjugate()/J,Sf))
		chi0_for_chi2_rq = chi0 if self.MAX_CHISQ_FREQ_BIN<0 else np.real(np.sum((Sf.conjugate()*Sf/J)[:self.MAX_CHISQ_FREQ_BIN])) # for the chi2 RQ - can be limited to low freqs only
		
		#construct matrices for t0 in the row and delta t in the column
		ds=int(downSample)
		At0 = timeMatrix(np.concatenate((At[:delayMax:ds],At[-delayMax::ds])))
		Atd = timeShiftMatrix(np.concatenate((At[:delayMax:ds],At[-delayMax::ds])))
		GM = (timeMatrix(np.concatenate((Gt[:delayMax:ds],Gt[-delayMax::ds])))).transpose()

		#compute full solution
		A2=(Atd-At0*GM)/(1.0-GM**2+1e-99)
		A1=At0-A2*GM

		#compute chi-square
		chit = (A1**2+A2**2+2*A1*A2*GM)*Norm
		chit_for_chi2_rq = (A1**2+A2**2+2*A1*A2*GM)*Norm_for_chi2_rq

		#sum parts of chi-square
		chi = (chi0 - chit)*chiScale
		chi_for_chi2_rq = (chi0_for_chi2_rq - chit_for_chi2_rq)*chiScale_for_chi2_rq

		#find time of best-fit. Limiting it to positive amplitudes only
		mask = (A1 > 0) & (A2 > 0)
		chi[~mask] = np.inf
		chi_for_chi2_rq[~mask] = np.inf
		dti,ti = np.unravel_index(np.argmin(chi),np.shape(chi)) 
		A1s = A1[dti,ti]
		A2s = A2[dti,ti]
		Xr = chi[dti,ti]
		Xr_for_chi2_rq = chi_for_chi2_rq[dti,ti]
		dtS = float(downSample)*dt

		if delayMax:
			ti=(ti+delayMax)%(2*delayMax)-delayMax
			dti=(dti+delayMax)%(2*delayMax)-delayMax
		t1=float(ti)*dtS
		t2=t1+float(dti)*dtS

		#keep times in domain
		if(t1 >= T):
			t1-=T
		if(t2 >= T):
			t2-=T

		#first time is first amplitude
		if(t2 < t1):
			t1t=t1
			t1=t2
			t2=t1t

			A1t=A1s
			A1s=A2s
			A2s=A1t

		return A1s,A2s,t1,t2,Xr,Xr_for_chi2_rq
	'''

	'''
	# call `_runOFsingle` on all specified events/channels
	def runOF(self,chs=None):
		RQ_list=['OF0_A', 'OF0_chi2',
				'OF_A', 'OF_chi2', 'OF_time',
				'OFL_A', 'OFL_chi2', 'OFL_time',
				'OFP1','OFP2','OFP1_time','OFP2_time','OFP_chi2','OFP_chi2_full','Npileup']
		if chs is None:
			chs = self.chs

		# set up RQ arrays
		for ch in chs:
			for RQ_name in RQ_list:
				self.RQs[f'{RQ_name}_{ch}']=[]

		#
		for trace in Traces[ch]:
			continue

		phi_prime = self.phi_primes[ch]
		norm = self.norms[ch]
		sf = self.sfs[ch]
		J = np.array(self.Js[ch]) # make a copy

		# ignore high-freq bins for chi2, if desired
		chi2mask = np.ones(len(J),dtype=bool)
		if self.MAX_CHISQ_FREQ_BIN >= 0:
			chi2mask[MAX_CHISQ_FREQ_BIN+1:] = False

		# low pass filter, if desired. Recall negative OF_LPF means no LPF
		if max_freq_bin is not None:
			J_LPF = np.array
			J[max_freq_bin:] += np.inf
			# also calculate LPF trace
			vf[OF_LPF:] = 0
			trace_lpf = np.fft.irfft(trace_freq)
			OF_RQs['trace'] = trace_lpf
		else:
			OF_RQs['trace'] = trace

		OF_results_ch = self._runOFsingle(trace, ch, OF_LPF=self.OF_LPF_dict[ch], pileup_window=self.OFP_WINDOW)

		for key in RQlist:
			if key in OF_results_ch:
				OF_results[f'{key}_{ch}'].append(OF_results_ch[key])
		if 'total' in ch.lower():
			for key in RQlist_total:
				if key in OF_results_ch:
					OF_results[f'{key}_{ch}'].append(OF_results_ch[key])
	'''


	def processTraces(self):
		# setup
		traces = self.traces
		pretrig = self.pretrig
		baselength = self.baselength
		taillength = self.taillength
		tlength = self.pretrig + self.posttrig # trace length, in samples
		tlength_2 = tlength / self.fsamp / 2 # half trace length, in seconds

		OF_results = dict()
		RQlist=['OF0_A', 'OF0_chi2',
				'OF_A', 'OF_chi2', 'OF_time',
				'OFL_A', 'OFL_chi2', 'OFL_time',
				#'OF_chi2time', 'OF0_chi2timeFiltered', 'OF0_chi2timeShort',
				'MF', 'Amplitude', 'Max', 'MaxHead', 'MaxTail', 'Integral', 'IntegralHead', 'IntegralTail', 'BaselineSlope', 'Slope',
				'PlateauAmplitude', 'RiseTime1', 'RiseTime2', 'RiseTime3', 'FallTime1', 'FallTime2', 'FallTime3', 'PulseWidth50',
				'PulseMaxInd', 'MeanBase', 'BaselineVariance', 'TailVariance']
		RQlist_total = ['OFP1','OFP2','OFP1_time','OFP2_time','OFP_chi2','OFP_chi2_full','Npileup']
		RQlist_64bit=[]

		for ch in self.chs:
			if ch == 'event_num':
				continue
			for rq_name in RQlist:
				OF_results[f"{rq_name}_{ch}"]=[]
			if 'total' in ch.lower():
				for rq_name in RQlist_total:
					OF_results[f'{rq_name}_{ch}']=[]

			chanTemplate = self.templates[ch]
			varPSD = self.filter_kernels_varPSD[ch]
			varPSDlim = self.filter_kernels_varPSD_limited[ch] 
			
			# gaus derivative filter for the Npileup RQ
			# AZ: this looks like a pile of undocumented hardcoded numbers.
			# JR: I concur
			if 'total' in ch.lower():
				gausfilter_sigma = 8 if ch in ["NFF_total","NFE_total","NFG_total","NFC_total","NFH_total","R1_total"] else 24
				gausfilter_truncate = 2 if ch in ["NFF_total","NFE_total","NFG_total","NFC_total","NFH_total","R1_total"] else 4
				gausfilter_norm = np.max(scipy.ndimage.gaussian_filter(chanTemplate,gausfilter_sigma,order=1,truncate=gausfilter_truncate))
				peakfind_height = peak1_amplitude[ch][1]*0.5 if ch in ["NFF_total","NFE_total","NFG_total","NFC_total","NFH_total","R1_total"]\
								  else peak1_amplitude[ch][1]*0.8
				peakfind_prominence = peak1_amplitude[ch][1]*0.1 if ch in ["NFF_total","NFE_total","NFG_total","NFC_total","NFH_total","R1_total"]\
									  else peak1_amplitude[ch][1]*0.5
			

			nevents = len(traces[ch])
			print(f'Processing {ch}. {nevents} events')
			for trace in traces[ch]:
				#if i % 100 == 0:
				#	print(i,'/',nevents)
				phi_prime = self.phi_primes[ch]
				norm = self.norms[ch]
				sf = self.sfs[ch]
				J = np.array(self.Js[ch]) # make a copy

				# ignore high-freq bins for chi2, if desired
				chi2mask = np.ones(len(J),dtype=bool)
				if self.maxchi2bin is not None:
					chi2mask[maxchi2bin+1:] = False

				# low pass filter, if desired
				if self.lpfs is not None: # low-pass filter, optionally
					max_freq_bin = self.lpfs[ch]
					J_LPF = np.array
					J[max_freq_bin:] += np.inf
					# also calculate LPF trace
					vf[OF_LPF:] = 0
					trace_lpf = np.fft.irfft(vf)
					OF_results['trace'] = trace_lpf
				else:
					max_freq_bin = len(J)
					OF_results['trace'] = trace

				OF_results_ch = self._runOFsingle(trace,norm,sf,phi_prime,J,chi2mask,max_freq_bin=max_freq_bin)

				for key in RQlist:
					if key in OF_results_ch:
						OF_results[f'{key}_{ch}'].append(OF_results_ch[key])
				if 'total' in ch.lower():
					for key in RQlist_total:
						if key in OF_results_ch:
							OF_results[f'{key}_{ch}'].append(OF_results_ch[key])
				

				# compute no-delay, time-domain chi2
				'''
				model = A0*chanTemplate
				chi2_TD = np.sum((trace - model)**2)/varPSD # factor of 2 missing?
				OF_results[f'OF0_chi2time_{ch}'].append(chi2_TD)
				chi2_TDfilt = np.sum((OF_results['trace'] - model)**2)/varPSD # factor of 2?
				OF_results[f'OF0_chi2timeFiltered_{ch}'].append(chi2_TDfilt)
				short_i1 = pretrig
				short_i2 = pretrig + self.ShortChi2Length
				chi2_TDshort = np.sum((OF_results['trace'][i1:i2] - model[i1:i2])**2)/varPSDlim # factor of 2?
				OF_results[f'OF0_chi2timeShort_{ch}'].append(chi2_TDshort)
				'''
				
				# Find pileups using the Gaus derivitive filter
				if 'total' in ch.lower():
					trace_deriv =scipy.ndimage.gaussian_filter(trace_lpf,gausfilter_sigma,order=1,truncate=gausfilter_truncate)/gausfilter_norm
					peaks = scipy.signal.find_peaks(trace_deriv,height=peakfind_height,prominence=peakfind_prominence,width=15)
					OF_results['Npileup_'+ch].append(len(peaks[0])-1)
				
				# compute other RQs using some form of trace
				rqtrace = OF_results['trace']
				if self.UseFilterForRQs:
					# Using butterworth filter instead
					rqtrace = lpf(trace, self.CutoffFrequenciesForRQs[ch], self.FilterOrderForRQs, fs=self.fsamp)
				
				baseline = np.mean(rqtrace[:baselength])
				x_time = np.arange(len(trace))
				S = rqtrace - baseline
				threshold = self.saturation_amplitude[ch]*self.MF_threshold
				A = self.matchedFilter(S[pretrig:], chanTemplate[pretrig:], threshold)
				s1 = np.sum(S[S>threshold])
				s2 = A*np.sum(chanTemplate[S<threshold])
				OF_results['MF_'+ch].append(s1+s2)
				OF_results['Amplitude_'+ch].append(np.max(S))
				OF_results['MeanBase_'+ch].append(baseline)
				OF_results['BaselineVariance_'+ch].append(np.var(rqtrace[:baselength]))
				OF_results['TailVariance_'+ch].append(np.var(rqtrace[-taillength:]))
				OF_results['Max_'+ch].append(np.max(rqtrace))
				OF_results['MaxHead_'+ch].append(np.max(rqtrace[:baselength]))
				OF_results['MaxTail_'+ch].append(np.max(rqtrace[-taillength:]))				
				OF_results['Integral_'+ch].append(np.sum(S))
				OF_results['IntegralHead_'+ch].append(np.sum(S[:baselength]))
				OF_results['IntegralTail_'+ch].append(np.sum(S[-taillength:]))
				OF_results["BaselineSlope_"+ch] = np.append(OF_results["BaselineSlope_"+ch],
															fitSlope(x_time[:baselength], S[:baselength]))
				OF_results["Slope_"+ch] = np.append(OF_results["Slope_"+ch], fitSlope(x_time, S))
				OF_results['PlateauAmplitude_'+ch] = np.append(OF_results['PlateauAmplitude_'+ch],
															   np.mean(rqtrace[self.pretrig+self.PlateauDelay:self.pretrig+self.PlateauDelay+self.PlateauLength]))
				
				#Rise and Fall time
				tracemax_ind=np.argmax(S[pretrig:pretrig+self.WindowForPulseRQs])+pretrig
				trace_max = np.max(S[pretrig:pretrig+self.WindowForPulseRQs])
				
				fallingEdge30 = tracemax_ind + np.argmax(S[tracemax_ind:] < (trace_max*0.3)) # This is the falling edge when the pulse drops to 30% of max
				fallingEdge50 = tracemax_ind + np.argmax(S[tracemax_ind:] < (trace_max*0.5)) # This is the falling edge when the pulse drops to 50% of max
				fallingEdge90 = tracemax_ind + np.argmax(S[tracemax_ind:] < (trace_max*0.9)) # This is the falling edge when the pulse drops to 90% of max
				
				def get_nan_if_equal(x, y):
					return x if x!=y else np.nan
				
				# if the argmax was zero, it means that the trace never crossed the set thrshold (30,50,90 %)
				# setting the falling time to nan then
				fallingEdge30 = get_nan_if_equal(fallingEdge30, tracemax_ind)
				fallingEdge50 = get_nan_if_equal(fallingEdge50, tracemax_ind)
				fallingEdge90 = get_nan_if_equal(fallingEdge90, tracemax_ind)
				
				risingEdge30 = tracemax_ind-np.argmax(S[:tracemax_ind][::-1] < (trace_max*0.3)) # This is the rising edge when the pulse drops to 30% of max
				risingEdge50 = tracemax_ind-np.argmax(S[:tracemax_ind][::-1] < (trace_max*0.5)) # This is the rising edge when the pulse drops to 50% of max
				risingEdge90 = tracemax_ind-np.argmax(S[:tracemax_ind][::-1] < (trace_max*0.9)) # This is the rising edge when the pulse drops to 90% of max
				
				# the RQs are int16, so nans should be converted to something else, like -1
				def get_y_if_x_is_nan(x, y):
					return y if np.isnan(x) else x
				
				OF_results["RiseTime1_"+ch].append(risingEdge50-risingEdge30)
				OF_results["RiseTime2_"+ch].append(risingEdge90-risingEdge50)
				OF_results["RiseTime3_"+ch].append(tracemax_ind-risingEdge90)
				
				OF_results["FallTime1_"+ch].append(get_y_if_x_is_nan(fallingEdge30-fallingEdge50, -1))
				OF_results["FallTime2_"+ch].append(get_y_if_x_is_nan(fallingEdge50-fallingEdge90, -1))
				OF_results["FallTime3_"+ch].append(get_y_if_x_is_nan(-tracemax_ind+fallingEdge90, -1))
				
				OF_results["PulseWidth50_"+ch].append(get_y_if_x_is_nan(fallingEdge50-risingEdge50, -1))
				OF_results["PulseMaxInd_"+ch].append(tracemax_ind-pretrig)
				
				# ------------------------
				# Put your code below to add more RQs. The RQ name needs to be added to RQlist first.
				# There are three "traces":
				# 1. trace: raw trace
				# 2. rqtrace: After LPF at the OF_LPF bin or after butterworth filter, if this option is enabled
				# 3. S: rqtrace after subtracting baseline
				# ------------------------

			# Turn them in np array and 32bit float, unless specified by the RQlist_64bit
			for rq_name in RQlist:
				OF_results[f"{rq_name}_{ch}"]=np.array(OF_results[f"{rq_name}_{ch}"])
				if len(OF_results[f"{rq_name}_{ch}"])>0:
					if type(OF_results[f"{rq_name}_{ch}"][0]) is np.float64 and (rq_name not in RQlist_64bit):
						OF_results[f"{rq_name}_{ch}"] = OF_results[f"{rq_name}_{ch}"] .astype(np.float32)
			if "total" in ch or "Total" in ch:
				OF_results[f"Npileup_{ch}"]=np.array(OF_results[f"Npileup_{ch}"])
				OF_results[f"Npileup_{ch}"]=OF_results[f"Npileup_{ch}"].astype(np.int16) # -8192 to 8129
				#OF_results["OFP1_time_"+ch]  =OF_results["OFP1_time_"+ch].astype(np.int16) # -250 to 250
				#OF_results["OFP2_time_"+ch]  =OF_results["OFP2_time_"+ch].astype(np.int16) # -250 to 250				
			# Change data type to save some space
			#OF_results["OF_time_"+ch]  =OF_results["OF_time_"+ch].astype(np.int16) # -8192 to 8129
			OF_results["RiseTime1_"+ch] =OF_results["RiseTime1_"+ch].astype(np.int16) # -8192 to 8129
			OF_results["RiseTime2_"+ch] =OF_results["RiseTime2_"+ch].astype(np.int16) # -8192 to 8129
			OF_results["RiseTime3_"+ch] =OF_results["RiseTime3_"+ch].astype(np.int16) # -8192 to 8129
			OF_results["FallTime1_"+ch] =OF_results["FallTime1_"+ch].astype(np.int16) # -8192 to 8129
			OF_results["FallTime2_"+ch] =OF_results["FallTime2_"+ch].astype(np.int16) # -8192 to 8129
			OF_results["FallTime3_"+ch] =OF_results["FallTime3_"+ch].astype(np.int16) # -8192 to 8129
			OF_results["PulseWidth50_"+ch]=OF_results["PulseWidth50_"+ch].astype(np.int16) # -8192 to 8129
			OF_results["PulseMaxInd_"+ch]=OF_results["PulseMaxInd_"+ch].astype(np.int16) # -8192 to 8129
			#OF_results["OF_time_"+ch]	=OF_results["OF_time_"+ch].astype(np.int16) # -4096 to 4096
			#OF_results["OFL_time_"+ch] =OF_results["OFL_time_"+ch].astype(np.int16) # -250 to 250 
		OF_results['event_num'] = traces['event_num'] # JR addition
		self.results = OF_results
	
	
	'''
	# some day...
	def _other_RQs(self, Traces, trigger_result):
		 res = dict()
					
		 # coincidence between channels
		 def getDt(trigger_result,ch1,ch2,dt_limit=100_000):
			 # dt is limited to dt_limit [samples]
			 tracelist = np.unique(trigger_result["trig_traceidx"])
			 dt=[]
			 N_coic=[]
			 for itrace in tracelist:
				 loc1 = trigger_result["trig_loc"][(trigger_result["trig_traceidx"]==itrace)&(trigger_result["trig_ch"]==ch1)]
				 loc2 = trigger_result["trig_loc"][(trigger_result["trig_traceidx"]==itrace)&(trigger_result["trig_ch"]==ch2)]
				 if len(loc1)>0:
					 for trigpt1 in loc1:
						 n = 0
						 for trigpt2 in loc2:
							 if abs(trigpt2-trigpt1)<dt_limit:
								 dt.append(trigpt2-trigpt1)
								 n+=1
						 N_coic.append(n)
			 dt=np.array(dt)
			 N_coic=np.array(N_coic)
			 return dt,N_coic
	
		 total_chans = [total_chan for total_chan in Traces.keys() if "total" in total_chan] #list of the total channels
		 for combination in itertools.combinations(total_chans,2): #iterate through all possible combinations of total channels
			 res["dt_"+combination[0]+"_"+combination[1]], res["dt_"+combination[0]+"_"+combination[1]+"_n"] = \
				 getDt(trigger_result,combination[0],combination[1]) #get coincidence RQs for combinations of total channels
	
		 return res
	'''


	# plot noise PSDs
	def plotPSD(self,chs=None):
		if chs is None:
			chs = self.chs
		for ch in chs:
			if ch not in self.PSDs:
				continue
			if ch in self.chnames:
				label = chnames[ch]
			else:
				label = ch
			psd = np.sqrt(2*self.PSDs[ch]/(self.tracelen*self.fsamp))
			plt.loglog(self.frequencies,psd,label=label)
		plt.grid()
		plt.grid(which='minor',alpha=0.2)
		plt.legend(loc=1)
		plt.xlim(min(self.frequencies),max(self.frequencies))
		plt.xlabel('frequency (Hz)')
		plt.ylabel(r'noise (A/$\sqrt{Hz})$')
		return

	# utility function
	def getLabelFromKey(self,key):
		ch = key.split('_')[-1]
		if ch in self.chs:
			if self.ch_names is not None:
				label = self.ch_names[self.chs.index(ch)]
			else:
				label = ch
		else:
			return None

	def plot1d(self,key):
		legend = False
		if key.split('_')[-1] in MIDASchs:
			ch = key.split('_')[-1]
			label = self.getLabelFromKey(key)
			plt.plot(self.results[key],color=MIDAScolors[MIDASchs.index(ch)],label=label)
			if label is not None:
				legend = True
		else:
			plt.plot(self.results[key])
		if legend:
			plt.legend()
		ylabel = '_'.join(key.split('_')[:-1]) # cut off "_ch"
		plt.xlabel('trigger number')
		plt.ylabel(ylabel)

	def hist1d(self,key,bins=None):
		legend = False
		if key.split('_')[-1] in MIDASchs:
			ch = key.split('_')[-1]
			label = self.getLabelFromKey(key)
			vals = self.results[key]
			if bins is None:
				bins = np.linspace(np.nanmin(vals),np.nanmax(vals),200)
			plt.hist(vals,bins=bins,histtype='step',color=MIDAScolors[MIDASchs.index(ch)],label=label)
			if label is not None:
				legend = True
		else:
			plt.plot(self.results[key])
		if legend:
			plt.legend()
		xlabel = '_'.join(key.split('_')[:-1]) # cut off "_ch"
		plt.xlabel(xlabel)
		plt.ylabel('number')


	def hist2d(self,key,key2,bins=None):
		if bins is None:
			xvals = self.results[key]
			xbins = np.linspace(np.nanmin(xvals),np.nanmax(xvals),200)
			yvals = self.results[key2]
			ybins = np.linspace(np.nanmin(yvals),np.nanmax(yvals),200)
			bins = [xbins,ybins]
		plt.hist2d(self.results[key],self.results[key2],bins=bins,norm=LogNorm())
		plt.xlabel(key)
		plt.ylabel(key2)
		plt.colorbar()



