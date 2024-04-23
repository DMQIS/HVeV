# modified version of `nexus_processing_alt/Nexus_RQ.py`

import itertools
import numpy as np
import pandas as pd
import scipy

from scdmsPyTools.Traces.Filtering import *
from scdmsPyTools.Traces.Stats import *

MF_threshold = 0.2


def butter_filter(trace, cutoff, filter_order, fs=625000, forward_backward=True):
	if forward_backward:
		sos = scipy.signal.butter(filter_order//2, cutoff, 'low', fs=fs, output='sos')
		filtered = scipy.signal.sosfiltfilt(sos, trace)
	else:
		sos = scipy.signal.butter(filter_order, cutoff, 'low', fs=fs, output='sos')
		filtered = scipy.signal.sosfilt(sos, trace)	 
	return filtered


class RQ: 
	def __init__(self, OFP_WINDOW=250, ENABLE_OFP=True, OFL_MAX_DELAY=-1, MAX_CHISQ_FREQ_BIN=-1, Fs=625000, Pretrig=4096, Posttrig=4096,
				 BaseLength=4086, TailLength=3096, ShortChi2Length=200, PlateauDelay=100, PlateauLength=100,
				 UseFilterForRQs=False, CutoffFrequenciesForRQs=None, FilterOrderForRQs=10, WindowForPulseRQs=100, saturation_amplitude=None):
		self.OFP_WINDOW = OFP_WINDOW
		self.ENABLE_OFP = ENABLE_OFP # optimal filter pileup detection!
		self.OFL_MAX_DELAY = OFL_MAX_DELAY
		self.MAX_CHISQ_FREQ_BIN = MAX_CHISQ_FREQ_BIN
		self.Fs = Fs
		self.Pretrig = Pretrig
		self.Posttrig = Posttrig
		self.BaseLength = BaseLength # definition of the baseline: from 0th sample to BaseLenth sample
		self.TailLength = TailLength # definition of the tail: from -TailLength to the end of the trace
		self.PlateauDelay = PlateauDelay # time between the trigger location and the start of the window, in which PlateauAmplitude is calculated
		self.PlateauLength = PlateauLength # length of the PlateauAmplitude window [samples]
		self.ShortChi2Length = ShortChi2Length
		self.WindowForPulseRQs = WindowForPulseRQs
		self.saturation_amplitude = saturation_amplitude
		
		# These setting are for filtering raw traces with Butterworth filter
		# before calculating some of the RQs.
		# The filter is not applied for the OF RQs!
		self.UseFilterForRQs = UseFilterForRQs 
		self.CutoffFrequenciesForRQs = CutoffFrequenciesForRQs
		self.FilterOrderForRQs = FilterOrderForRQs
	
	def process_traces(self, Traces, trigger_result):
#		 res=dict()
#		 res_OF = self._run_OF(Traces)
#		 res_2 = self._other_RQs(Traces, trigger_result)

#		 # Combine results
#		 res=res_OF
#		 for key in res_2:
#			 res[key]=res_2[key]

		return self._run_OF(Traces)
	
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
	
	def _make_filter_kernel_OF(self, filter_kernels, PSD, OF_LPF_dict=-1):
		# Make OF filter for each channel
		
		if type(PSD) is dict and type(filter_kernels) is dict:
			filter_kernels_normed_freq = dict()
			filter_kernels_normed = dict()
			filter_kernels_OF_norm= dict()
			filter_kernels_OF_norm_for_chi2_rq= dict()
			filter_kernels_varPSD = dict()
			filter_kernels_varPSD_limited = dict()
			trigTemplates_freq = dict()
			
			Js=dict()
			for key in PSD:
				OF_LPF = OF_LPF_dict[key]
				PSD_template = PSD[key]
				trigTemplate = filter_kernels[key]
				trigTemplate_fft = np.fft.rfft(trigTemplate) # shortcut for Real input
				if OF_LPF>0:
					trigTemplate_fft[OF_LPF:]=0
				fft_norm = 1/(len(trigTemplate)/2)  
				J=np.abs(PSD_template)**2
				J[0]=np.inf
				OF = np.conjugate(trigTemplate_fft)/J
				OF_norm = np.real(np.sum(OF*trigTemplate_fft))
				if self.MAX_CHISQ_FREQ_BIN < 0:
					OF_norm_for_chi2_rq = OF_norm 
				else:
					OF_norm_for_chi2_rq = np.real(np.sum((OF*trigTemplate_fft)[:self.MAX_CHISQ_FREQ_BIN]))
				OF = OF/OF_norm
				trigTemplate_OF = np.fft.irfft(OF.conjugate())
				trigTemplates_freq[key]=trigTemplate_fft
				filter_kernels_normed_freq[key]=OF
				filter_kernels_normed[key]=trigTemplate_OF/fft_norm
				filter_kernels_OF_norm[key]=OF_norm
				filter_kernels_OF_norm_for_chi2_rq[key]=OF_norm_for_chi2_rq
				Js[key]=J
				# The variance of noise in time domain. Used in time-domain chi2
				filter_kernels_varPSD[key] = (np.fft.irfft(PSD_template**2)*self.Fs)[0]
				if OF_LPF > 0:
					psd_lim = np.copy(PSD_template)
					psd_lim[OF_LPF:]=0
					filter_kernels_varPSD_limited[key] = (np.fft.irfft(psd_lim**2)*self.Fs)[0]
				else:
					filter_kernels_varPSD_limited[key] = filter_kernels_varPSD[key]
				
			self.Js = Js #
			self.PSDs=PSD
			self.OF_LPF_dict = OF_LPF_dict
			self.OF_norm = filter_kernels_OF_norm #
			self.OF_norm_for_chi2_rq = filter_kernels_OF_norm_for_chi2_rq
			self.avg_pulses = filter_kernels
			self.filter_kernels_normed = filter_kernels_normed
			self.filter_kernels_normed_freq = filter_kernels_normed_freq #
			self.filter_kernels_varPSD = filter_kernels_varPSD #
			self.filter_kernels_varPSD_limited = filter_kernels_varPSD_limited #
			self.trigTemplates_freq = trigTemplates_freq #
		return
	
	# A modified version from scdmsPyTools
	def OptimumFilterAmplitude_PileupMod(self, Signal, Template, NoisePSD, downSample=8, LPF=-1, delayMax=250):

		dt = 1.0/self.Fs
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
	
	def _run_OF_ch(self, trace, ch, OF_LPF=-1, pileup_window=250):
		OFresults = {}
		N = float(len(trace))
		dt = 1/self.Fs

		# Use Golwala-Kurinsky notation
		phi_prime = self.filter_kernels_normed_freq[ch]
		#OF_norm_for_chi2_rq = self.OF_norm_for_chi2_rq[ch]
		norm = self.OF_norm[ch]
		sf = self.trigTemplates_freq[ch]
		J = np.array(self.Js[ch]) # make a copy
		vf = np.fft.rfft(trace)
		OFtrace = phi_prime*vf # the filtered trace (in Fourier space)
		
		# low pass filter, if desired. Recall negative OF_LPF means no LPF
		if OF_LPF > 0 and OF_LPF < len(J):
			J[OF_LPF:] += np.inf
			# also calculate LPF trace
			vf[OF_LPF:] = 0
			trace_lpf = np.fft.irfft(trace_freq)
			OFresults['trace'] = trace_lpf
		else:
			OFresults['trace'] = trace

		# ignore high-freq bins for chi2, if desired
		chi2mask = np.ones(len(J),dtype=bool)
		if self.MAX_CHISQ_FREQ_BIN >= 0:
			chi2mask[MAX_CHISQ_FREQ_BIN+1:] = False

		# --------------------
		# 1. OF without delay
		A_nodelay = np.real(np.sum(OFtrace))
		chi0 = np.sum(np.abs(vf[chi2mask])**2/J[chi2mask]) # first part of chi2
		chi2_nodelay = 2 * (chi0-np.abs(A_nodelay)**2*norm)
		OFresults['OF0_A'] = A_nodelay
		OFresults['OF0_chi2'] = chi2_nodelay

		# --------------------		
		# 2. OF with delay
		#  Calculate A(t0) for all t0 in one step:
		As = N/2*np.fft.irfft(OFtrace)
		chi2s = 2 * (chi0-np.abs(As)**2*norm)
		#  then find best t0
		ind  = np.argmin(chi2s)
		t0   = ind * dt
		A    = As[ind] # = np.sum(np.exp(1j*2*np.pi*f*t0)*phi*vf) / norm
		chi2 = chi2s[ind]
		OFresults['OF_time'] = t0
		OFresults['OF_A'] = A
		OFresults['OF_chi2'] = chi2
		# again, limiting max delay
		if self.OFL_MAX_DELAY > 0:
			indL = np.argmin(chi2s[:self.OFL_MAX_DELAY])
			OFresults['OFL_time'] = indL * dt
			OFresults['OFL_A'] = As[indL]
			OFresults['OFL_chi2'] = chi2s[indL]
		
		# --------------------  
		# 3. Pile-up OF
		# TODO: Jamie did not go through what this does yet
		# Limit the search window to +/-250 points around middle of trace
		# Runtime is 0.004 s for 250 points window.
		# I don't understand why 256 is x3 to x4 times slower than 250...
		if 'total' in ch.lower() and self.ENABLE_OFP:
			#half_tracelength=int(Ns//2)
			#T = self.avg_pulses[ch][half_tracelength-pileup_window:half_tracelength+pileup_window]
			#NoisePSD = scipy.signal.resample(self.PSDs[ch],len(T)//2+1)*1e-6 # Turn unit into uA; resample; note the LPF setting also need resample
			#Signal = trace[half_tracelength-pileup_window:half_tracelength+pileup_window]
			#A1s,A2s,t1,t2,Xrp = OptimumFilterAmplitude_Pileup(Signal,T,NoisePSD,self.Fs,downSample=1,LPF=int(OF_LPF/(half_tracelength/pileup_window)))
			#t1,t2 = int((t1*self.Fs+pileup_window)%int(2*pileup_window)-pileup_window),int((t2*self.Fs+pileup_window)%int(2*pileup_window)-pileup_window) # Turn the unit into samples and centered at 0
			# Use the modified version
			NoisePSD = np.copy(self.PSDs[ch])
			Signal = trace
			T = self.avg_pulses[ch]
			A1s,A2s,t1,t2,Xrp,Xrp_for_chi2_rq = self.OptimumFilterAmplitude_PileupMod(Signal,T,NoisePSD,downSample=1,LPF=int(OF_LPF),delayMax=pileup_window)
			# Turn the unit into samples and centered at 0
			t1,t2 = np.rint(t1*self.Fs), np.rint(t2*self.Fs)
			# store		
			OFresults['OFP_A1'] = A1s
			OFresults['OFP_A2'] = A2s
			OFresults['OFP_time1'] = t1
			OFresults['OFP_time2'] = t2
			OFresults['OFP_chi2'] = Xrp
			OFresults['OFP_chi2_full'] = Xrp_for_chi2_rq

		return OFresults


	def _run_OF(self, Traces):
		pretrig = self.Pretrig
		base_length = self.BaseLength
		tail_length = self.TailLength
			
		OF_results = dict()
		tlength = self.Pretrig + self.Posttrig # trace length, in samples
		tlength_2 = tlength / self.Fs / 2 # half trace length, in seconds
		RQlist=['OF0_A', 'OF0_chi2',
				'OF_A', 'OF_chi2', 'OF_time',
				'OFL_A', 'OFL_chi2', 'OFL_time',
				#'OF_chi2time', 'OF0_chi2timeFiltered', 'OF0_chi2timeShort',
				'MF', 'Amplitude', 'Max', 'MaxHead', 'MaxTail', 'Integral', 'IntegralHead', 'IntegralTail', 'BaselineSlope', 'Slope',
				'PlateauAmplitude', 'RiseTime1', 'RiseTime2', 'RiseTime3', 'FallTime1', 'FallTime2', 'FallTime3', 'PulseWidth50',
				'PulseMaxInd', 'MeanBase', 'BaselineVariance', 'TailVariance']
		RQlist_total = ['OFP1','OFP2','OFP1_time','OFP2_time','OFP_chi2','OFP_chi2_full','Npileup']
		RQlist_64bit=[]
		for ch in Traces:
			if ch == 'TTL':
				continue
			for rq_name in RQlist:
				OF_results[f"{rq_name}_{ch}"]=[]
			if 'total' in ch.lower():
				for rq_name in RQlist_total:
					OF_results[f'{rq_name}_{ch}']=[]

			chanTemplate = self.avg_pulses[ch]
			varPSD = self.filter_kernels_varPSD[ch]
			varPSDlim = self.filter_kernels_varPSD_limited[ch] 
			
			# gaus derivitive filter for the Npileup RQ
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
			
			for trace in Traces[ch]:
				OF_results_ch = self._run_OF_ch(trace, ch, OF_LPF=self.OF_LPF_dict[ch], pileup_window=self.OFP_WINDOW)

				for key in RQlist:
					if key in OF_results_ch:
						OF_results[f'{key}_{ch}'].append(OF_results_ch[key])
				if 'total' in ch.lower():
					for key in RQlist_total:
						if key in OF_results_ch:
							OF_results[f'{key}_{ch}'].append(OF_results_ch[key])
				

				# compute no-delay, time-domain chi2
				# TODO make this optional
				trace_lpf = OF_results_ch['trace']
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
				
				#compute other RQs
				filtered_trace = trace_lpf
				if self.UseFilterForRQs:
					# Using butterworth filter instead
					filtered_trace = butter_filter(trace, self.CutoffFrequenciesForRQs[ch], self.FilterOrderForRQs, fs=self.Fs)
				
				baseline = np.mean(filtered_trace[:base_length])
				x_time = np.arange(len(trace))
				S = filtered_trace - baseline
				threshold = self.saturation_amplitude[ch]*MF_threshold
				A = self.matchedFilter(S[pretrig:], chanTemplate[pretrig:], threshold)
				s1 = np.sum(S[S>threshold])
				s2 = A*np.sum(chanTemplate[S<threshold])
				OF_results['MF_'+ch].append(s1+s2)
				OF_results['Amplitude_'+ch].append(np.max(S))
				OF_results['MeanBase_'+ch].append(baseline)
				OF_results['BaselineVariance_'+ch].append(np.var(filtered_trace[:base_length]))
				OF_results['TailVariance_'+ch].append(np.var(filtered_trace[-tail_length:]))
				OF_results['Max_'+ch].append(np.max(filtered_trace))
				OF_results['MaxHead_'+ch].append(np.max(filtered_trace[:base_length]))
				OF_results['MaxTail_'+ch].append(np.max(filtered_trace[-tail_length:]))				
				OF_results['Integral_'+ch].append(np.sum(S))
				OF_results['IntegralHead_'+ch].append(np.sum(S[:base_length]))
				OF_results['IntegralTail_'+ch].append(np.sum(S[-tail_length:]))
				OF_results["BaselineSlope_"+ch] = np.append(OF_results["BaselineSlope_"+ch],
															slope(x_time[:base_length], S[:base_length], removeMeans=True))
				OF_results["Slope_"+ch] = np.append(OF_results["Slope_"+ch], slope(x_time, S, removeMeans=True))
				OF_results['PlateauAmplitude_'+ch] = np.append(OF_results['PlateauAmplitude_'+ch],
															   np.mean(filtered_trace[self.Pretrig+self.PlateauDelay:self.Pretrig+self.PlateauDelay+self.PlateauLength]))
				
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
				# 2. filtered_trace: After LPF at the OF_LPF bin or after butterworth filter, if this option is enabled
				# 3. S: filtered_trace after subtracting baseline
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
		return OF_results
	
	
#	 def _other_RQs(self, Traces, trigger_result):
#		 res = dict()
						
#		 # coincidence between channels
#		 def getDt(trigger_result,ch1,ch2,dt_limit=100_000):
#			 # dt is limited to dt_limit [samples]
#			 tracelist = np.unique(trigger_result["trig_traceidx"])
#			 dt=[]
#			 N_coic=[]
#			 for itrace in tracelist:
#				 loc1 = trigger_result["trig_loc"][(trigger_result["trig_traceidx"]==itrace)&(trigger_result["trig_ch"]==ch1)]
#				 loc2 = trigger_result["trig_loc"][(trigger_result["trig_traceidx"]==itrace)&(trigger_result["trig_ch"]==ch2)]
#				 if len(loc1)>0:
#					 for trigpt1 in loc1:
#						 n = 0
#						 for trigpt2 in loc2:
#							 if abs(trigpt2-trigpt1)<dt_limit:
#								 dt.append(trigpt2-trigpt1)
#								 n+=1
#						 N_coic.append(n)
#			 dt=np.array(dt)
#			 N_coic=np.array(N_coic)
#			 return dt,N_coic
		
#		 total_chans = [total_chan for total_chan in Traces.keys() if "total" in total_chan] #list of the total channels
#		 for combination in itertools.combinations(total_chans,2): #iterate through all possible combinations of total channels
#			 res["dt_"+combination[0]+"_"+combination[1]], res["dt_"+combination[0]+"_"+combination[1]+"_n"] = \
#				 getDt(trigger_result,combination[0],combination[1]) #get coincidence RQs for combinations of total channels
		
#		 return res
