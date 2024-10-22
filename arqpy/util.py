'''
Utility functions

Routines in this module:

gauss(x, a, x0, sigma)
pulse2(t, tau1, tau2, t0=0, normalized=True)
pulse3(t, tau1, tau2, tau3, c1, t0=0, normalized=True)
sqrtpsd(t,trace)
lpf(trace, fcut, forder, fs=DCRCfreq, forward_backward=True)
fitPulse(t,trace,poles=2,x0=None,method='Nelder-Mead')
loadEvents(event_nums=None,data_type='SLAC',**kwargs)
makePSDs(traces,chs=None,nbins=None,ntraces=None,fsamp=None)
plotPSDs(psds,fsamp=DCRCfreq,tracelen=None,chs=None,names=None)

'''
__all__ = ['gauss','pulse2','pulse3','sqrtpsd','lpf','MIDAScolors','MIDASchs',
			'fitPulse','fitSlope','loadEvents','makePSDs','plotPSDs']

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
import scipy
from scipy.optimize import minimize

# CDMS-specific
try:
	import cdms
except ImportError:
	print('WARNING: Missing CDMS library')
# MIDAS PulseViewer settings, in TES# order
MIDASchs = ['PBS1','PAS1','PCS1','PFS1','PDS1','PES1','PBS2','PFS2','PES2','PAS2','PDS2','PCS2']
MIDAScolors = ['#00ff00','#ffcc00','#ff00ff','#ff0000','#0000ff','#00ffff','#008000','#808000','#993300','#800080','#3366ff','#ff9900']
DCRCfreq = 625000.0

# Gaussian with amplitude `a`
def gauss(x, a, mu, sigma):
	return a * np.exp(-(x-mu)**2 / (2*sigma**2))


# Pulse (2-pole)
def pulse2(t, tau1, tau2, t0=0, normalized=True):
	pulse = np.zeros(len(t))
	dt = t - t0
	m = (dt>0)
	pulse[m] = -np.exp(-dt[m]/tau1) + np.exp(-dt[m]/tau2)
	if normalized:
		# normalized to peak at 1
		#norm = 1/((tau2/tau1)**(tau2/(tau1-tau2))) - (tau2/tau1)**(tau1/(tau1-tau2))
		norm = 1/((tau2/tau1)**(tau1/(tau1-tau2)) - (tau2/tau1)**(tau2/(tau1-tau2)))
		pulse *= norm
	return pulse


# Pulse (3-pole)
def pulse3(t, tau1, tau2, tau3, c1, t0=0, normalized=True):
	pulse = np.zeros(len(t))
	dt = t - t0
	m = (dt>0)
	pulse[m] = -np.exp(-dt[m]/tau1)
	pulse[m] += (1-c1)*np.exp(-dt[m]/tau2) + c1*np.exp(-dt[m]/tau3)
	if normalized:
		pulse /= np.max(pulse)
	return pulse
	

# Pulse (N-pole)
def pulseN(t, p, normalized=True):
	# p = (tau1,...,tauN,c1,...,cN-2,t0)
	pulse = np.zeros(len(t))
	dt = t - t0
	m = (dt>0)
	nc = int((len(p)-3)/2) # number of independent coefficients
	taus = p[:nc+2]
	cs = p[nc+2:-1]
	t0 = p[-1]
	c0 = 1-np.sum(cs)
	pulse[m] = -np.exp(-dt[m]/taus[0]) + c0*np.exp(-dt[m]/taus[1])
	for i in range(len(cs)):
		pulse[m] += cs[i]*np.exp(-dt[m]/taus[i+2])
	if normalized:
		pulse /= np.max(pulse)
	return pulse


# constant-fraction discriminator; assumes positive-going pulse
def cfd(y,frac=0.2,verbose=True):
	ipeak = np.argmax(y)
	thresh = frac*y[ipeak]
	for i in range(ipeak):
		if y[ipeak-i] < thresh:
			return ipeak-i
	if verbose:
		print('CFD failed')
	return -1

def fitSlope(x,y):
	x0 = x - np.mean(x)
	y0 = y - np.mean(y)
	return np.sum(x0*y0)/np.sum(x0**2)


# fit a pulse shape to a trace, and returns fit parameters
def fitPulse(t,trace,poles=2,x0=None,method='Nelder-Mead'):
	# fit around pulse
	start = cfd(trace)
	t0 = t[start]
	x = t
	y = trace
	# choose fit function and guess
	if poles == 2:
		fitfunc = lambda p: p[0]*pulse2(x,10**p[1],10**p[2],p[3])+p[4]
	elif poles == 3:
		fitfunc = lambda p: p[0]*pulse3(x,10**p[1],10**p[2],10**p[3],p[4],p[5])+p[6]
	elif poles > 3:
		fitfunc = lambda p: p[0]*pulseN(x,p[1:-1])+p[-1]
	if x0 is None:
		baseline = np.mean(y[:10])
		amplitude = np.max(y) - baseline
		if poles == 2:
			x0 = (amplitude, -5,-4, t0, baseline)
		elif poles == 3:
			x0 = (amplitude, -5,-4,-3.5, 0.1, t0, baseline)
		elif poles > 3:
			nc = poles-2
			guess = [amplitude, -5,-4]
			guess += [-3.5]*nc + [0.01]*nc + [t0, baseline]
	minfunc = lambda p: np.nansum((y-fitfunc(p))**2)
	fitres = minimize(minfunc,x0=x0,method=method)
	mle = np.array(fitres.x) # maximum likelihood estimate
	mle[1:1+poles] = 10**mle[1:1+poles]
	return mle

# turn traces in PSDs ^(1/2)
def makePSDs(traces,chs=None,nbins=None,ntraces=None,fsamp=None):
	# handles dict, or list of traces
	if type(traces) is dict:
		if chs is None:
			chs = traces.keys() # use all
		if nbins is None:
			nbins=len(traces[chs[0]][0])
		if ntraces is None:
			nbins=len(traces[chs[0]])
		psds = {}
		for ch in chs:
			psd_ch = []
			for i in range(ntraces):
				psd = np.abs(np.fft.rfft(traces[ch][i][:nbins]))
				psd_ch.append(psd)
			if len(psd_ch) > 0:
				psds[ch] = np.median(psd_ch,axis=0)
		if fsamp is not None:
			psds['frequencies'] = np.fft.rfftfreq(nbins,1/fsamp)
	else:
		if nbins is None:
			nbins=len(traces[0])
		if ntraces is None:
			nbins=len(traces)
		psd_ch = []
		for i in ntraces:
			psd = np.abs(np.fft.rfft(traces[i][:nbins]))
			psd_ch.append(psd)
		psds = np.median(psd_ch,axis=0)
	return psds

def plotPSDs(psds,fsamp=DCRCfreq,tracelen=None,chs=None,names=None):
	# handles PSD dict
	if chs is None:
		chs = sorted(psds.keys())
	for ch in chs:
		if ch not in psds.keys() or ch == 'frequencies':
			continue
		if names is not None:
			label = names[chs.index(ch)]
		else:
			label = ch
		# handle this case
		if tracelen is None:
			n = int(2*(len(psds[chs[0]])-1))
			if np.imag(psds[ch][-1]) != 0:
				n += 1
			psdfreq = np.fft.rfftfreq(n,1/fsamp)
			norm = 2/(n*fsamp)
		else:
			psdfreq = np.fft.rfftfreq(tracelen,1/fsamp)
			norm = 2/(tracelen*fsamp)
		psd = np.sqrt(norm*psds[ch])
		if 'frequencies' in psds:
			psdfreq = psds['frequencies']
		if ch in MIDASchs:
			color = MIDAScolors[MIDASchs.index(ch)]
			plt.loglog(psdfreq,psd,label=label,color=color)
		else:
			plt.loglog(psdfreq,psd,label=label)
	plt.grid()
	plt.grid(which='minor',alpha=0.2)
	plt.legend(loc=1)
	plt.xlim(min(psdfreq[psdfreq>0]),max(psdfreq))
	plt.xlabel('frequency (Hz)')
	plt.ylabel(r'noise (A/$\sqrt{Hz})$')
	return

# "PSD" in CDMS-preferred units
def sqrtpsd(t,trace):
	time_step = t[1]-t[0]
	psd = np.abs(np.fft.rfft(trace))**2
	freq = np.fft.rfftfreq(trace.size, time_step)
	# normalize
	# gitlab.com/supercdms/Reconstruction/BatCommon/-/blob/develop/pulse/PulseTools.cxx
	norm = 2.0 * time_step / len(t)
	# DC and nyquist lose factor of 2
	psd[freq==0] /= 2
	if len(t) % 2 == 0:
		psd[idx][-1] /= 2
	psd = np.sqrt(norm*psd)
	return freq, psd


# Low-pass filter (Butterworth)
def lpf(trace, fcut, forder, fs=DCRCfreq, forward_backward=True):
	if forward_backward:
		forder = int(forder/2)
		sos = scipy.signal.butter(forder, fcut, 'low', fs=fs, output='sos')
		filtered_trace = scipy.signal.sosfiltfilt(sos, trace)
	else:
		sos = scipy.signal.butter(forder, fcut, 'low', fs=fs, output='sos')
		filtered_trace = scipy.signal.sosfilt(sos, trace)	 
	return filtered_trace


# IO
def loadEvents(event_nums=None,data_type='SLAC',**kwargs):
	if data_type == 'SLAC':
		# return trace dictionary, where traces[det][ch] = [trace0,trace1,...]
		# parse keyword args
		trigid = True # use EventTriggerID if True, array index if False
		loadtrig = True
		try:
			files = kwargs['files']
			detectors = kwargs['detectors']
			chs = kwargs['chs']
			ADC2A = kwargs['ADC2A']
		except:
			print('ERROR: SLAC data require `files`, `detectors`, `chs`, `ADC2A`')
		if 'trigid' in kwargs:
			trigid = kwargs['trigid']
		if 'loadtrig' in kwargs:
			loadtrig = kwargs['loadtrig']
		traces = {}
		if loadtrig:
			traces['triggers'] = []
		for det in detectors:
			traces[det] = {}
			for ch in chs:
				traces[det][ch] = []
		reader = cdms.rawio.IO._DataReader()
		for fn in files:
			events = None
			try:
				# first get DriverPGAGains
				reader.set_filename_list([fn])
				pgagains = dict()
				for det in detectors:
					odbkey = f'/Detectors/Det0{det}/Settings/Phonon/DriverPGAGain'
					reader.set_odb_list([odbkey])
					odb = reader.get_odb_dict()
					pgagains[det] = odb[odbkey]
				# now load event dict
				if event_nums is not None:
					reader.set_event_map({fn:event_nums}, trigid)
				else:
					# get number of events + read by index, NOT EventTriggerID
					ned = reader.get_nbevents_dict()
					all_event_nums = [i for i in range(ned['NbEventsNotEmpty'])]
					# read NOT by event['event']['EventTriggerID']
					reader.set_event_map({fn:all_event_nums}, False)
				reader.set_detector_list(detectors) # array of ints
				reader.set_channel_list(chs)
				if loadtrig:
					events = reader.get_data_list(True,True,True,True,False)
				else:
					events = reader.get_data_list(True,True,True,True,True)
			except RuntimeError:
				continue
			if events is None:
				print(f'ERROR: events not found')
				return
			for event in events:
				if event['event']['TriggerType'] != 3: # entry doesn't have traces
					continue
				for det in detectors:
					for ch in chs:
						rawtrace = event[f'Z{det}'][ch] # uint16
						# invert + convert to float
						trace = -(np.array(rawtrace,dtype=float)-32768) * ADC2A / pgagains[det][MIDASchs.index(ch)]
						traces[det][ch].append(trace)
				if loadtrig:
					t0 = event['trigger']['TriggerTime']
					nev = event['event']['EventNumber'] + 1
					triggers = []
					while events[nev]['event']['TriggerType'] == 16: # LED trigger
						trigtime = events[nev]['trigger']['TriggerTime'] - t0
						triggers.append(trigtime)
						nev = nev + 1
					traces['triggers'].append(triggers)
		return traces
	else:
		print('ERROR: data_type unrecognized')
		return

# npy save / load
def saveTraces(traces,filename):
	trace_arr = []
	for det in traces:
		if det == 'triggers':
			continue
		for ch in traces[det]:
			trace_arr.append(traces[det][ch])
	np.save(filename,trace_arr)

def loadTraces(filename):
	return np.load(filename)




