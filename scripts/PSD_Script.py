import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cdms
sys.path.append('../scripts')
from Nexus_RQ import RQ
from Nexus_utils import *   # loader, trigger, Pulse2, glob


chs = ['PCS1','PFS1','PBS2','PFS2'] # channels to use (which also serve as dictionary keys)
names = ['TAMU_A','TAMU_B','NFH_A','NFH_B']
taus = [[1e-5,3e-4],[1e-5,3e-4], [1e-6,5e-4],[1e-6,5e-4]] # pulse rise/fall times (seconds)
sats = [10,10,10,10] # saturation amplitudes (uA), only for matched filter RQs

# general
fsamp = 625000 # Hz
pretrig = 4096 # bins
posttrig = 4096
tracelen = pretrig + posttrig # trace used for RQ processing
psdfreq = np.fft.rfftfreq(tracelen,1/fsamp)
ADC2A = 1/2**16 *8/5e3 /2.4/4 # 16-bit ADC, 8V range, 5kOhm R_FB, 2.4 turn ratio, gain = 4
# MIDAS
allchs = ['PBS1','PAS1','PCS1','PFS1','PDS1','PES1','PBS2','PFS2','PES2','PAS2','PDS2','PCS2']
colors = ['#00ff00','#ffcc00','#ff00ff','#ff0000','#0000ff','#00ffff','#008000','#808000','#993300','#800080','#3366ff','#ff9900']

# trigger options
randomrate = 0 # random triggers to add per trace
trigger_channels = chs # channels to trigger on
trigger_threshold_uA = [0.05, 0.03, 0.05, 0.05] # in uA
trigger_threshold = [x*1e-6/ADC2A for x in trigger_threshold_uA] # OF amp, in ADCu, to trigger on
deactivation_threshold = [0.5*x for x in trigger_threshold]
window = False # add window function to filtered pulse used for trigger

def extract_run(datadir): # Extract the run number from the datadir path
    # Assuming the run number is the last part of the directory before the final slash
    parts = datadir.rstrip('/').split('/')
    return parts[-1]

import re

def extract_run_number(filename_pattern):
    # Define a regular expression pattern to match the run number
    pattern = r'RUN(\d{5})_'
    match = re.search(pattern, filename_pattern)
    
    if match:
        # Extract the run number from the match
        run_number = match.group(1)
        # Convert it to an integer and return
        return int(run_number)
    else:
        raise ValueError("Run number not found in filename pattern")


def makePSD(filename_pattern, datadir):
    fns = glob(filename_pattern)[:1] # limit to the first file
    myreader = cdms.rawio.IO.RawDataReader(filepath=fns)
    events = myreader.read_events(output_format=1, # pandas
                                  skip_empty=True,
                                  channel_names=chs,
                                  phonon_adctoamps=False) # convert to A manually
    series = events.index[0][0]
    evs = events.loc[series].index
    
    # get PGAgain (although for us it always = 1)
    odb = myreader.get_full_odb()
    
    psds = np.zeros((12,len(psdfreq)))
    for ch in chs:
        tes = allchs.index(ch)
        pgagain = odb[f'/Detectors/Det01/Settings/Phonon/DriverPGAGain'][tes]
        psds_tes = []
        for i in range(len(evs)):
            ev = evs[i]
            trace = events.loc[series].loc[ev].loc[('Z1',ch)]
            # must cast to float from uint (or else)
            trace = (np.array(trace,dtype=float)-32768) * ADC2A / pgagain
            # non-robust pulse rejection
            if np.max(trace)-np.min(trace) > 2e-6:
                continue
            psd = np.abs(np.fft.rfft(trace[:tracelen]))
            psds_tes.append(psd)
        if len(psds_tes) > 0:
            psds[tes] = np.median(psds_tes,axis=0)
    np.save('olaf11_psds.npy',psds)
    RUN = extract_run(datadir)
    numb = extract_run_number(filename_pattern)
    # plot
    for ch in chs:
        tes = allchs.index(ch)
        psd = psds[tes] * np.sqrt(2.0 / (fsamp*tracelen)) # normalize
        plt.loglog(psdfreq,psd,color=colors[tes],label=ch)
    plt.legend(loc='best')
    plt.title(f'{RUN}: PSD for Run {numb}')
    plt.xlabel('frequency (Hz)')
    plt.ylabel(r'noise (A/$\sqrt{Hz})$')
    plt.grid(color='k',alpha=0.3)