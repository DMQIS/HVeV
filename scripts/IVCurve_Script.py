import numpy as np
import matplotlib.pyplot as plt
import glob
import cdms
import sys
import os

# MIDAS channel names and colors
chs = ['PBS1', 'PAS1', 'PCS1', 'PFS1', 'PDS1', 'PES1', 'PBS2', 'PFS2', 'PES2', 'PAS2', 'PDS2', 'PCS2']
cs = ['#00ff00', '#ffcc00', '#ff00ff', '#ff0000', '#0000ff', '#00ffff', '#008000', '#808000', '#993300', '#800080', '#3366ff', '#ff9900']
names = ['NW A', 'NW B', 'TAMU A', 'TAMU B', 'SiC squares A', 'SiC squares B', 'SiC NFH A', 'SiC NFH B', 'SiC NFC1 A', 'SiC NFC1 B', 'SiC NFC2 A', 'SiC NFC2 B']
det = 1

# Suppress text output
class NullWriter:
    def write(self, _):
        pass
    def flush(self):
        pass

def extract_run_number(datadir): # Extract the run number from the datadir path
    # Assuming the run number is the last part of the directory before the final slash
    parts = datadir.rstrip('/').split('/')
    return parts[-1]

# Function to plot IV curves
def plot_iv_curves(datadir, rns, det, verbose): # verbose flag to toggle printed output on or off
    ibis = np.zeros((len(rns), 12, 2))

    if verbose == False:
        sys.stdout = NullWriter()  # Redirect stdout
        sys.stderr = NullWriter()  # Redirect stderr
    else:
        pass
    
    # Loop over runs and extract data
    for i in range(len(rns)):
        rn = rns[i]
        fns = glob.glob(f'{datadir}RUN00{rn}*.gz')
        myreader = cdms.rawio.IO.RawDataReader(filepath=fns, verbose=False)
        events = myreader.read_events(output_format=1, skip_empty=True, phonon_adctoamps=True)
        series = events.index[0][0]
        evns = events.loc[series].index
        odb = myreader.get_full_odb()
        
        for tes in range(12):
            ch = chs[tes]
            meds = []
            
            for j in range(40): # ? is this variable or always fixed?
                try:
                    evn = evns[j]
                    trace = events.loc[series].loc[evn].loc[(f'Z{det}', ch)]
                    meds.append(np.median(trace))
                except:
                    continue
            
            qetbias = odb[f'/Detectors/Det0{det}/Settings/Phonon/QETBias (uA)'][tes]
            ibis[i, tes, 0] = qetbias
            ibis[i, tes, 1] = np.median(meds)
    RUN = extract_run_number(datadir)
    # Plot the IV curves after processing the data
    plt.figure(figsize=(10, 6))
    for tes in range(12):
        ib = ibis[:, tes, 0]
        isig = ibis[:, tes, 1]
        plt.plot(ib, isig * 1e6, '.', color=cs[tes], label=names[tes])
    plt.xlabel(r'TES bias ($\mu$A)')
    plt.ylabel(r'Measured TES branch current ($\mu$A)')
    plt.title(f'{RUN}: IV Curves for Runs {rns[0]}-{rns[-1]}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Restore stdout and stderr
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
