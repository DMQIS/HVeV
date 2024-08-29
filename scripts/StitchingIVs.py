# Imports
import numpy as np
import matplotlib.pyplot as plt
import glob
import cdms
import sys
import os
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import mode
import warnings

# Constants
mA = 1e-3
OHM = 1
uOHM = 1e-6
mOHM = 1e-3
A2uA = 1e6
uA2A = 1e-6
V2nV = 1e9
W2pW = 1e12
ADC_BITS = 16
R_FB = 5000*OHM
R_CABLE = 0*OHM 
ADC_GAIN = 2
ADC_RANGE = 8
R_COLD_SHUNT = 5*mOHM
R_TOTAL = R_CABLE + R_FB
M_FB = 2.4
ADC2A = 1/2**ADC_BITS *ADC_RANGE/(R_FB+R_CABLE) /M_FB/ADC_GAIN
FLUX_JUMP_DETECTION_THRESHOLD = 20e-6
SUPERCONDUCTING_RANGE = 6
EPSILON = 1e-15

# MIDAS channel, names, and colors
chs = ['PBS1', 'PAS1', 'PCS1', 'PFS1', 'PDS1', 'PES1', 'PBS2', 'PFS2', 'PES2', 'PAS2', 'PDS2', 'PCS2']
cs = ['#00ff00', '#ffcc00', '#ff00ff', '#ff0000', '#0000ff', '#00ffff', '#008000', '#808000', '#993300', '#800080', '#3366ff', '#ff9900']
NAMES = ['NW A', 'NW B', 'TAMU A', 'TAMU B', 'SiC squares A', 'SiC squares B', 'SiC NFH A', 'SiC NFH B', 'SiC NFC1 A', 'SiC NFC1 B', 'SiC NFC2 A', 'SiC NFC2 B']
det = 1 

def getibis(datadir, rns, det, verboseFlag):
    ibis = np.zeros((len(rns), 12, 2))
    for i in range(len(rns)):
        rn = rns[i]
        fns = glob.glob(f'{datadir}RUN00{rn}*.gz')
        myreader = cdms.rawio.IO.RawDataReader(filepath=fns, verbose=verboseFlag)
        events = myreader.read_events(output_format=1, skip_empty=True, phonon_adctoamps=True)
        series = events.index[0][0]
        evns = events.loc[series].index
        odb = myreader.get_full_odb()
        
        for tes in range(12):
            ch = chs[tes]
            meds = []
            
            for j in range(40):
                try:
                    evn = evns[j]
                    trace = events.loc[series].loc[evn].loc[(f'Z{det}', ch)]
                    meds.append(np.median(trace))
                except:
                    continue
            
            qetbias = odb[f'/Detectors/Det0{det}/Settings/Phonon/QETBias (uA)'][tes]
            ibis[i, tes, 0] = qetbias
            ibis[i, tes, 1] = np.median(meds)
    
    return ibis

# Function to plot raw vb vs isig
def none(vb, isig):
    return vb, isig

# Retrieve OLAF run number
def extract_runNumber(datadir): # Extract the run number from the datadir path
    # Assuming the run number is the last part of the directory before the final slash
    parts = datadir.rstrip('/').split('/')
    return parts[-1]

# Stitches local jumps together, shifts everything down after jump
def stitch_flux_jumps_by_threshold(vb, isig):
    isig_stitched = isig.copy()
    flux_jumps = []

    # Detect potential flux jumps
    for index in range(len(isig_stitched) - 1):
        if abs(isig_stitched[index + 1] - isig_stitched[index]) > FLUX_JUMP_DETECTION_THRESHOLD:
            flux_jumps.append((index, isig_stitched[index + 1] - isig_stitched[index]))

    # Correct for detected flux jumps
    for jump_index, jump_value in flux_jumps:
        isig_stitched[jump_index + 1:] -= jump_value

    return isig_stitched

# Helper function
def find_two_sigma_outliers(data):
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    lower_bound = mean - 2 * std_dev
    upper_bound = mean + 2 * std_dev
    outliers = [(i, x) for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
    return outliers

# Helper function
def find_least_similar_element(data):
    if len(data) < 2:
        return 0

    def mean_absolute_deviation(x, data):
        return np.mean([abs(x - y[1]) for y in data])

    deviations = [mean_absolute_deviation(x[1], data) for x in data]
    least_similar_index = np.argmax(deviations)
    return least_similar_index

# Aviv's function for stitching to fix 0,0
def stitch_flux_jump_by_intersect(vb, isig, superconducting_range=6):
    isig_stitched = isig.copy()
    
    # Ensure there are enough points to perform polyfit
    if len(vb[-superconducting_range:]) < 2:
        print("Not enough data points for polyfit.")
        return isig_stitched
    
    try:
        m, flux_jump_magnitude = np.polyfit(vb[-superconducting_range:], isig[-superconducting_range:], 1)
    except np.linalg.LinAlgError:
        print("Polyfit failed due to singular matrix.")
        return isig_stitched
    
    for index in range(len(isig_stitched) - 1):
        if abs(isig_stitched[index + 1] - isig_stitched[index]) > abs(flux_jump_magnitude) * 0.95:
            isig_stitched[index + 1:] -= flux_jump_magnitude

    return isig_stitched

# Aviv's function for stitching
def fancy_flux_fix(vb,isig): #isig is current signal
    diffs = np.diff(isig) #finds the differences between consecutive elements in isig
    outliers = find_two_sigma_outliers(diffs) # find most likely jumps by looking at the largest differences
    transition_index = find_least_similar_element(outliers) #finds transition 
    print(transition_index)
    flux_jumps = outliers.copy()
    if transition_index == 0:
        print("No superconducting transition found for {vb}")
    else:
        flux_jumps.pop(transition_index)
    isig_stitched = isig.copy()
    for flux_jump in flux_jumps:
        isig_stitched[flux_jump[0]+1:] = isig_stitched[flux_jump[0]+1:] - flux_jump[1] #see what I can do here, want linear type fit
    return stitch_flux_jump_by_intersect(vb,isig_stitched)

# Finds SC transition
def find_significant_transition(vb, isig): 
    # Data is sorted by increasing vb
    if len(isig) < 3:
        return -1, None  # Not enough data for derivative analysis
    
    # Calculate the first derivative
    first_derivative = np.diff(isig)
    # print("First Derivative:", first_derivative)
    
    # Identify zero-crossings in the first derivative
    zero_crossings = np.where(np.diff(np.sign(first_derivative)))[0]
    # print("Zero Crossings:", zero_crossings)
    
    if len(zero_crossings) == 0:
        # print("No significant transitions found.")
        return -1, None

    if len(zero_crossings) == 1:
        # print("One significant transition found.")
        transition_index = zero_crossings[0]
        # Ensure index is within bounds
        if transition_index + 1 >= len(vb):
            print("Index out of bounds.")
            return -1, None
        transition_vb = vb[transition_index + 1]
        return transition_index + 1, transition_vb
    
    # More than one significant transition to look at
    magnitudes = np.abs(np.diff(first_derivative[zero_crossings]))
    
    if len(magnitudes) == 0:
        # print("No significant changes in derivative magnitudes found.")
        return -1, None
    
    # Find the index of the most significant zero-crossing
    most_significant_index = zero_crossings[np.argmax(magnitudes)]
    
    # Ensure index is within bounds
    if most_significant_index + 1 >= len(vb):
        return -1, None
    
    transition_vb = vb[most_significant_index + 1]
    
    # Return the index of the transition in vb
    return most_significant_index + 1, transition_vb

# Stitch flux jumps by looking at slope differences
def stitch_by_deriv(vb, isig):

# Calculate the first derivative
    first_derivative = np.diff(isig)
    
    # Identify significant changes in the first derivative
    zero_crossings = np.where(np.diff(np.sign(first_derivative)))[0]
    
    if len(zero_crossings) == 0:
        # print("No significant changes in derivative found.")
        return vb, isig

    # Apply corrections at significant derivative changes
    isig_stitched = isig.copy()
    
    for zero_crossing in zero_crossings:
        flux_jump_index = zero_crossing + 1  # Shift by 1 due to np.diff reducing length by 1
        if flux_jump_index < len(isig_stitched):
            jump_value = first_derivative[zero_crossing]  # Use derivative magnitude to adjust
            isig_stitched[flux_jump_index:] -= jump_value  # Correct for the flux jump
    return vb, isig_stitched
    
# Helper function
def linear_fit(x, m, c):
    return m * x + c

# Helper function
def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

# Main function for stitching flux jumps, calls many helpers
def JumpBuster(vb, isig):  # V6
    sorted_indices = np.argsort(vb)  # Sort the arrays
    vb = vb[sorted_indices]
    isig = isig[sorted_indices]

    # Calls function with SORTED vb array
    vb_index, transition_vb = find_significant_transition(vb, isig)  # Find possible SC transition

    if transition_vb is None:
        # No superconducting transition found
        isig_set = stitch_flux_jumps_by_threshold(vb, isig)
        vb_set = vb
    else:
        # Split into SC and NM regimes
        SC_vb = vb[:vb_index+1]
        SC_isig = isig[:vb_index+1]

        # Shift everything down to (0,0)
        SC_vb -= SC_vb[0]
        SC_isig -= SC_isig[0]
        
        SC_isig = stitch_flux_jumps_by_threshold(SC_vb, SC_isig)

        # Cut vb and isig to only normal regime
        Nm_vb = vb[vb_index+1:]
        Nm_isig = isig[vb_index+1:]

        # Fit an exponential decay to the data after the transition point
        if len(Nm_vb) > 1:  # Ensure there is enough data to fit
            try:
                # Initial guess for exponential fit
                initial_guess = [np.max(Nm_isig), 0.01]

                # Fit exponential decay model
                popt_exp, _ = curve_fit(exponential_decay, Nm_vb, Nm_isig, p0=initial_guess)
                a, b = popt_exp

                # Generate fitted exponential decay values
                fitted_exponential = exponential_decay(Nm_vb, *popt_exp)

                # Adjust the NM data to fit the exponential decay
                Nm_isig = fitted_exponential
                Nm_isig = stitch_flux_jumps_by_threshold(Nm_vb, Nm_isig)


            except (RuntimeError, OptimizeWarning) as e: # if cant be fit...
                # print(f"Error fitting exponential decay: {e}")
                Nm_isig = vb[vb_index+1:]
                
        
        # Combine sections back together
        vb_set = np.concatenate((SC_vb, Nm_vb), axis=0)
        isig_set = np.concatenate((SC_isig, Nm_isig), axis=0)

    return vb_set, isig_set

STITCHING_METHODS = {"JumpBuster":JumpBuster, "none":none, "fancy":fancy_flux_fix, "threshold":stitch_flux_jumps_by_threshold,"intersect":stitch_flux_jump_by_intersect}

# Main plotting function, can plot IV, RV, or PV plots
def plot_sweep(ibis, datadir, rns, exclude, include, stitch_type="", plot_type=""): #V4
    # Title/ Table Info
    runNumber = extract_runNumber(datadir)
    start = rns[0]
    end = rns[-1]
    TES = []
    SC_VB = []

    for tes in range(len(NAMES)):
        if NAMES[tes] in include and NAMES[tes] not in exclude:
            # Extract and convert vb and isig
            vb = ibis[:, tes, 0] * uA2A
            isig = ibis[:, tes, 1]

            most_significant_index, transition_vb = find_significant_transition(vb, isig)
            sc_transition_isig = vb[most_significant_index]
            TES.append(NAMES[tes])
            
            trans = "N/A"
            if transition_vb != None:
                trans = transition_vb*A2uA
                trans = float(trans)  # Convert to float
                if trans == 0.00:
                    SC_VB.append("N/A")
                else:
                    SC_VB.append(f"{trans:.2f}")
            else:
                SC_VB.append(trans)
        
            if np.all(vb == 0): # Check if data is logical, if not, skip
                # print(f"Warning: 'vb' array is composed entirely of zeroes for {NAMES[tes]}.")
                continue
            
            # Apply the stitching method if necessary
            if stitch_type in STITCHING_METHODS:
                if stitch_type != "none":
                    vb, isig = STITCHING_METHODS[stitch_type](vb, isig)
            # Plotting based on the 
            if plot_type == "iv":
                plt.plot(vb * A2uA, isig * A2uA, '.', color=cs[tes], label=NAMES[tes])
                plt.ylabel(r'Measured TES branch current ($\mu$A)')
                #plt.axvline(SC_trans, color='r', linestyle='--', label=f"SC Transition for {tes}")
            elif plot_type == "rv":
                rp = np.mean(vb[-SUPERCONDUCTING_RANGE:-1] / isig[-SUPERCONDUCTING_RANGE:-1])
                print("rp = " + str(rp))
                r = vb / isig - rp
                plt.plot(vb[:-1] * V2nV, r[:-1], '.', color=cs[tes], label=NAMES[tes])
                plt.ylabel(r'R($\Omega$)')
            elif plot_type == "pv":
                mask = (vb < 1400 / V2nV)
                rp = np.mean(vb[-SUPERCONDUCTING_RANGE:-1] / isig[-SUPERCONDUCTING_RANGE:-1])
                print("rp = " + str(rp))
                r = vb / isig - rp
                plt.plot(vb[mask] * A2, isig[mask]**2 * r[mask] * W2pW, '.', color=cs[tes], label=NAMES[tes])
                plt.ylabel(r'P (pW)')
        else: 
            # Extract and convert vb and isig
            vb = ibis[:, tes, 0] * uA2A
            isig = ibis[:, tes, 1]

            most_significant_index, transition_vb = find_significant_transition(vb, isig)
            sc_transition_isig = vb[most_significant_index]
            TES.append(NAMES[tes])
            SC_VB.append("N/A")

    TES.append("Stitch Type")
    SC_VB.append(stitch_type)
    
    table_data = []
    for i in range(len(TES)):
        table_data.append([TES[i], SC_VB[i]])

    plt.table(cellText=table_data,
          colLabels=['TES', 'Transition (V)'],
          cellLoc='center',
          loc='center',
          bbox=[1, 0, .6, 1])  # (x, y, width, height)

    plt.title(f"{runNumber}: Runs {start}-{end}")
    plt.xlabel(r'TES bias (nV)')
    plt.ylim(0, 25)
    plt.xlim(0, 350)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axvline(x=sc_transition_isig, color='black', linestyle='--')
    plt.legend(ncol=1, loc='upper right')
    plt.show()
