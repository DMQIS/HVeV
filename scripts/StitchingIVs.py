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
        # Format the run number to be five digits long with leading zeros
        rn_str = f'{rn:05}'
        fns = glob.glob(f'{datadir}RUN{rn_str}*.gz')
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
    # print(transition_index)
    flux_jumps = outliers.copy()
    if transition_index == 0:
        print("No superconducting transition found for {vb}")
    else:
        flux_jumps.pop(transition_index)
    isig_stitched = isig.copy()
    for flux_jump in flux_jumps:
        isig_stitched[flux_jump[0]+1:] = isig_stitched[flux_jump[0]+1:] - flux_jump[1] #see what I can do here, want linear type fit
    return stitch_flux_jump_by_intersect(vb,isig_stitched)

# Finds most significant transition
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
        if transition_vb == 0:
            transition_vb = None
        return transition_index + 1, transition_vb * A2uA
    
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
    
    transition_vb *= A2uA
    
    # Return the index of the transition in vb
    return most_significant_index + 1, transition_vb


def find_SC_transition(vb, isig):
    if len(isig) < 3:
        return -1, None  # Not enough data for derivative analysis

    dvb = np.diff(vb)
    disig = np.diff(isig)
    
    first_derivative = np.where(dvb != 0, disig / dvb, 0)
    
    dvb = np.diff(vb[:-1])  # Note: Adjust length to match first_derivative
    dfirst_derivative = np.diff(first_derivative)
    
    second_derivative = np.where(dvb != 0, dfirst_derivative / dvb, 0)

    # Identify points where the second derivative transitions from positive to negative
    transition_indices = []
    for i in range(1, len(second_derivative)):
        if second_derivative[i] < 0 and second_derivative[i - 1] >= 0:
            transition_indices.append(i)

    if not transition_indices:
        return -1, None  # No significant transition found
    
    # Determine the most significant transition based on the magnitude of the second derivative
    # Here, choose the transition with the largest magnitude of the second derivative
    magnitudes = np.abs(second_derivative[transition_indices])
    most_significant_index = transition_indices[np.argmax(magnitudes)]
    
    # Ensure index is within bounds
    if most_significant_index + 1 >= len(vb):
        return -1, None
    
    transition_vb = vb[most_significant_index + 1]
    
    if transition_vb == 0:
        return -1, None
    
    return most_significant_index + 1, transition_vb * A2uA


def stitch_by_deriv(vb, isig):
    # Calculate differences
    dvb = np.diff(vb)
    disig = np.diff(isig)
    
    # Calculate first derivative with safe division
    first_derivative = np.where(dvb != 0, disig / dvb, 0)
    
    # Identify significant changes in the first derivative
    # Find zero crossings in the first derivative
    zero_crossings = np.where(np.diff(np.sign(first_derivative)))[0]
    
    if len(zero_crossings) == 0:
        return vb, isig  # No significant changes found, return original

    isig_stitched = isig.copy()
    
    for zero_crossing in zero_crossings:
        flux_jump_index = zero_crossing + 1  # Shift by 1 due to np.diff reducing length by 1
        if flux_jump_index < len(isig_stitched):
            # Calculate the jump value based on the difference in the derivative
            jump_value = first_derivative[zero_crossing] if zero_crossing < len(first_derivative) else 0
            
            # Correct the flux jump in the `isig` data
            isig_stitched[flux_jump_index:] -= jump_value
            
    return vb, isig_stitched

def stitch_by_diffs(vb, isig, threshold=0.1):
    # Calculate differences between consecutive points
    dvb = np.diff(vb)
    disig = np.diff(isig)
    
    # Calculate differences in disig to find constant differences
    diff_disig = np.diff(disig)
    
    # Identify significant changes where difference exceeds the threshold
    significant_changes = np.where(np.abs(diff_disig) > threshold)[0]
    
    if len(significant_changes) == 0:
        return vb, isig  # No significant changes found, return original
    
    isig_stitched = isig.copy()
    
    for change_index in significant_changes:
        # Determine the index in the original data array
        adjustment_index = change_index + 1
        
        if adjustment_index < len(isig_stitched):
            # Calculate the amount to adjust based on the significant change
            adjustment_value = disig[adjustment_index]
            
            # Adjust all subsequent points by the identified amount
            isig_stitched[adjustment_index:] -= adjustment_value
            
    return vb, isig_stitched

def shift_to_zero(vb, isig):
    # Shift data such that the smallest value is at zero
    isig -= np.min(isig)
    return vb, isig

def stitch_by_intersect(vb, isig):
    vb -= vb[0]
    isig -= isig[0]

    vb, isig = stitch_by_deriv(vb, isig)
    
    return vb, isig
    
# Helper function
def linear_fit(x, m, c):
    return m * x + c

# Helper function
def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def estimate_decay_constant(vb, isig):
    safe_isig = np.where(isig != 0, isig, 1e-12)  # Avoid division by zero
    Y = safe_isig[-1] / safe_isig[0]
    Range = vb[-1] - vb[0]
    P = Y ** (1/ Range)
    rate = 1 - P
    return rate

def shift_to_zero(vb, isig):
    # Shift data such that the smallest value is at zero
    isig -= np.min(isig)
    return vb, isig

def apply_shift_if_needed(vb, isig, transition, threshold=0.65, tolerance=1e-8):
    # Calculate the proportion of negative values in isig
    negative_count = np.sum(isig < 0)
    total_count = len(isig)
    
    if total_count == 0:
        raise ValueError("The isig array is empty")
    
    proportion_negative = negative_count / total_count
    
    # print(f"Proportion of negative values in isig: {proportion_negative:.2f}")
    
    # Apply shift if the proportion of negative values exceeds the threshold
    if proportion_negative > threshold:
        # print("Applying shift to zero")
        vb, isig = shift_to_zero(vb, isig)
    
    # Apply shift to intercept if necessary
    if not np.isclose(isig[0], 0, atol=tolerance):
        # print("Applying shift to intercept")
        SC_isig = isig[:transition + 1]
        NM_isig = isig[transition + 1:]
        SC_isig -= SC_isig[0]

        isig = np.concatenate((SC_isig, NM_isig), axis=0)
    
    return vb, isig

def JumpBuster(vb, isig): 
    sorted_indices = np.argsort(vb)  # Sort the arrays
    vb = vb[sorted_indices]
    isig = isig[sorted_indices]

    vb -= vb[0]
    isig -= isig[0]

    vb_index, transition_vb = find_SC_transition(vb, isig)  # Implement your transition detection logic

    if transition_vb != None and vb_index != 0: # Found the transition
        sc_vb = vb[:vb_index+1] 
        sc_isig = isig[:vb_index+1]

        nm_vb = vb[vb_index+1:]
        nm_isig = isig[vb_index+1:]

        nm_vb, nm_isig = shift_to_zero(nm_vb, nm_isig)

        vb = np.concatenate((sc_vb, nm_vb), axis=0)
        isig = np.concatenate((sc_isig, nm_isig), axis=0)
    else:
        vb, isig = stitch_by_diffs(vb, isig)
        
    return vb, isig


STITCHING_METHODS = {"JumpBuster":JumpBuster, "none":none, "fancy":fancy_flux_fix, "threshold":stitch_flux_jumps_by_threshold,"intersect":stitch_flux_jump_by_intersect}

# Main plotting function, can plot IV, RV, or PV plots
def plot_sweep(ibis, datadir, rns, exclude, include, stitch_type="", plot_type=""):
    # Title/ Table Info
    runNumber = extract_runNumber(datadir)
    start = rns[0]
    end = rns[-1]
    TES = []
    SC_VB = []

    # Determine plot types
    plot_types = plot_type.split('+')
    num_plots = len(plot_types)
    
    # Create subplots
    if num_plots == 1:
        fig, ax = plt.subplots(figsize=(10, 6))  # Single subplot, adjust size as needed
        axs = ax  # Use single Axes object
    else:
        fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 4), sharey=False)
        axs = np.array(axs)  # Convert to array to ensure consistency in indexing


    # Ensure axs is always treated correctly
    if num_plots == 1:
        axs = [axs]  # Wrap single Axes in a list for consistent handling

    for ax in axs:
        ax.axvline(x=0, color='black', linestyle='--')
        ax.axhline(y=0, color='black', linestyle='--')
    
    for tes in range(len(NAMES)):
        if NAMES[tes] in include:
            # Extract and convert vb and isig
            vb = ibis[:, tes, 0] * uA2A
            isig = ibis[:, tes, 1]

            SC_trans_index, transition_vb = find_significant_transition(vb, isig)
            sc_transition_isig = vb[SC_trans_index]
            
            if NAMES[tes] in exclude:
                TES.append(NAMES[tes])
                SC_VB.append("N/A")
                continue  # Skip further processing for excluded TES
            
            TES.append(NAMES[tes])
            
            trans = "N/A"
            if transition_vb is not None:
                trans = transition_vb
                trans = float(trans)  # Convert to float
                if trans == 0.00:
                    SC_VB.append("N/A")
                else:
                    SC_VB.append(f"{trans:.2f}")
            else:
                SC_VB.append(trans)
        
            if np.all(vb == 0): # Check if data is logical, if not, skip
                continue
            
            # Apply the stitching method if necessary
            if stitch_type in STITCHING_METHODS:
                if stitch_type != "none":
                    vb, isig = STITCHING_METHODS[stitch_type](vb, isig)
            else:
                print("Not a valid stitching method.")
            
            # Plotting based on plot_type
            for i, ptype in enumerate(plot_types):
                ax = axs[i]

                if ptype == "iv":
                    ax.plot(vb * A2uA, isig * A2uA, '.', color=cs[tes], label=NAMES[tes])
                    ax.set_ylabel(r'Measured TES branch current ($\mu$A)')
                    ax.set_xlabel(r'TES bias (nV)')            
                
                elif ptype == "rv":
                    safe_isig = np.where(isig != 0, isig, 1e-12)
                    rp = np.mean(vb[0:SC_trans_index] / safe_isig[0:SC_trans_index])
                    r = (vb / safe_isig) - rp
                    ax.plot(vb[:-1] * A2uA, r[:-1], '.', color=cs[tes], label=NAMES[tes])
                    ax.set_ylabel(r'R($\Omega$)')
                    ax.set_xlabel(r'TES bias (nV)')
                    ax.set_xlim(vb.min() * A2uA, vb.max() * A2uA)
                    ax.set_ylim(r.min(), r.max())
                
                elif ptype == "pv":
                    safe_isig = np.where(isig != 0, isig, 1e-12)
                    mask = (vb < 1400 / V2nV)
                    
                    if np.any(mask):  # Check if mask has selected any data
                        rp = np.mean(vb[0:SC_trans_index] / safe_isig[0:SC_trans_index])
                        r = (vb / safe_isig) - rp
                        power = isig[mask]**2 * r[mask] * W2pW
                        ax.plot(vb[mask] * A2uA, power, '.', color=cs[tes], label=NAMES[tes])
                        ax.set_ylabel(r'P (pW)')
                        ax.set_xlabel(r'TES bias (nV)')
                        ax.set_xlim(vb.min() * A2uA, vb.max() * A2uA)
                        ax.set_ylim(0, np.max(power))  # Avoid setting identical limits
                    else:
                        ax.set_xlim(0, 1)  # Default limits if no data
                        ax.set_ylim(0, 1)
                        
        else: 
            # Handle cases where TES is in exclude list
            TES.append(NAMES[tes])
            SC_VB.append("N/A")

    if "iv" in plot_types:
        iv_ax = axs[plot_types.index("iv")]
        # ax.set_xlim(vb.min() * A2uA, vb.max() * A2uA)
        # ax.set_ylim(isig.min(), isig.max())
        TES.append("Stitch Type")
        SC_VB.append(stitch_type)
                
        table_data = []
        for i in range(len(TES)):
            table_data.append([TES[i], SC_VB[i]])
            
        iv_ax.table(cellText=table_data, colLabels=['TES', 'Transition (V)'], cellLoc='center', loc='center', bbox=[1, 0, .6, 1])  # (x, y, width, height)
        iv_ax.legend(ncol=1, loc='upper right')

    plt.suptitle(f"{runNumber}: Runs {start}-{end}")
    plt.tight_layout(rect=[0, .01, 1, 0.99])  # Adjust to fit title and labels
    plt.show()
