# Imports
import numpy as np
import matplotlib.pyplot as plt
import glob
import cdms

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
R_PARA = 15*mOHM # Is this the correct value?
R_TOTAL = R_CABLE + R_FB
M_FB = 2.4
ADC2A = 1/2**ADC_BITS *ADC_RANGE/(R_FB+R_CABLE) /M_FB/ADC_GAIN
FLUX_JUMP_DETECTION_THRESHOLD = 20e-6
SUPERCONDUCTING_RANGE = 6
EPSILON = 1e-15

# MIDAS channel, names, and colors
chs = ['PBS1', 'PAS1', 'PCS1', 'PFS1', 'PDS1', 'PES1', 'PBS2', 'PFS2', 'PES2', 'PAS2', 'PDS2', 'PCS2']
cs = [
    "#4A90E2",  # Sky Blue
    "#50E3C2",  # Mint Green
    "#B8E986",  # Light Green
    "#F5A623",  # Golden Orange
    "#D0021B",  # Bright Red
    "#7B92A5",  # Steel Blue
    "#BD10E0",  # Bright Purple
    "#F8E71C",  # Bright Yellow
    "#D1D8E0",  # Light Gray
    "#9B9B9B",  # Medium Gray
    "#F4A7B9",  # Soft Pink
    "#E94F77"   # Reddish Pink
]

NAMES = ['NW A', 'NW B', 'TAMU A', 'TAMU B', 'SiC squares A', 'SiC squares B', 'SiC NFH A', 'SiC NFH B', 'SiC NFC1 A', 'SiC NFC1 B', 'SiC NFC2 A', 'SiC NFC2 B']
det = 1 

# Function to retrieve data from DB
def getibis(datadir, rns, det, verboseFlag):
    ibis = np.zeros((len(rns), 12, 2))
    for i in range(len(rns)):
        rn = rns[i]
        rn_str = f'{rn:05}' # Format the run number to be five digits long with leading zeros
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
    parts = datadir.rstrip('/').split('/')
    return parts[-1]

# Helper function
def find_two_sigma_outliers(data):
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    lower_bound = mean - 2 * std_dev
    upper_bound = mean + 2 * std_dev
    outliers = [(i, x) for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
    return outliers

# Stitched flux jumps by finding outliers in the differences (ignoring the transition jump)
def stitch_by_diffs(vb,isig): 
    diffs = np.diff(isig) 
    outliers = find_two_sigma_outliers(diffs)
    transition_index, transition_vb = find_SC_transition(vb, isig)
    flux_jumps = outliers.copy()
    if transition_index != 0 and transition_index != None and len(flux_jumps) > 0 and transition_index < len(flux_jumps):
        flux_jumps.pop(transition_index)
    isig_stitched = isig.copy()
    for flux_jump in flux_jumps:
        isig_stitched[flux_jump[0]+1:] = isig_stitched[flux_jump[0]+1:] - flux_jump[1] 
    return vb, isig_stitched


def find_SC_transition(vb, isig, A2uA=1.0):
    if len(isig) < 3:
        return -1, None  # Not enough data for derivative analysis

    dvb = np.diff(vb)
    disig = np.diff(isig)
    
    # First derivative, skipping division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        first_derivative = np.divide(disig, dvb, out=np.zeros_like(disig), where=np.abs(dvb) > 0)
    
    # Second derivative, skipping division by zero
    dfirst_derivative = np.diff(first_derivative)
    dvb2 = np.diff(vb[:-1])  # Adjust length to match first_derivative
    with np.errstate(divide='ignore', invalid='ignore'):
        second_derivative = np.divide(dfirst_derivative, dvb2, out=np.zeros_like(dfirst_derivative), where=np.abs(dvb2) > 0)

    # Identify points where the second derivative transitions from positive to negative
    transition_indices = []
    for i in range(1, len(second_derivative)):
        if second_derivative[i] < 0 and second_derivative[i - 1] >= 0:
            transition_indices.append(i)

    if not transition_indices:
        return -1, None  # No significant transition found
    
    # Determine the most significant transition based on the magnitude of the second derivative
    magnitudes = np.abs(second_derivative[transition_indices])
    most_significant_index = transition_indices[np.argmax(magnitudes)]
    
    # Ensure index is within bounds
    if most_significant_index + 1 >= len(vb):
        return -1, None
    
    transition_vb = vb[most_significant_index + 1]
    
    if transition_vb == 0:
        return -1, None

    transition_vb *= 1e6
    
    return most_significant_index + 1, transition_vb

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

        sc_vb, sc_isig = stitch_by_diffs(sc_vb, sc_isig)

        if sc_isig[0] != 0:
            sc_isig -= sc_isig[0]

        nm_vb = vb[vb_index+1:]
        nm_isig = isig[vb_index+1:]

        if nm_isig[-1] != 0 or nm_isig[-1] < 1e-1:
            nm_isig -= nm_isig[-1]

        vb = np.concatenate((sc_vb, nm_vb), axis=0)
        isig = np.concatenate((sc_isig, nm_isig), axis=0)

    
    else: # Transition not found
        vb, isig = stitch_by_diffs(vb, isig)
        return vb, isig
        
    return vb, isig

STITCHING_METHODS = {"JumpBuster":JumpBuster, "none":none}

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
        fig, ax = plt.subplots(figsize=(8, 6)) 
        axs = ax 
        # plt.tight_layout(rect=[0, .99, 1, 0.01]) 
        # plt.subplots_adjust(wspace=0, hspace=0)
    else:
        fig, axs = plt.subplots(1, num_plots, figsize=(5*num_plots, 4), sharey=False)
        axs = np.array(axs)  

    if num_plots == 1:
        axs = [axs] 

    for ax in axs:
        ax.axvline(x=0, color='black', linestyle='--')
        ax.axhline(y=0, color='black', linestyle='--')
    
    for tes in range(len(NAMES)):
        if NAMES[tes] in include:
            # Extract and convert vb and isig
            vb = ibis[:, tes, 0] * uA2A
            isig = ibis[:, tes, 1]

            SC_trans_index, transition_vb = find_SC_transition(vb, isig)
            sc_transition_isig = vb[SC_trans_index]
            
            if NAMES[tes] in exclude:
                TES.append(NAMES[tes])
                SC_VB.append("N/A")
                continue  # Skip further processing for excluded TES
            
            TES.append(NAMES[tes])
            
            trans = "N/A"
            if transition_vb is not None:
                trans = transition_vb
                trans = float(trans)
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

                elif ptype == "rv" or ptype == "pv":
                    safe_isig = np.where(isig != 0, isig, 1e-16)
                    rp = np.mean(vb[0:SC_trans_index] / safe_isig[0:SC_trans_index])
                    r = (vb / safe_isig) - rp
                    
                    if ptype == "rv":
                        lower_y_lim = np.percentile(r, 10) 
                        upper_y_lim = np.percentile(r, 90) 
                        y_val = max(abs(lower_y_lim), abs(upper_y_lim))
                        
                        ax.plot(vb[:-1] * A2uA, abs(r[:-1]), '.', color=cs[tes], label=NAMES[tes])
                        ax.set_ylabel(r'R($\Omega$)')
                        ax.set_xlabel(r'TES bias (nV)')
                        ax.set_xlim(vb.min() * A2uA, vb.max() * A2uA)  # Convert x-limits to the desired unit
                        ax.set_ylim(-y_val, y_val)
                        
                    else:  # PV plot
                        num = R_COLD_SHUNT * isig
                        R_tot = R_COLD_SHUNT + R_PARA
                        denom = R_tot + r
                        I_S = num / denom
                        power = (I_S**2 * r * W2pW)
                        
                        lower_y_lim = np.percentile(power, 5)   
                        upper_y_lim = np.percentile(power, 95) 
                        y_val = max(abs(lower_y_lim), abs(upper_y_lim))

                        ax.plot(vb * A2uA, power, '.', color=cs[tes], label=NAMES[tes])
                        ax.set_ylabel(r'P (pW)')
                        ax.set_xlabel(r'TES bias (nV)')
                        ax.set_xlim(vb.min() * A2uA, vb.max() * A2uA)  
                        ax.set_ylim(-y_val, y_val)
                        
            # Remove legend from plot if it exists
            for ax in axs:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
        
        else: 
            TES.append(NAMES[tes])
            SC_VB.append("N/A")

    if "iv" in plot_types:
        iv_ax = axs[plot_types.index("iv")]
        TES.append("Stitch Type")
        SC_VB.append(stitch_type) 
        table_data = []
        for i in range(len(TES)):
            table_data.append([TES[i], SC_VB[i]])
            
        # Add the table with colored text
        table = iv_ax.table(cellText=table_data, colLabels=['TES', 'Transition (V)'], cellLoc='center', loc='center', bbox=[-1, 0, .6, 1])
        
        # Apply color to table font
        for i, key in enumerate(TES):
            if key == "Stitch Type":
                continue
            row_index = i
            cell = table[(row_index + 1, 0)]
            color = cs[i] 
            cell.set_text_props(color=color)
    plt.suptitle(f"{runNumber}: Runs {start}-{end}")
    if num_plots != 1:
        plt.tight_layout(rect=[0, .01, 1, 0.99]) 
        plt.subplots_adjust(wspace=0.3, hspace=0) 
    plt.show()
