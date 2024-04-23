# PSDs, templates, OFs, RQs
from Nexus_RQ import RQ
from Nexus_utils import Pulse2

# temporary hard-coded stuff
chs = ['PAS2','PCS2'] # channel names, which also serve as dictionary keys
PSDarrays = np.load('olaf6psds.npy')
taus = [[6e-6,2e-4],[7e-6,1e-4]]
Templates = dict() # could have used = {}
PSDs = dict()
OF_LPF_dict = dict()
for i in range(len(chs)):
	key = chs[i]
	PSDs[key] = PSDarrays[i]
	Templates[key] = Pulse2(x,*taus[i])
	OF_LPF_dict[key] = 0



# instantiate Reduced Quantity object
rq = RQ(OFP_WINDOW=250,ENABLE_OFP=True,OFL_MAX_DELAY=25,MAX_CHISQ_FREQ_BIN=-1,
		Fs=625000,Pretrig=4096,Posttrig=4096,BaseLength=4086,TailLength=3096,
		ShortChi2Length=200,PlateauDelay=100, PlateauLength=100,UseFilterForRQs=False,
		CutoffFrequenciesForRQs=None,FilterOrderForRQs=10,WindowForPulseRQs=100)

# make Optimal Filter
#  calling a private function smh...
rq._make_filter_kernel_OF(Templates, PSDs, OF_LPF_dict=OF_LPF_dict)
rq.ADC2uA = ADC2uA
rq.OF_LPF = OF_LPF
rq.OF_LPF_dict = OF_LPF_dict

ld = loader(filename_pattern=filename_pattern,file_range=filerange,data_type=DATA_TYPE) 
ld.keylist=channel_list

for i_file in range(len(ld.file_list)):
	print(f"Processing file #_{filerange_low+i_file}, {i_file+1}/{len(ld.file_list)}")
	try:
		res = ld.load_next_nfiles(1)

		tg = trigger(res, channel_config, trigger_channels, trigger_threshold_set, TTL_THRESHOLD=TTL_THRESHOLD, INRUN_RANDOM=INRUN_RANDOM,
					 WINDOW_TEMPLATE=WINDOW_TEMPLATE, USE_GAUS_KERNEL=USE_GAUS_KERNEL,
					 gauss_sigma=GAUSS_SIGMA, BYPASS_HIGH_ENERGY_IN_FILTER=BYPASS_HIGH_ENERGY_IN_FILTER,
					 trigger_type=1, align_max=True, filter_kernels=filter_kernels, PSD=filter_PSDs, OF_LPF=OF_LPF,
					 pre_trig=PRETRIG_CONF, post_trig=POSTTRIG_CONF, pre_trig_kernel=PRETRIG_CONF, post_trig_kernel=POSTTRIG_CONF, fs=Fs,
					 keep_metadata=True, verbose=True, use_ttl_falling_edge=use_ttl_falling_edge, remove_trigger_offset=REMOVE_TRIGGER_OFFSET)
		Traces, trigger_result_new = tg.run_trigger()
		n_events=len(trigger_result_new["trig_loc"])
		print(f"  {n_events} events triggered,")
	except Exception as e:
		print("Trigger exception:",e)
		break

	print("   Calculating RQs...")
	RQ_new = rq.process_traces(Traces, trigger_result_new)
	RQ_new["trig_fileidx"] = np.repeat(filerange_low+i_file, n_events).astype(np.int16)

	if i_file==0:
		trigger_result = trigger_result_new
		RQs = RQ_new
	else:
		trigger_result = append_dicts(trigger_result,trigger_result_new)
		RQs= append_dicts(RQs, RQ_new)

	for ch in np.unique(trigger_result_new["trig_ch"]):
		print("	* ",ch, sum(trigger_result_new["trig_ch"]==ch))
	

# Combine RQ with trigger info, save
for key in trigger_result:
	RQs[key]=trigger_result[key]
# dumping the config file as a string into the RQ dictionary
RQs['config'] = config_str
joblib.dump(RQs,filename_RQ) 

