import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage as sp
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ----------------------------------- Plot generic parameters -------------------------

mpl.rcdefaults()
mpl.rcParams['mathtext.default']= 'regular'
mpl.rcParams['font.size'] = 24.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.weight'] = "normal"
mpl.rcParams['axes.labelsize'] = 24.
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.major.width'] = 0.6
mpl.rcParams['axes.linewidth'] = 0.6
mpl.rcParams['pdf.fonttype'] = 3

mpl.rcParams["xtick.minor.visible"] = "on"
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.top"] = "on"
mpl.rcParams["xtick.major.size"] = 8
mpl.rcParams["xtick.minor.size"] = 5

mpl.rcParams["ytick.minor.visible"] = "on"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.major.size"] = 8
mpl.rcParams["ytick.minor.size"] = 5
mpl.rcParams["ytick.right"] = "on"

minorLocatorx = MultipleLocator(12.5)
minorLocatory = MultipleLocator(25)

# ---------------------------------load data ----------------------------------------------


exp_location = {}
exp_location["3.4.1"] = "/Users/mlizaire/Codes/IFT6135/HW2/assignment2/results/results/job_6253821_3p4/results/TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=6_dp_keep_prob=0.9_num_epochs=20_save_dir=results_0/learning_curves.npy"
exp_location["3.4.2"] = "/Users/mlizaire/Codes/IFT6135/HW2/assignment2/results/results/job_6253821_3p4/results/TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.9_num_epochs=20_save_dir=results_0/learning_curves.npy"
exp_location["3.4.3"] = "/Users/mlizaire/Codes/IFT6135/HW2/assignment2/results/results/job_6253821_3p4/results/TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=2048_num_layers=2_dp_keep_prob=0.6_num_epochs=20_save_dir=results_0/learning_curves.npy"
exp_location["3.4.4"] = "/Users/mlizaire/Codes/IFT6135/HW2/assignment2/results/results/job_6253821_3p4/results/TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=1024_num_layers=6_dp_keep_prob=0.9_num_epochs=20_save_dir=results_0/learning_curves.npy"
files = {}

files["3.4.1"] = [exp_location["3.4.1"], "crimson", "Model 3.4.1"]
files["3.4.2"] = [exp_location["3.4.2"], "darkorange", "Model 3.4.2"]
files["3.4.3"] = [exp_location["3.4.3"],"purple", "Model 3.4.3"]
files["3.4.4"] = [exp_location["3.4.4"], "royalblue", "Model 3.4.4"]

M = files.keys()

train_ppls = {}
val_ppls = {}
train_losses = {}
val_losses = {}
times = {}

for m in M :
	file = files[m]
	filename = file[0]
	Data =  np.load(filename, allow_pickle=True)[()]
	
	train_ppls[m] = Data["train_ppls"]
	train_losses[m] = Data["train_losses"]
	val_ppls[m] = Data["val_ppls"]
	times[m] = Data["times"]

#---------------------------------------- Create plot --------------------------------
fig , axes = plt.subplots(1,1,figsize=(7,7)) 
fig.subplots_adjust(left=0.18, right=.9, bottom=0.15, top=0.9);
axes.set_ylim([0, 1200])
axes.set_xlim([0, 20])
# # axes.xaxis.set_minor_locator(minorLocatorx)
# # axes.yaxis.set_minor_locator(minorLocatory)

axes.set_xlabel("Epochs")
axes.set_ylabel("PPL")

yy, =axes.plot(np.arange(20), train_ppls["3.4.1"], color ="k", lw =1, label = "Train")
zz, =axes.plot(np.arange(20), val_ppls["3.4.1"], color ="k", ls="--", lw =1, label = "Valid" )

for m in M:
	file = files[m]
	print("m=",m)
	print(np.cumsum(times[m][:7]))
	print(train_ppls[m][7])

	axes.plot(np.arange(20), train_ppls[m], color =file[1], label = file[2] )
	axes.plot(np.arange(20), val_ppls[m], color =file[1], ls="--" )
leg1 = axes.legend(loc=1)
leg2 = axes.legend(handles=[yy, zz], labels =['Train', 'Valid'],  loc=2, fontsize=14, frameon=False)

axes.add_artist(leg1)

fig.savefig("./Figures/PPLvsEPOCHS_3p4.pdf")
plt.show()

# #---------------------------------------- Create plot --------------------------------
fig , axes = plt.subplots(1,1,figsize=(7,7)) 
fig.subplots_adjust(left=0.18, right=.9, bottom=0.15, top=0.9);
axes.set_ylim([0, 1200])
axes.set_xlim([0, 800])
# # axes.xaxis.set_minor_locator(minorLocatorx)
# # axes.yaxis.set_minor_locator(minorLocatory)

axes.set_xlabel("wall-clock-time (s)")
axes.set_ylabel("PPL")

yy, =axes.plot(np.cumsum(times["3.4.1"]), train_ppls["3.4.1"], color ="k", lw =1, label = "Train")
zz, =axes.plot(np.cumsum(times["3.4.1"]), val_ppls["3.4.1"], color ="k", ls="--", lw =1, label = "Valid" )

for m in M:
	file = files[m]

	axes.plot(np.cumsum(times[m]), train_ppls[m], color =file[1], label = file[2] )
	axes.plot(np.cumsum(times[m]), val_ppls[m], color =file[1], ls="--" )
leg1 = axes.legend(loc=1)
leg2 = axes.legend(handles=[yy, zz], labels =['Train', 'Valid'],  loc=2, fontsize=14, frameon=False)

axes.add_artist(leg1)

fig.savefig("./Figures/PPLvsWCT_3p4.pdf")

plt.show()




