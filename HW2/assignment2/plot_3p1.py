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

data = np.load("./results/results/job_6248705_3p1/results/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_best_save_dir=results_0/learning_curves.npy", allow_pickle=True)[()]


exp_location = {}
exp_location["3.1.1"] = "./results/results/job_6248705_3p1/results/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_best_save_dir=results_0/learning_curves.npy"
exp_location["3.1.2"] = "./results/results/job_6248705_3p1/results/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=20_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_dir=results_0/learning_curves.npy"
exp_location["3.1.3"] = "./results/results/job_6248705_3p1/results/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=10.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_dir=results_0/learning_curves.npy"
exp_location["3.1.4"] = "./results/results/job_6248705_3p1/results/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_dir=results_0/learning_curves.npy"
exp_location["3.1.5"] = "./results/results/job_6248705_3p1/results/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_dir=results_0/learning_curves.npy"
files = {}

files["3.1.1"] = [exp_location["3.1.1"], "crimson", "Model 3.1.1"]
files["3.1.2"] = [exp_location["3.1.2"], "darkorange", "Model 3.1.2"]
files["3.1.3"] = [exp_location["3.1.3"],"purple", "Model 3.1.3"]
files["3.1.4"] = [exp_location["3.1.4"], "royalblue", "Model 3.1.4"]
files["3.1.5"] = [exp_location["3.1.5"], "limegreen", "Model 3.1.5"]
# files["Bi2201"] = ["../Analyse/Bi2p05Sr1p95_n9.txt", 0, 2, "crimson", 1220]
# files["Bi2212"] = ["../Analyse/rho_xx_vs_T_Bi2212_p23.txt", 0, 3, "limegreen" , 774]
# files["NdLSCO"] = ["../Analyse/Tsweep_NdLSCO_0p24_H16T_bis.dat", 0, 2,"royalblue", 664]

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
# fig , axes = plt.subplots(1,1,figsize=(7,7)) 
# fig.subplots_adjust(left=0.18, right=.9, bottom=0.15, top=0.9);
# axes.set_ylim([0, 1200])
# axes.set_xlim([0, 40])
# # # axes.xaxis.set_minor_locator(minorLocatorx)
# # # axes.yaxis.set_minor_locator(minorLocatory)

# axes.set_xlabel("Epochs")
# axes.set_ylabel("PPL")

# yy, = axes.plot(np.arange(40), train_ppls["3.1.1"], color ="k", lw =1)
# zz, = axes.plot(np.arange(40), val_ppls["3.1.1"], color ="k", ls="--", lw =1 )

# for m in M:
# 	file = files[m]

# 	axes.plot(np.arange(40), train_ppls[m], color =file[1], label = file[2] )
# 	axes.plot(np.arange(40), val_ppls[m], color =file[1], ls="--" )
# 	# print("m",m)
# 	# print(train_ppls[m][39])
# 	# print(train_ppls[m][6])

# leg1 = axes.legend(loc=1)
# leg2 = axes.legend(handles=[yy, zz], labels =['Train', 'Valid'],  loc=2, fontsize=14, frameon=False)

# axes.add_artist(leg1)

# fig.savefig("./Figures/PPLvsEPOCHS_3p1.pdf")
# plt.show()


# #---------------------------------------- Create plot --------------------------------
fig , axes = plt.subplots(1,1,figsize=(7,7)) 
fig.subplots_adjust(left=0.18, right=.9, bottom=0.15, top=0.9);
axes.set_ylim([0, 1200])
axes.set_xlim([0, 3500])
# # axes.xaxis.set_minor_locator(minorLocatorx)
# # axes.yaxis.set_minor_locator(minorLocatory)

axes.set_xlabel("wall-clock-time (s)")
axes.set_ylabel("PPL")

yy, = axes.plot(np.cumsum(times["3.1.1"]), train_ppls["3.1.1"], color ="k", lw =1)
zz, = axes.plot(np.cumsum(times["3.1.1"]), val_ppls["3.1.1"], color ="k", ls="--", lw =1 )

for m in M:
	file = files[m]
	print(len(times[m]))
	axes.plot(np.cumsum(times[m]), train_ppls[m], color =file[1], label = file[2] )
	axes.plot(np.cumsum(times[m]), val_ppls[m], color =file[1], ls="--" )
	print(np.cumsum(times[m][:39]))
	print(np.cumsum(times[m][:6]))
leg1 = axes.legend(loc=1)
leg2 = axes.legend(handles=[yy, zz], labels =['Train', 'Valid'],  loc=2, fontsize=14, frameon=False)

axes.add_artist(leg1)


# fig.savefig("./Figures/PPLvsWCT_3p1.pdf")

plt.show()




