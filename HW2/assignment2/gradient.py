import random
import os
import numpy as np
import collections
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from solution_copy import RNN, GRU
from torch.autograd import Variable
from helper_functions import ptb_iterator, ptb_raw_data, repackage_hidden, init_device


#----------------- Gradients function  --------------------------
def compute_gradients(model, data, loss_fn, device ): 
    grads = []
    norm_grads = []

    model.eval()
    model.zero_grad()
    hidden = model.init_hidden()
    hidden = hidden.to(device)
    hidden = repackage_hidden(hidden)
    
    list_hidden_state = model.list_hidden_states

    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        if step == 1:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            
            outputs, hidden_new = model(inputs, hidden)
            
            loss_final_timestep = loss_fn(outputs[-1], targets[-1])
            loss_final_timestep.backward(retain_graph=True)

            print("  Averaging over the batch")
            batch_avg_list_hidden_states = torch.zeros(model.num_layers, model.seq_len, model.hidden_size)
            for layer in np.arange(model.num_layers):
                for t in np.arange(model.seq_len):
                    batch_avg_list_hidden_states[layer][t] = torch.mean(model.list_hidden_states[layer][t].grad, dim=0)
            
            print("  Concatanating the hidden layers")
            concat_avg_list_hidden_states = torch.cat((batch_avg_list_hidden_states[0],batch_avg_list_hidden_states[1]), dim=1)
            
            print("  Computing norms")
            norms_lay1 = torch.norm(batch_avg_list_hidden_states[0], dim = 1)
            norms_lay2 = torch.norm(batch_avg_list_hidden_states[1], dim = 1)
            norms = torch.norm(concat_avg_list_hidden_states, dim = 1)

            # t = np.arange(35)
            # plt.plot(t, norms_lay1.tolist())
            # plt.plot(t, norms_lay2.tolist())
            # plt.show()

            return norms, norms_lay1, norms_lay2


# -------------------- Plot function ------------------------------
def plot_loss_vs_timestep_notnormalised(models_dic, grad_dic):
    mpl.rcParams['font.size'] = 24.
    mpl.rcParams['font.family'] = 'Arial'
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

    t = np.arange(35)

    fig , axes = plt.subplots(1,1,figsize=(8,7)) 
    fig.subplots_adjust(left=0.18, right=.9, bottom=0.15, top=0.9);
    axes.set_ylim([0, 0.0016])
    axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes.set_xlim([0, 35])

    axes.set_xlabel(r"Time-step ($t$)")
    axes.set_ylabel(r"||$\nabla_t \mathcal{L}_T$||")

    for archi in models_dic:
        axes.plot(t, grad_dic[archi].tolist(), "-o", color = models_dic[archi][2], label = str(archi), clip_on=False)
    
    axes.legend(frameon=False)
    plt.show()
    fig.savefig("./Figures/gradientsLoss_notnormalised.pdf")
    plt.close()


# -------------------- Plot function ------------------------------
def plot_loss_vs_timestep(models_dic, grad_dic):
    mpl.rcParams['font.size'] = 24.
    mpl.rcParams['font.family'] = 'Arial'
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

    t = np.arange(35)

    fig , axes = plt.subplots(1,1,figsize=(8,7)) 
    fig.subplots_adjust(left=0.18, right=.9, bottom=0.15, top=0.9);
    axes.set_ylim([0, 1])
    axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes.set_xlim([0, 35])

    axes.set_xlabel(r"Time-step ($t$)")
    axes.set_ylabel(r"||$\nabla_t \mathcal{L}_T$||")

    for archi in models_dic:
        maximum = grad_dic[archi].max().item()
        minimum = grad_dic[archi].min().item()
        axes.plot(t, ((grad_dic[archi]- minimum)/(maximum-minimum )).tolist(), "-o", color = models_dic[archi][2], label = str(archi), clip_on=False)
    
    axes.legend(frameon=False)
    plt.show()
    fig.savefig("./Figures/gradientsLoss.pdf")
    plt.close()

# -------------------- Plot function ------------------------------
def plot_loss_per_layer_vs_timestep(models_dic, grad_dic_1, grad_dic_2):
    mpl.rcParams['font.size'] = 24.
    mpl.rcParams['font.family'] = 'Arial'
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

    t = np.arange(35)

    fig , axes = plt.subplots(1,1,figsize=(8,7)) 
    fig.subplots_adjust(left=0.18, right=.9, bottom=0.15, top=0.9);
    axes.set_ylim([0, 1])
    axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes.set_xlim([0, 35])

    axes.set_xlabel(r"Time-step ($t$)")
    axes.set_ylabel(r"||$\nabla_t \mathcal{L}_T$||")

    for archi in models_dic:
        maximum1 = grad_dic_1[archi].max().item()
        minimum1 = grad_dic_1[archi].min().item()
        axes.plot(t, ((grad_dic_1[archi]- minimum1) / (maximum1-minimum1)).tolist(), "-o", color = models_dic[archi][2], label = str(archi)+" Layer 1", clip_on=False)
        maximum2 = grad_dic_2[archi].max().item()
        minimum2 = grad_dic_2[archi].min().item()
        axes.plot(t, ((grad_dic_2[archi]- minimum2) / (maximum2-minimum2)).tolist(), "-o", color = models_dic[archi][2], mfc ="white", label = str(archi)+" Layer2", clip_on=False)
    
    # axes.legend(frameon=False)
    axes.text(2, 0.92, "RNN", color= "crimson")
    axes.text(2, 0.84, "GRU", color= "royalblue")
    axes.text(3.2, 0.73, "Layer 1", color= "k", fontsize=18)
    axes.text(3.2, 0.68, "Layer 2", color= "k", fontsize=18)
    axes.plot(2, 0.74, "-o", color="k")
    axes.plot(2, 0.70, "-o", color="k", mfc = "white")
    plt.show()
    fig.savefig("./Figures/gradientsLoss_2layers.pdf")
    plt.close()

# -------------------- Plot function ------------------------------
def plot_loss_per_layer_vs_timestep_notnormalised(models_dic, grad_dic_1, grad_dic_2):
    mpl.rcParams['font.size'] = 24.
    mpl.rcParams['font.family'] = 'Arial'
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

    t = np.arange(35)

    fig , axes = plt.subplots(1,1,figsize=(8,7)) 
    fig.subplots_adjust(left=0.18, right=.9, bottom=0.15, top=0.9);
    axes.set_ylim([0, 0.0013])
    axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes.set_xlim([0, 35])

    axes.set_xlabel(r"Time-step ($t$)")
    axes.set_ylabel(r"||$\nabla_t \mathcal{L}_T$||")

    for archi in models_dic:

        axes.plot(t, grad_dic_1[archi].tolist(), "-o", color = models_dic[archi][2], label = str(archi)+" Layer 1", clip_on=False)

        axes.plot(t, grad_dic_2[archi].tolist(), "-o", color = models_dic[archi][2], mfc ="white", label = str(archi)+" Layer2", clip_on=False)
    
    # axes.legend(frameon=False)
    axes.text(2, 0.0012, "RNN", color= "crimson")
    axes.text(2, 0.00112, "GRU", color= "royalblue")
    axes.text(3.2, 0.00098, "Layer 1", color= "k", fontsize=18)
    axes.text(3.2, 0.00092, "Layer 2", color= "k", fontsize=18)
    axes.plot(2, 0.001, "-o", color="k")
    axes.plot(2, 0.00095, "-o", color="k", mfc = "white")
    plt.show()
    fig.savefig("./Figures/gradientsLoss_2layers_notnormalised.pdf")
    plt.close()

# -------------------- Main function --------------------------

def main():
    print("||Main|| Begins")
    
    device = init_device()
    print('||Main||  Device :', device)
    
    print('||Main||  Loading Data')
    train_data, valid_data, test_data, word_to_id, id_2_word = ptb_raw_data(data_path="data")
    vocab_size = len(word_to_id)

    models_dic = {'RNN' : [RNN(emb_size = 200,
                              hidden_size = 512,
                              seq_len = 35,
                              batch_size = 128,
                              vocab_size = vocab_size,
                              num_layers = 2,
                              dp_keep_prob = 0.8, 
                              keep_hidden_states=True),
                              "results/results/job_6248705_3p1/results/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_best_save_dir=results_0",
                              "crimson"
                            ],

                  'GRU' : [GRU(emb_size = 200,
                              hidden_size = 512,
                              seq_len = 35,
                              batch_size = 128,
                              vocab_size = vocab_size,
                              num_layers = 2,
                              dp_keep_prob = 0.5, 
                              keep_hidden_states = True), 
                              "results/results/job_6253799_3p2/results/GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=40_save_best_save_dir=results_0",
                              "royalblue"
                            ]
                 }
    
    grad_dic = {'RNN' : None, 'GRU' : None}
    grad_dic_lay1 = {'RNN' : None, 'GRU' : None}
    grad_dic_lay2 = {'RNN' : None, 'GRU' : None}
    
    for archi in models_dic:
        print('||Main||  Loading ', str(archi))
        model = models_dic[archi][0]
        load_path = os.path.join(models_dic[archi][1], 'best_params.pt')
        model.load_state_dict(torch.load(load_path, map_location='cpu'))

        loss_fn = nn.CrossEntropyLoss()

        print('||Main||  Computing gradients ...')
        print(archi)
        grad_dic[archi], grad_dic_lay1[archi],  grad_dic_lay2[archi]  = compute_gradients(model, train_data, loss_fn=loss_fn, device= device )


    print('||Main||  Ploting figure')
    # plot_loss_vs_timestep(models_dic, grad_dic)
    # plot_loss_vs_timestep_notnormalised(models_dic, grad_dic)
    # plot_loss_per_layer_vs_timestep(models_dic, grad_dic_lay1, grad_dic_lay2 )
    plot_loss_per_layer_vs_timestep_notnormalised(models_dic, grad_dic_lay1, grad_dic_lay2 )

    print('||Main||  DONE ;)')


if __name__== '__main__':
    main()
#%%
# models_dic = {'RNN' : [RNN(emb_size = 200,
#                         hidden_size = 512,
#                         seq_len = 35,
#                         batch_size = 128,
#                         vocab_size = 10000,
#                         num_layers = 2,
#                         dp_keep_prob = 0.8, 
#                         keep_hidden_states=True),
#                         "results/results/job_6248705_3p1/results/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_best_save_dir=results_0"
#                     ],

#             'GRU' : [GRU(emb_size = 200,
#                         hidden_size = 512,
#                         seq_len = 35,
#                         batch_size = 128,
#                         vocab_size = 10000,
#                         num_layers = 2,
#                         dp_keep_prob = 0.5, 
#                         keep_hidden_states = True),
#                         "results/results/job_6253799_3p2/results/GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=40_save_best_save_dir=results_0" 
#                     ]
#             }

# for archi in models_dic:
#     print(models_dic[archi][1])
#     print(models_dic["RNN"][1])
        




