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


#------------------- Define device
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


#------------------ LOAD DATA
# data_path  = "/Users/mlizaire/Codes/IFT6135/HW2/assignment2/data/"
data_path  = "data"
raw_data = ptb_raw_data(data_path)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data




# #---------------PARAMETERS
# MODEL_PATH = "/Users/mlizaire/Codes/IFT6135/HW2/assignment2/results/results/job_6253799_3p2/results/GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=40_save_best_save_dir=results_0"
MODEL_PATH = "results/results/job_6253799_3p2/results/GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=40_save_best_save_dir=results_0"

# EMB_SIZE = 200
# HIDDEN_SIZE = 512
BATCH_SIZE = 128
# VOCAB_SIZE = 10000
# NUM_LAYERS = 2
# DP_KEEP_PROB = 0.5
# SEQ_LEN = 35
GENERATED_SEQ_LEN = 34

#--------------- LOAD MODEL
load_path = os.path.join(MODEL_PATH, 'best_params.pt')
model = GRU(emb_size=200,
                    hidden_size=512,
                    seq_len=35,
                    batch_size=128,
                    vocab_size=10000,
                    num_layers=2,
                    dp_keep_prob=0.5, 
                    keep_hidden_states=True)

model.load_state_dict(torch.load(load_path, map_location='cpu'))

# hidden = model.init_hidden()
# hidden = hidden.to(device)


#-----------Loss function ------------------------------
loss_fn = nn.CrossEntropyLoss()

#----------------- Gradients function  --------------------------
def compute_gradients(model, data, loss_fn ): 
    
    print("hidden list", model.list_hidden_states)
    grads = []
    norm_grads = []

    model.eval()
    hidden = model.init_hidden()
    hidden = hidden.to(device)
    list_hidden_state = model.list_hidden_states

    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        if step == 0:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            print("avant", hidden[0][4][2])
            outputs, hidden_new = model(inputs, hidden)
            
            loss_final_timestep = loss_fn(outputs[-1], targets[-1])
            loss_final_timestep.backward(retain_graph=True)
            liste_test = [hidden, hidden]
            # print("TESTER", tester.requires_grad)
            # print("TESTER", tester[0, 0,:,:].grad)
            # print("out", torch.argmax(outputs.grad, dim=2))
            # print("out", outputs.grad.shape)
            print("list", model.list_hidden_states[1][34].grad.shape)
            # print("new list", model.new_list_hidden_states[34][1,0,0])
            # print("test", model.list_hidden_states)
            # print("encore", hidden_new.grad[1])
            print("Equal??", torch.equal(model.list_hidden_states[1][34], hidden_new[1]))
            print("moy", torch.mean(model.list_hidden_states[0][34].grad, dim=0).shape)


            batch_avg_list_hidden_states = torch.zeros(model.num_layers, model.seq_len,model.hidden_size)
            for layer in np.arange(model.num_layers):
                for t in np.arange(model.seq_len):
                    # print("kk",torch.mean(model.list_hidden_states[layer][t].grad, dim=0).shape)
                    batch_avg_list_hidden_states[layer][t] = torch.mean(model.list_hidden_states[layer][t].grad, dim=0)
            print("check shape", batch_avg_list_hidden_states[0].shape)
            concat_avg_list_hidden_states = torch.cat((batch_avg_list_hidden_states[0],batch_avg_list_hidden_states[1]), dim=1)
            print("check concat", concat_avg_list_hidden_states.shape)
            print("check norm", torch.norm(concat_avg_list_hidden_states, dim = 1).shape)
        
            return torch.norm(concat_avg_list_hidden_states, dim = 1)


##----------------------
# norm_gradients = compute_gradients(model, train_data, loss_fn=loss_fn )
# t = np.arange(model.seq_len)



# plt.plot(t, norm_gradients.tolist(), "-o")
# print("Here is you figure!")
# plt.show()
          



def plot_loss_vs_timestep(models_dic, grad_dic):
    mpl.rcParams['font.size'] = 24.
    mpl.rcParams['font.family'] = 'Arial'

    t = np.arange(35)

    fig , axes = plt.subplots(1,1,figsize=(7,7)) 
    fig.subplots_adjust(left=0.18, right=.9, bottom=0.15, top=0.9);
    axes.set_ylim([0, 0.0016])
    axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes.set_xlim([0, 35])

    axes.set_xlabel(r"Time-step ($t$)")
    axes.set_ylabel(r"||$\nabla_t \mathcal{L}_T$||")

    for archi in models_dic:
        axes.plot(t, grad_dic[archi].tolist(), "-o", color = models_dic[archi][2], label = str(archi) )
    
    axes.legend()
    plt.show()
    fig.savefig("./Figures/gradientsLoss.pdf")
    plt.close()



def main():
    print("Executing ||Main|| to compute gradients")
    
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
    
    for archi in models_dic:
        print('||Main||  Loading ', str(archi))
        model = models_dic[archi][0]
        load_path = os.path.join(models_dic[archi][1], 'best_params.pt')
        model.load_state_dict(torch.load(load_path, map_location='cpu'))

        loss_fn = nn.CrossEntropyLoss()

        print('||Main||  Computing gradients ...')
        grad_dic[archi] = compute_gradients(model, train_data, loss_fn=loss_fn )

    print('||Main||  Ploting figure')
    plot_loss_vs_timestep(models_dic, grad_dic)

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
        




