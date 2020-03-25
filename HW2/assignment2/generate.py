#%%
import random
import os
import numpy as np
import collections
import torch

from solution import RNN, GRU


def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

BATCH_SIZE = 128
first_words = torch.LongTensor(BATCH_SIZE).random_(0, 10000)


#%% Generation for RNN


#---------------PARAMETERS


MODEL_PATH = '/Users/mlizaire/Codes/IFT6135/HW2/assignment2/results/results/job_6248705_3p1/results/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=1.0_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.8_num_epochs=40_save_best_save_dir=results_0'
EMB_SIZE = 200
HIDDEN_SIZE = 512
BATCH_SIZE = 128
VOCAB_SIZE = 10000
NUM_LAYERS = 2
DP_KEEP_PROB = 0.8
SEQ_LEN = 35
GENERATED_SEQ_LEN = 34

#--------------- LOAD MODEL


load_path = os.path.join(MODEL_PATH, 'best_params.pt')
model = RNN(emb_size=EMB_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    seq_len=SEQ_LEN,
                    batch_size=BATCH_SIZE,
                    vocab_size=VOCAB_SIZE,
                    num_layers=NUM_LAYERS,
                    dp_keep_prob=DP_KEEP_PROB)

model.load_state_dict(torch.load(load_path, map_location='cpu'))
hidden = model.init_hidden()
model.eval()

#--------------- GENERATE SAMPLES

first_words = torch.LongTensor(BATCH_SIZE).random_(0, 10000)
# samples = model.generate(torch.zeros(BATCH_SIZE).to(torch.long), hidden, generated_seq_len=GENERATED_SEQ_LEN)
samples = model.generate(first_words, hidden, generated_seq_len=GENERATED_SEQ_LEN)

#-------------- CONVERTING TO WORDS
data_path  = "/Users/mlizaire/Codes/IFT6135/HW2/assignment2/data/"
filename = os.path.join(data_path, "ptb.train.txt")
word_2_id, id_2_word = _build_vocab(filename)

sequences = []

print("THIS IS RNN")
for i in range(15):
    word_sequence = []
    id_sequence = np.array(torch.t(samples)[i])
    for index in id_sequence:
        word = id_2_word[int(index)]
        word_sequence.append(word)
    print("First word : ", id_2_word[int(first_words[i])])
    print("Rest of sequence : ", word_sequence)
    sequences.append(word_sequence)
# print(sequences)

# H = []
# s_1 = np.array(torch.t(samples)[0])
# H.append([id_2_word[int(idi)] for idi in s_1])
# print("First word", first_words[:10] )
# print(H)


###GENERATION FOR GRU

#---------------PARAMETERS

MODEL_PATH = "/Users/mlizaire/Codes/IFT6135/HW2/assignment2/results/results/job_6253799_3p2/results/GRU_ADAM_model=GRU_optimizer=ADAM_initial_lr=0.001_batch_size=128_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.5_num_epochs=40_save_best_save_dir=results_0"
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
                    dp_keep_prob=0.5)

model.load_state_dict(torch.load(load_path, map_location='cpu'))
hidden = model.init_hidden()
model.eval()

#--------------- GENERATE SAMPLES

# first_words = torch.LongTensor(BATCH_SIZE).random_(0, 10000)
# samples = model.generate(torch.zeros(BATCH_SIZE).to(torch.long), hidden, generated_seq_len=GENERATED_SEQ_LEN)
samples = model.generate(first_words, hidden, generated_seq_len=GENERATED_SEQ_LEN)

#-------------- CONVERTING TO WORDS
data_path  = "/Users/mlizaire/Codes/IFT6135/HW2/assignment2/data/"
filename = os.path.join(data_path, "ptb.train.txt")
word_2_id, id_2_word = _build_vocab(filename)

sequences = []

print("THIS GRU")
for i in range(15):
    word_sequence = []
    id_sequence = np.array(torch.t(samples)[i])
    for index in id_sequence:
        word = id_2_word[int(index)]
        word_sequence.append(word)
    print("First word : ", id_2_word[int(first_words[i])])
    print("Rest of sequence : ", word_sequence)
    sequences.append(word_sequence)