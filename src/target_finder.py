"""Target sequence predictor

This script builds a logistic regression classifier that predict if a sequence will be the target of a TF depending
on whether it contains the binding motif.
It has the following functions:
    * sliding_window: this splits an input DNA sequence into subsequences called kmers
    * bows_to_counts: this counts the number of kmers created from a sequence producing a kmer:count dictionary
    * make_dataset: this function will randomly generate several sequences for us and insert our target motif
    into half of those sequences.
    * predict: accepts a new sequence, processes it and predicts whether it has our binding motif
"""
import gradio
import numpy as np
import itertools
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import gradio as gr


def sliding_window(seq, window_size):
    """

    :param seq: DNA sequence
    :param window_size: kmer-size
    :return: a list of kmers for seq
    """
    bows = []
    for i in range(len(seq) - window_size + 1):
        bows.append(''.join(seq[i: i + window_size]))
    return bows


def bows_to_count(seqs, kmer_size=6):
    """

    :param seqs: a list of lists, where each list contains kmers generated from a sequence
    :param kmer_size: length of kmers
    :return: a list of dictionaries with key:value == kmer:count
    """
    bows_counts = {''.join(x): 0 for x in itertools.product(['A', 'C', 'G', 'T', 'N'], repeat=kmer_size)}
    bows_encoded = []
    for seq in seqs:
        bows_counts_copy = bows_counts.copy()
        for word in seq:
            bows_counts_copy[word] += 1
        bows_encoded.append(bows_counts_copy)
    return bows_encoded


def make_dataset(seq_length, dataset_size=500, kmer_size=6):
    """

    :param seq_length: length of our sequences
    :param dataset_size: number of random sequences to generate to build our model
    :param kmer_size:
    :return: dataset of dataset_size number of sequences
    """
    motif = 'GATCGGCT'
    pos_nuc = ['A', 'C', 'G', 'T', 'N']
    dataset = [[np.random.choice(pos_nuc, 1, p=[0.2375, 0.2375, 0.2375, 0.2375, 0.05])[0]
                for _ in range(seq_length)] for _ in range(dataset_size)]
    ran_idx = np.random.choice(dataset_size, size=int(dataset_size/2), replace=False)
    for idx in ran_idx:
        ran_start = np.random.choice(a=seq_length-len(motif), size=1)[0]
        dataset[idx][ran_start:ran_start+len(motif)] = list(motif)
    y = np.zeros(shape=(dataset_size,))
    y[ran_idx] = 1
    dataset_to_bows = [sliding_window(seq, window_size=kmer_size) for seq in dataset]

    x = bows_to_count(dataset_to_bows, kmer_size=kmer_size)
    x = pd.DataFrame(x).values
    return x, y


x, y = make_dataset(seq_length=100, dataset_size=3000, kmer_size=6)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
pipeline = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
pipeline.fit(x_train, y_train)
print(pipeline.score(x_test, y_test))


def predict(seq):
    """

    :param seq: new sequence given by the user from our web app
    :return: predictions
    """
    seq = seq.upper()
    inp = bows_to_count([sliding_window(seq=seq, window_size=6)], kmer_size=6)
    inp = pd.DataFrame(inp).values
    y = pipeline.predict(inp)[0]
    if y == 0:
        pred = 'No GATCGGCT-box'
    else:
        pred = 'Has GATCGGCT-box'
    return [pred, str(pipeline.predict_proba(inp).ravel()[int(y)])]


gui_inp = gr.inputs.Textbox(label='sequence', placeholder='sequence here....')
gui_out_pred = gr.components.Textbox(label='Predicted class')
gui_out_prob = gr.components.Textbox(label='Probability (confidence)')


gui = gradio.Interface(fn=predict,
                       inputs=gui_inp,
                       outputs=[gui_out_pred, gui_out_prob]).launch()



