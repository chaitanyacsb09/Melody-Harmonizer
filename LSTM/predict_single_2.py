import glob
import os
import ntpath
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from os import listdir
from keras.preprocessing import sequence

cuda_ = "cuda:2"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")
# device = "cpu"

#24x24 confusion matrix, for analysing the predictions
confusion_matrix = [[0] * 24 for _ in range(24)]


# Define the LSTM model class (similar to the one defined previously)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, units, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.units = units
        self.lstm = nn.LSTM(input_dim, units, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.lstm2 = nn.LSTM(units, units, batch_first=True)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dense_attention = nn.Linear(units, 1)
        self.dense_output = nn.Linear(units, output_dim)
    
    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        lstm_output = self.dropout1(lstm_output)
        lstm_output, _ = self.lstm2(lstm_output)
        lstm_output = self.dropout2(lstm_output)
        attention = self.dense_attention(lstm_output)
        attention = torch.softmax(attention, dim=1)
        attention = attention.permute(0, 2, 1)
        sent_representation = torch.bmm(attention, lstm_output)
        sent_representation = torch.sum(sent_representation, dim=1)
        probabilities = self.dense_output(sent_representation)
        return probabilities
    
# Load the model weights
def load_model_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set the model to evaluation mode
def giveChords(oneHots, chord_dictionary):
    chords = []
    for chord in oneHots:
        chordIdx = chord.index(1)
        chords.append(chord_dictionary[chordIdx])

    return chords

def giveChordIndices(oneHots, chord_dictionary):
    chordIndices = []
    for chord in oneHots:
        chordIdx = chord.index(1)
        chordIndices.append(chordIdx)
    
    return chordIndices

def giveAccuracy(predicted, actual):

    matching_elements = 0
    
    # Iterate through the vectors and compare elements at the same positions
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            matching_elements += 1
    
    # return (matching_elements / len(predicted)) * 100
    return matching_elements

def getInScaleScore(prediction_list):
    chords_in_scale = ['C:maj','D:maj', 'D:min',
                        'E:maj', 'E:min',
                        'F:maj', 'F:min',
                        'G:maj', 'G:min',
                        'G#:maj', 'A:min',
                        'A#:maj','B:min']
    inScaleCount = 0
    for chord in prediction_list:
        if chord in chords_in_scale:
            inScaleCount += 1

    return inScaleCount

def update_confusion_matrix(actual_chord_indices, predicted_chord_indices):
    for i, j in zip(actual_chord_indices, predicted_chord_indices):
        confusion_matrix[i][j] += 1

def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
    plt.figure(figsize=(20, 8))
    sns.set(font_scale=1.2)  

    
    sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt='g', cbar=False,
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    
    if save_path:
        plt.savefig(save_path, format='jpeg')

    plt.show()

def one_hot_encoding(length, one_index):
    """Return the one hot vector."""
    vectors = [0] * length
    vectors[one_index] = 1
    return vectors

def getNpy(csv_path):
        note_dictionary = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_dictionary = ['C:maj', 'C:min',
                        'C#:maj', 'C#:min',
                        'D:maj', 'D:min',
                        'D#:maj', 'D#:min',
                        'E:maj', 'E:min',
                        'F:maj', 'F:min',
                        'F#:maj', 'F#:min',
                        'G:maj', 'G:min',
                        'G#:maj', 'G#:min',
                        'A:maj', 'A:min',
                        'A#:maj', 'A#:min',
                        'B:maj', 'B:min']
        note_dict_len = len(note_dictionary)
        chord_dict_len = len(chord_dictionary)
        
        csv_ins = open(csv_path, 'r', encoding='utf-8')
        next(csv_ins)  # skip first line
        reader = csv.reader(csv_ins)

        note_sequence = []
        song_sequence = []  # list for each song(each npy file) in the test set
        song_Chords = [] #list for each song, chords
        pre_measure = None
        prev_measure_note = None
        numNotes = 0
        for line in reader:
            measure = int(line[0])
            chord = line[1]
            note = line[2]

            # find one hot index
            chord_index = chord_dictionary.index(chord)
            note_index = note_dictionary.index(note)

            one_hot_note_vec = one_hot_encoding(note_dict_len, note_index)
            one_hot_chord_vec = one_hot_encoding(chord_dict_len, chord_index)

            if pre_measure is None:  # case : first line
                note_sequence.append(one_hot_note_vec)
                song_Chords.append(one_hot_chord_vec)

            elif pre_measure == measure:  # case : same measure note
                note_sequence.append(one_hot_note_vec)

            else:  # case : next measure note
                song_sequence.append(note_sequence)
                song_Chords.append(one_hot_chord_vec)
                note_sequence = [one_hot_note_vec]
            pre_measure = measure


        song_sequence.append(note_sequence)  # case : last measure note

        combined = []
        combined.append(song_sequence)
        combined.append(song_Chords)

        return combined



def predict():
    chord_dictionary = ['C:maj', 'C:min',
                        'C#:maj', 'C#:min',
                        'D:maj', 'D:min',
                        'D#:maj', 'D#:min',
                        'E:maj', 'E:min',
                        'F:maj', 'F:min',
                        'F#:maj', 'F#:min',
                        'G:maj', 'G:min',
                        'G#:maj', 'G#:min',
                        'A:maj', 'A:min',
                        'A#:maj', 'A#:min',
                        'B:maj', 'B:min']
    input_dim = 12
    output_dim = 24
    units = 128
    model = LSTMModel(input_dim, output_dim, units)
    model.to(device)
    weights_dir = 'model_weights/'
    weights_files = listdir(weights_dir)
    
    for i, file in enumerate(weights_files):
        print(str(i) + " : " + file)
    
    file_number_weights = int(input('Choose the weights:'))
    weights_file = weights_files[file_number_weights]
    weights_path = os.path.join(weights_dir, weights_file)
    
    load_model_weights(model, weights_path)
    
    file_path = 'dataset/test_npy/*.npy'
    npy_files = glob.glob(file_path)
    index = int(input("Index Of Song: "))
    correctPredictions = 0
    totalPredictions = 0
    chords_in_scale = 0
    # print("Here")
    # return 
    # for song in npy_files:
    #     combined = np.load(song, allow_pickle=True)
    #     songSeq = combined[0]
    #     songChords = combined[1]
    #     note_sequence = sequence.pad_sequences(songSeq)
    #     note_sequence = torch.Tensor(note_sequence).to(device)
        
        
    #     # Predict
    #     prediction_list = []
    #     with torch.no_grad():
    #         net_output = model(note_sequence)
    #         chord_indices = net_output.argmax(dim=1)
    #         prediction_list = [chord_dictionary[i] for i in chord_indices]
        
    #     actualChords = giveChords(songChords, chord_dictionary)
    #     currCorrect  = giveAccuracy(prediction_list, actualChords)

    #     correctPredictions += currCorrect
    #     chords_in_scale += getInScaleScore(prediction_list)
    #     totalPredictions += len(actualChords)
    #     print(ntpath.basename(song))
    #     chord_indices = chord_indices.tolist()
    #     actual_chord_indices = giveChordIndices(songChords, chord_dictionary)

    #     # print(chord_indices)
    #     # print(actual_chord_indices)
    #     update_confusion_matrix(actual_chord_indices, chord_indices)
    #     # print(ntpath.basename(song), prediction_list)
    #     # break
    
    # print(f"Correct Predictions: {correctPredictions} | Inscale Predictions: {chords_in_scale} | Total Predictions: {totalPredictions}")
    # print(f"Accuracy: {(correctPredictions/totalPredictions) * 100}")
    # print(f'Inscale Score: {(chords_in_scale/totalPredictions) * 100}')
    # plot_confusion_matrix(confusion_matrix, chord_dictionary, save_path='confusion_matrix.jpg')
    # print(confusion_matrix)
    
    
    # song = npy_files[index]
    # combined = np.load(song, allow_pickle=True)
    combined = getNpy(sys.argv[1])
    # print(combined)
    songSeq = combined[0]
    songChords = combined[1]

    note_sequence = sequence.pad_sequences(songSeq)
    note_sequence = torch.Tensor(note_sequence).to(device)
    # song_chords = torch.Tensor(songChords)
    # print(song_chords)
    # exit()
    # Predict
    prediction_list = []
    with torch.no_grad():
        net_output = model(note_sequence)
        chord_indices = net_output.argmax(dim=1)
        prediction_list = [chord_dictionary[i] for i in chord_indices]
        
    actual_chords = giveChords(songChords, chord_dictionary)
    chords_in_scale += getInScaleScore(prediction_list)
    totalPredictions += len(actual_chords)
    # print(ntpath.basename(song), prediction_list)
    print("Predicted Chords")
    print(prediction_list)
    # s = ""
    # for x in prediction_list:
    #     s += f'| {x} |'
    # print(s)

    print("Actual Chords")
    print(actual_chords)

    print(f"Accuracy: {(giveAccuracy(prediction_list, actual_chords) / totalPredictions) * 100}")
    print(f'Inscale Score: {(chords_in_scale/totalPredictions) * 100}')

if __name__ == '__main__':
    predict()
