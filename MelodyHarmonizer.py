from Basic_Pitch.basicpitch import *
from Rule_Based.rulebased import *
from LSTM import *
import sys

if __name__ == "__main__":
    # Check if the filename is provided as an argument
    if len(sys.argv) != 4:
        print("Usage: python script_name.py file_name bpm num_bars")
        exit()

    
    audio_file = sys.argv[1]
    bpm = int(sys.argv[2])
    num_bars = int(sys.argv[3])

    # bar_wise_melody = transcribeAudio(audio_file, num_bars, bpm)

    melody_audio, mixed_audio, chord_sequence_names = generateChords(audio_file, bpm, num_bars)
    mixed_audio.export('mixed_audio.wav', format='wav')
    print(chord_sequence_names)

        