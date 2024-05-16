#Importing essential libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import numpy as np
from IPython.display import Audio
import sounddevice as sd
import essentia.standard as es
from pydub import AudioSegment
from pydub.playback import play


def getNote(frequency):
    """
    Function returning Note name for frequency
    """
    
    # A list of all the possible note names
    note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    ]
    # The reference frequency for A4 (440 Hz)
    A4_frequency = 440

    # Calculate the number of semitones away from A4
    diff = 0
    if frequency > 0:
        diff = math.log2(frequency / A4_frequency)
        
    semitones = 12 * (diff)

    # Calculate the index of the note in the list
    note_index = round(semitones) % 12

#     # Get the octave number
#     octave = 4 + (round(semitones) // 12)

    # Build the note name string
    note_name = note_names[note_index - 3]

    return note_name


def detectPitch(audio_sample, sample_rate):
    '''
    Function returning pitches for audio sample
    '''
    pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
    pitch_values, pitch_confidence = pitch_extractor(audio_sample)
    
    note = None
    if len(pitch_values) > 0 and len(pitch_confidence) > 0:
        pitch = pitch_values[np.argmax(pitch_confidence)]
        note = getNote(pitch)
#         print("Estimated pitch:", note)
    else:
        print("Pitch estimation failed.")

    return note

def getBarPoints(audio_data, sample_rate=44100, bpm=120, num_bars=4):
    """
        Function Returning the bar values
    """

    tempo, beatFrames = librosa.beat.beat_track(y = audio_data, sr = sample_rate, bpm=bpm, units='samples')

    # Plotting Waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y = audio_data, sr = sample_rate, x_axis = 's')
    
    #Adding Bar Lines
    beat_len = (60/bpm)
    bar_len = 4 * beat_len
    bar_points = []
    
    for i in range(0,num_bars):
        bar_points.append(i * bar_len)
        
    for bar in bar_points:
        plt.axvline(x=bar, color='r', linestyle='--')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()
    
    return bar_points

def getBarPitches(audio_data, bar_points, sample_rate=44100, bpm=120):
    """
    Funtion returning pitch for each first 1/4 note of a bar
    """
    bar_pitches = []
    beat_len = (60/bpm)
    for bar in bar_points:
        sample_start = int(bar * sample_rate)
        sample_end = int(sample_start + beat_len * sample_rate) #estimating the pitch for first 1/4 note
    #     sd.play(audioData[sampleStart: sampleEnd], samplerate = sampleRate)
    #     sd.wait()
        bar_pitches.append(detectPitch(audio_data[sample_start: sample_end], sample_rate))
    
    return bar_pitches


def generateChordSequence(bar_pitches):
    cMajChords = {
    "C": ["C", "E", "G"],
    "D": ["D", "F", "A"],
    "E": ["E", "G", "B"],
    "F": ["F", "A", "C"],
    "G": ["G", "B", "D"],
    "A": ["A", "C", "E"],
    "B": ["B", "D", "F"]
     }
    cMaj_chord_names = {
    "C": "C:maj",
    "D": "D:min",
    "E": "E:min",
    "F": "F:maj",
    "G": "G:maj",
    "A": "A:min",
    "B": "B:dim"}
    chordSequence = []
    chord_sequence_name = []
    for pitch in bar_pitches:
        chord = cMajChords.get(pitch, ["C", "E", "G"])
        chord_name = cMaj_chord_names.get(pitch, "C:maj")
        chordSequence.append(chord)
        chord_sequence_name.append(chord_name)
    
    return chordSequence, chord_sequence_name

def generateChordSequenceExtended(bar_pitches):
    cMajChords = {
    "C": ["C", "E", "G"],
    "D": ["D", "F", "A"],
    "E": ["E", "G", "B"],
    "F": ["F", "A", "C"],
    "G": ["G", "B", "D"],
    "A": ["A", "C", "E"],
    "B": ["B", "D", "F"],
#     "D#": ["D#", "G", "A#"],
    "D#": ["G#", "C", "D#"],
    "A#": ["A#", "D", "F"],
     "G#": ["G#", "C", "D#"]   
     }
    cMaj_chord_names = {
    "C": "C:maj",
    "D": "D:min",
    "E": "E:min",
    "F": "F:maj",
    "G": "G:maj",
    "A": "A:min",
    "B": "B:dim",
    "D#": "G#:maj",
    "A#": "A#:maj",
    "G#": "G#:maj"}
    chordSequence = []
    chord_sequence_name = []
    for pitch in bar_pitches:
        chord = cMajChords.get(pitch, ["C", "E", "G"])
        chord_name = cMaj_chord_names.get(pitch, "C:maj")
        chordSequence.append(chord)
        chord_sequence_name.append(chord_name)
    
    return chordSequence, chord_sequence_name

def create_chord_audio(chord, duration_ms, octave=4, chord_volume=0.8):
    sample_rate = 44100 
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    chord_audio_data = np.zeros(len(t))

    note_to_freq = {
        'C': 261.63,
        'C#': 277.18,
        'D': 293.66,
        'D#': 311.13,
        'E': 329.63,
        'F': 349.23,
        'F#': 369.99,
        'G': 392.00,
        'G#': 415.30,
        'A': 440.00,
        'A#': 466.16,
        'B': 493.88
    }


    note_to_freq = {note: freq * (2 ** (octave - 4)) for note, freq in note_to_freq.items()}

    for note in chord:
        freq = note_to_freq[note]
        note_audio_data = np.sin(2 * np.pi * freq * t)
        chord_audio_data += note_audio_data

    chord_audio_data = (chord_audio_data / np.max(np.abs(chord_audio_data)) * 32767).astype(np.int16)

    chord_audio = AudioSegment(
        chord_audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=chord_audio_data.dtype.itemsize,
        channels=1
    )

    chord_audio = chord_audio - chord_volume

    return chord_audio

def create_chord_audio_GTR(chord, duration_ms, chord_volume=10):
    
    sample_paths = {
    'C': 'GTR_sample_library/C.wav',
    'C#': 'GTR_sample_library/C#.wav',
    'D': 'GTR_sample_library/D.wav',
    'D#': 'GTR_sample_library/D#.wav',
    'E': 'GTR_sample_library/E.wav',
    'F': 'GTR_sample_library/F.wav',
    'F#': 'GTR_sample_library/F#.wav',
    'G': 'GTR_sample_library/G.wav',
    'G#': 'GTR_sample_library/G#.wav',
    'A': 'GTR_sample_library/A.wav',
    'A#': 'GTR_sample_library/A#.wav',
    'B': 'GTR_sample_library/B.wav',
    }
    # Create an empty audio segment
    chord_audio = AudioSegment.silent(duration=duration_ms)

    for note in chord:
        # Load the custom audio sample for each note
        note_audio = AudioSegment.from_file(sample_paths[note], format='wav')

        # Adjust the duration of the sample to match the chord duration
        note_audio = note_audio[:duration_ms]

        # Adjust the volume of the note
        note_audio = note_audio - chord_volume

        # Overlay the note onto the chord_audio
        chord_audio = chord_audio.overlay(note_audio)

    return chord_audio


def getMixedAudio(audio_file_name, bpm, chord_sequence_notes):
    
    chord_progression = AudioSegment.silent(duration=0)
    beat_len = (60/bpm)
    bar_len = 4 * beat_len
    bar_len = bar_len * 1000 #converted to ms
    for chord in chord_sequence_notes:
#         chord_audio = create_chord_audio(chord, bar_len, 4, 10)
        chord_audio = create_chord_audio_GTR(chord, bar_len, 10)
        chord_progression += chord_audio

    melody_audio = AudioSegment.from_file(audio_file_name, format='wav')

    mixed_audio = melody_audio.overlay(chord_progression, position=0)
    
    return mixed_audio


def generateChords(audio_file_name, bpm, num_bars):
    #Loading the audio File:
    audio_data, sample_rate = librosa.load(audio_file_name, sr=44100)
    
    #Getting Bar points and plotting the waveform with Bar Points
    bar_points = getBarPoints(audio_data, sample_rate, bpm=bpm, num_bars=num_bars)
    
    #Getting Pitches for each first 1/4 note of a bar
    bar_pitches = getBarPitches(audio_data, bar_points, 44100,bpm)
    
    #using the bar pitches, building the chord sequence
#     chord_sequence_notes, chord_sequence_names = generateChordSequence(bar_pitches)
    chord_sequence_notes, chord_sequence_names = generateChordSequenceExtended(bar_pitches)
    
    #generating mixed audio of given melody and chords
    mixed_audio = getMixedAudio(audio_file_name, bpm, chord_sequence_notes)
    melody_audio = AudioSegment.from_file(audio_file_name, format='wav')
    '''
    Returning the mixed audio with the generated chord sequence in name format
    '''
    return melody_audio, mixed_audio, chord_sequence_names
    