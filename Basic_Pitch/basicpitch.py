'''
Program that gives transcription of melody, given its bpm and bar count
output: list of list, where each sub list represents the notes played in that bar in the units of sixteenth note
i.e each sublist will have sixteen elements, representing the sixteen notes that are being played for that bar
'''
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from pydub import AudioSegment
import pretty_midi as pm
import librosa as lr


# audio_file_name = "sjsa.wav"
# melody_audio = AudioSegment.from_file(audio_file_name, format='wav')

def transcribeAudio(audio_file, num_bars, bpm = 120):

    sixteenth_duration = 15 / bpm
    # num_bars = 10
    bar_wise_notes = [['rest'] * 16 for _ in range(num_bars)]


    model_output, midi_data, note_events = predict(audio_file, minimum_note_length= sixteenth_duration * 1000)

    # print(midi_data)

    # print("------------------------------------------------------------------------------------------")
    count = 0

    # print(note_events)
    for instrument in midi_data.instruments:
        # print(instrument.name)
        #to keep track of last note
        
        prev_index = 0
        prev_note = 'rest' 
        prev_note_onset = 0
        

        for onset, note in zip(instrument.get_onsets(), instrument.notes):
            
            onset_beat_16_pos = round(onset / sixteenth_duration) #onset in terms of sixteenth note positions
            note_name = pm.note_number_to_name(note.pitch)

            prev_note_duration = onset_beat_16_pos - prev_note_onset #The duration of prev note, will be from its onset to the onset of this note
            index = round(onset_beat_16_pos // 16)

            start = prev_note_onset % 16
            end = start + prev_note_duration

            for i in range(start, end,1):
                # print(prev_note)
                bar_wise_notes[prev_index][i] = prev_note

            if(prev_note_onset == onset_beat_16_pos and prev_note_onset != 0):
                note_name = prev_note
            prev_note = note_name
            prev_note_onset = onset_beat_16_pos
            prev_index = index
            

            # print(f'time: {onset} | {onset_beat_16_pos} | {note_name}') # sixteenth note duration in DNR
            # print(index)
        
        last_note_duration = 16 * (prev_index + 1) - prev_note_onset #+1 to map it to num bar and associated sixteent note number to it
        # print(prev_index)
        # print(last_note_duration)
        start = prev_note_onset % 16 
        end = start + last_note_duration
        
        for i in range(start, end,1):
            bar_wise_notes[prev_index].append(prev_note)
            # print(prev_note)

    return bar_wise_notes



