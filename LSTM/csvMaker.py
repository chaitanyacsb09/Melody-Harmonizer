import csv

num_measures = int(input("Enter the number of measures: "))

data = []

for measure in range(1, num_measures + 1):
    melody = input(f"For measure {measure}, enter the melody notes (space-separated): ").split()
    chord = input(f"For measure {measure}, enter the chord: ")

    for note in melody:
        data.append((measure, chord, note))

with open('melody_chord_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['measure', 'chord', 'note']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for measure, chord, note in data:
        writer.writerow({'measure': measure, 'chord': chord, 'note': note})

print("CSV file 'melody_chord_data.csv' has been created.")
