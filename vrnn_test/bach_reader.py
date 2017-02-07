import numpy as np

def bach_file_prep(file_dir='data/bach_choral/jsbach_chorals_harmony.data'):
    with open(file_dir) as f_in:
        content = f_in.readlines()
        bass_idx = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
        choral_names = {}
        chord_enc = {}
        chord_dec = {}  # not very elegant but works
        chord_count = []
        data = []
        for line in content:
            line = line.rstrip()
            line = line.split(',')

            # number choral index from 0 to 59
            if line[0] not in choral_names:
                choral_names[line[0]] = len(choral_names)
            line[0] = choral_names[line[0]]

            # cast event number to int
            line[1] = int(line[1])

            # map YES/NO to 1/0
            def map_yn(l):
                if l == 'YES':
                    return 1
                if l == ' NO':
                    return 0
                return l

            line = map(map_yn, line)

            # change bass note to number (keys above)
            line[-3] = bass_idx[line[-3]]

            # cast meter to int
            line[-2] = int(line[-2])

            # index chords
            chord = line[-1].lstrip()
            if chord not in chord_enc:
                chord_enc[chord] = len(chord_enc)
                chord_dec[len(chord_dec)] = chord
                chord_count.append(0)
            line[-1] = chord_enc[chord]
            chord_count[chord_enc[chord]] += 1

            data.append(line)

        data = np.asarray(data, dtype=int)
        l = sorted([(chord_count[k], chord_dec[k]) for k in range(len(chord_count))])[::-1]
        acc = 0
        for idx, i in enumerate(l):
            acc += i[0]
            print(i, acc, float(acc) / 5665, idx)
    return data, chord_enc, chord_dec

bach_file_prep()
