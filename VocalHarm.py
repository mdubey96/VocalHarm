### Copyright Mohit Dubey 2019 ###
### A Python App To Add Customizable Vocal Harmonies to a Melody

import numpy as np
import librosa

inputfile = "lick.wav"
key = "C"
interval = "3"

data, fs = librosa.load(inputfile) 
newdata = np.zeros(len(data))
pitches, magnitudes = librosa.core.piptrack(y=data, sr=fs, fmin=75, fmax=1600, threshold = .95)

#Compute Pitches  TODO: Make this flexible to any mode or key
chroma = ["G","G#","A","A#","B","C","C#","D","D#","E","F","F#"]
keyindex = chroma.index(key)
noteindices = [keyindex,keyindex+2,keyindex+4,keyindex+5,keyindex+7,keyindex+9,keyindex+11]
keynotes = np.take(chroma,noteindices,mode='wrap')
print("Notes in Key")
print(keynotes)

def computeharm(data, sr):
 ## find onsets of pitches
 onset_frames = librosa.onset.onset_detect(y=data, sr=fs)
 onset_times = librosa.frames_to_time(onset_frames, sr=fs)
 onset_samples = librosa.core.frames_to_samples(onset_frames) 
 onset_env = librosa.onset.onset_strength(y=data, sr=fs)
 
## extract pitches
 for i, element in enumerate(onset_frames):
  index = magnitudes[:, onset_frames[i]].argmax()
  pitch = pitches[index, onset_frames[i]]
  note = librosa.core.hz_to_note(pitch, octave=False, cents=False)
  keynoteslist = keynotes.tolist()
  if note in keynoteslist:
    noteindex = keynoteslist.index(note)
    newnote = np.take(keynotes,noteindex+int(interval)-1,mode="wrap")
  
    distance = chroma.index(note) - chroma.index(newnote)
    print("Onset", onset_times[i])
    print("Root", note)
    print("Harmony", newnote)
    distance = (12 - distance) % 12
    print("Interval",distance)
    if i < len(onset_frames)-1:    
     newdata[onset_samples[i]:onset_samples[i+1]] += librosa.effects.pitch_shift(data[onset_samples[i]:onset_samples[i+1]], fs, n_steps=distance)
    else:
     newdata[onset_samples[i]:] += librosa.effects.pitch_shift(data[onset_samples[i]:], fs, n_steps=distance) 

   
  output = data +  newdata
  librosa.output.write_wav("output.wav", output, fs, norm=False)


computeharm(data, fs)

