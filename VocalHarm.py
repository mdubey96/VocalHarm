### Copyright Mohit Dubey 2019 ###
### A Python App To Add Customizable Vocal Harmonies to a Melody
### key detection from https://gist.github.com/bmcfee/1f66825cef2eb34c839b42dddbad49fd


import numpy as np
import librosa
import scipy.linalg
import scipy.stats


inputfile = "lick.wav"
key = ""
interval = "-3"
balance = ".7"

data, fs = librosa.load(inputfile) 
newdata = np.zeros(len(data))
pitches, magnitudes = librosa.core.piptrack(y=data, sr=fs, fmin=75, fmax=1600, threshold = .95)

chroma = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

##Estimate Key
X = librosa.feature.chroma_stft(data)
X = np.abs(X)
X = X.mean(axis=1)

def ks_key(X):
    '''Estimate the key from a pitch class distribution
    
    Parameters
    ----------
    X : np.ndarray, shape=(12,)
        Pitch-class energy distribution.  Need not be normalized
        
    Returns
    -------
    major : np.ndarray, shape=(12,)
    minor : np.ndarray, shape=(12,)
    
        For each key (C:maj, ..., B:maj) and (C:min, ..., B:min),
        the correlation score for `X` against that key.
    '''
    X = scipy.stats.zscore(X)
    
    # Coefficients from Kumhansl and Schmuckler
    # as reported here: http://rnhart.net/articles/key-finding/
    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    major = scipy.stats.zscore(major)
    
    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    minor = scipy.stats.zscore(minor)
    
    # Generate all rotations of major
    major = scipy.linalg.circulant(major)
    minor = scipy.linalg.circulant(minor)
    
    return major.T.dot(X) #, minor.T.dot(X)


##Predict key if empty
if key == "":
 keyinfo = ks_key(X)
 maxind = np.argmax(keyinfo)
 key = np.take(chroma,maxind)


#Compute Pitches  TODO: Make this flexible to any mode or key
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
    newnote = np.take(keynotes,noteindex+(np.sign(int(interval))*(np.abs(int(interval))-1)),mode="wrap")
    distance = chroma.index(note) - chroma.index(newnote)
    print("Onset", onset_times[i])
    print("Root", note)
    print("Harmony", newnote)
    distance = np.sign(int(interval))*((12 - (np.sign(int(interval))*distance)) % 12)
    print("Interval",distance)
    if i < len(onset_frames)-1:    
     newdata[onset_samples[i]:onset_samples[i+1]] += librosa.effects.pitch_shift(data[onset_samples[i]:onset_samples[i+1]], fs, n_steps=distance)
    else:
     newdata[onset_samples[i]:] += librosa.effects.pitch_shift(data[onset_samples[i]:], fs, n_steps=distance) 

   
  output = data + float(balance) * newdata
  librosa.output.write_wav("output.wav", output, fs, norm=False)


computeharm(data, fs)

