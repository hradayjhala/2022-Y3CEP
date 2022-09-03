from tqdm import tqdm
import glob
import numpy as np

class SongImporter(object):
    def __init__(self, path, midi_coordinator):
        self.path = path
        self._midi_coordinator = midi_coordinator

    def getSongs(self):
        files = glob.glob('{}/*.mid*'.format(self._path))
        songs = []
            for file in tqdm(files):
                try:
            song = np.array(self._midi_coordinator.midiToMatrix(file))
                    if np.array(song).shape[0] > 50:
                     songs.append(song)
            except Exception as e:
                raise e
        return songs 
