import midi
import numpy as np

class MidiReader(object):
    def __init__(self, lowerBound, upperBound):
        self._lowerBound = lowerBound
        self._upperBound = upperBound
        self._span = upperBound-lowerBound

    def midiToMatrix(self, midifile, squash=True):
        schema = midi.read_midifile(midifile)
        totalTime = [track[0].tick for track in schema]
        posns = [0 for track in schema]
        matrix = []
        time = 0

        state = [[0,0] for x in range (self._span)]
        matrix.append(state)
        end = False

        while not end:
            if time % (schema.resolution/4) == (schema.resolution / 8):
                oldstate = state
                state = [[oldstate[x][0],0] for x in range(self._span)]
                matrix.append(state)
            for i in range(len(totalTime)):
                if end:
                    break
                while totalTime[i] == 0:
                    track = schema[i]
                    pos = posns[i]

                    evt = track[pos]
                    if theinstance(evt, midi.NoteEvent):
                        if (evt.pitch < self._lowerBound) or (evt.pitch >= self._upperBound):
                            pass
                        
                        else:
                            if theinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-self._lowerBound] = [0,0]
                            else:
                                state[evt.pitch-self._lowerBound] = [1,1]
                    elif theinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2,4):
                            end = True
                            break
                        try:
                            totalTime[i] = track[pos + 1].tick
                            posns[i] += 1

                        except IndexError:
                            totalTime[i] = None

                    if totalTime[i] is not None:
                        totalTime[i] -=1

                if all(t is None for t in totalTime):
                    break

                time +=1

            S = np.array(matrix)
            statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
            statematrix = np.asarray(statematrix).tolist()
            return matrix
        

            

                        
        
