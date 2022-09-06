import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import numpy as np


class MusicGenerator(object):
    def __init__(self, midi_coordinator, learning_rate = 0.005, num_timesteps = 15, batch_size = 100, num_of_epochs = 200):
        self.learning_rate = tf.constant(0.005, tf.float32)
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self._midi_coordinator = midi_coordinator

        # Define TF variables and placeholders
        self._visible_dim = 2*(midi_coordinator._upperBound – midi_coordinator._lowerBound)*num_timesteps
        self._hidden_dim = 50
        self._input  = tf.placeholder(tf.float32, [None, self._visible_dim], name="input")
        self._weights  = tf.Variable(tf.random_normal([self._visible_dim, self._hidden_dim], 0.01), name="weights")
        self._hidden_bias = tf.Variable(tf.zeros([1, self._hidden_dim],  tf.float32, name="hidden_bias"))
        self._visible_bias = tf.Variable(tf.zeros([1, self._visible_dim],  tf.float32, name="visible_bias"))
        
        visible_cdstates = self.gibsSampling(1) 
        hidden_states = self.callculate_state(tf.sigmoid(tf.matmul(self._input, self._weights) + self._hidden_bias)) 
        hidden_cdstates = self.callculate_state(tf.sigmoid(tf.matmul(visible_cdstates, self._weights) + self._hidden_bias)) 
        
        size = tf.cast(tf.shape(self._input)[0], tf.float32)
        weights_delta  = tf.multiply(self.learning_rate/size, tf.subtract(tf.matmul(tf.transpose(self._input), hidden_states), tf.matmul(tf.transpose(visible_cdstates), hidden_cdstates)))
        visible_bias_delta = tf.multiply(self.learning_rate/size, tf.reduce_sum(tf.subtract(self._input, visible_cdstates), 0, True))
        hidden_bias_delta = tf.multiply(self.learning_rate/size, tf.reduce_sum(tf.subtract(hidden_states, hidden_cdstates), 0, True))
        
        self._updates = [self._weights.assign_add(weights_delta), self._visible_bias.assign_add(visible_bias_delta), self._hidden_bias.assign_add(hidden_bias_delta)]
    
    def gibsSampling(self, number_of_iterations):
        counter = tf.constant(0)
        [_, _, visible_cdstates] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                             self.singleStep, [counter, tf.constant(number_of_iterations), self._input])
        
        # Stop tensorflow from propagating gradients back through the gibbs step
        visible_cdstates = tf.stop_gradient(visible_cdstates) 
        return visible_cdstates
    
        return tf.floor(probability + tf.random_uniform(tf.shape(probability), 0, 1))
    
    def singleStep(self, count, index, input_indexed):
        hidden_states = self.callculate_state(tf.sigmoid(tf.matmul(input_indexed, self._weights) + self._hidden_bias))
        visible_cdstates = self.callculate_state(tf.sigmoid(tf.matmul(hidden_states, tf.transpose(self._weights)) + self._visible_bias))
        return count+1, index, visible_cdstates
    
    def generateSongs(self, songs):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in tqdm(range(self.num_of_epochs)):
                for song in songs:
                    song = np.array(song)
                    song = song[:int(np.floor(song.shape[0]/self.num_timesteps)*self.num_timesteps)]
                    song = np.reshape(song, [song.shape[0]/self.num_timesteps, song.shape[1]*self.num_timesteps])
                    for i in range(1, len(song), self.batch_size): 
                        tr_x = song[i:i+self.batch_size]
                        sess.run(self._updates, feed_dict={self._input: tr_x})
        
            sample = self.gibsSampling(1).eval(session=sess, feed_dict={self._input: np.zeros((50, self._visible_dim))})
            for i in range(sample.shape[0]):
                if not any(sample[i,:]):
                    continue
                matrix = np.reshape(sample[i,:], (self.num_timesteps, 2*self._midi_coordinator._span))
                self._midi_coordinator.matrixToMidi(matrix, "song_{}".format(i))
