import fnmatch
import os
import random
import re
import threading

import librosa
import sys
import copy
import numpy as np
from numpy import linalg
import tensorflow as tf
import scipy



def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    #print (files)
    return files



def wav2spec(filename):
    fs= 16000
    framelen = 512
    frameshift = 256

    _,audio = scipy.io.wavfile.read(filename)
    #audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    D =librosa.core.stft(audio,framelen, frameshift)
    D = np.transpose(D)
    amplitude= np.absolute(D)
    angle= np.angle(D)
    return amplitude,angle


def mk_audio(output,angle,fs,filename):
    framelen= 512
    frameshift = 256

    output = np.reshape(output, (output.shape[1], output.shape[2]))
    output_angle = np.reshape(angle, (angle.shape[1], angle.shape[2]))
    output_re = output * np.exp(1j * output_angle)
    output_re = np.transpose(output_re)
    x_r=librosa.core.istft(output_re, frameshift, framelen)

    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(filename, (x_r * maxv).astype(np.int16), fs)
    return x_r


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        #print(filename)
        absspec,angle = wav2spec(filename)
        yield absspec,angle, filename

def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 audio_test_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        self.test_files = find_files(audio_test_dir)
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.angle_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.sample_test_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.angle_test_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                        ['float32','float32','float32','float32'],
                        shapes=[(None, None),(None,None),(None,None),(None,None)])
        self.enqueue = self.queue.enqueue(\
		[self.sample_placeholder, self.angle_placeholder,\
		 self.sample_test_placeholder, self.angle_test_placeholder])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        print(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        #audio_list = []
        #iterator = load_generic_audio(self.audio_dir, self.sample_rate)
        #for audio in iterator:
        #  audio_list.append(audio)
        #print(type(audio_list))
        #print(type(audio_list[0]))
        #print(type(audio_list[0][0]))
        while not stop:
            #for audio_copy in audio_list:
                #audio = copy.deepcopy(audio_copy)
            iterator = load_generic_audio(self.audio_dir, self.sample_rate)
            for amplitude, angle, trainfile in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                testfile = self.test_files[random.randint(0, (len(self.test_files) - 1))]
                amplitude_test, angle_test = wav2spec(testfile)
		#print(testfile)
            	#np.savetxt(os.path.basename(testfile)+"amplitude.csv", amplitude_test,fmt="%.3f", delimiter=",")
            	#np.savetxt(os.path.basename(testfile)+"angle.csv", angle_test,fmt="%.3f", delimiter=",")

		#print(trainfile + "train")
            	#np.savetxt(os.path.basename(trainfile)+"-train-amplitude.csv", amplitude,fmt="%.3f", delimiter=",")
            	#np.savetxt(os.path.basename(trainfile)+"-train-angle.csv", angle,fmt="%.3f", delimiter=",")
                sess.run(self.enqueue,
                  feed_dict={self.sample_placeholder: amplitude,
                             self.angle_placeholder: angle,
                             self.sample_test_placeholder: amplitude_test,
                             self.angle_test_placeholder: angle_test})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
