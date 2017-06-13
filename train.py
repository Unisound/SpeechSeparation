"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
#import tensorflow.contrib.mpi as mpi
import ctypes
import scipy
#ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)

def _debug_print_func(val):
    print (val.shape)
    return False

def _debug_print_detail_func(val):
    print (val)
    return False

from speech_separation import cosSimilar,stft,istft
from speech_separation import SpeechSeparation, AudioReader, mu_law_decode, optimizer_factory
NUM_OF_FREQUENCY_POINTS = 257
BATCH_SIZE = 1
#DATA_DIRECTORY = './VCTK-Corpus'
DATA_DIRECTORY = './pinao-corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 2000
NUM_STEPS = int(1e6)
LEARNING_RATE = 1e-4
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 10
METADATA = False
N_SEQS = 10  # Number of samples to generate every time monitoring.
N_SECS = 3
SAMPLE_RATE = 16000
BITRATE = 16000
LENGTH = N_SECS*BITRATE
NUM_GPU = 1

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--num_gpus', type=int, default=NUM_GPU,
                        help='num of gpus.. Default: ' + str(NUM_GPU) + '.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--num_of_frequency_points', type=int, default=NUM_OF_FREQUENCY_POINTS,
                        help='num_of_frequency_points. Default: '
                             + str(NUM_OF_FREQUENCY_POINTS) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--test_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE,
                        help='sample rate for training. Default: ' + str(SAMPLE_RATE) + '.')
#    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
#                        help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
###############SAMPLE_RNN################
    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
           raise ValueError('Arg is neither `True` nor `False`')

    def check_non_negative(value):
        ivalue = int(value)
        if ivalue < 0:
             raise argparse.ArgumentTypeError("%s is not non-negative!" % value)
        return ivalue

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 1:
             raise argparse.ArgumentTypeError("%s is not positive!" % value)
        return ivalue

    def check_unit_interval(value):
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
             raise argparse.ArgumentTypeError("%s is not in [0, 1] interval!" % value)
        return fvalue
    '''
    # TODO: Fix the descriptions
    # Hyperparameter arguements:
    #parser.add_argument('--exp', help='Experiment name',
    #        type=str, required=False, default='_')
    '''
    parser.add_argument('--seq_len', help='How many samples to include in each\
            Truncated BPTT pass', type=check_positive, required=True)
    parser.add_argument('--big_frame_size', help='How many samples per big frame',\
            type=check_positive, required=True)
    parser.add_argument('--frame_size', help='How many samples per frame',\
            type=check_positive, required=True)
    parser.add_argument('--q_levels', help='Number of bins for quantization of\
            audio samples. Should be 256 for mu-law.',\
            type=check_positive, required=True)
    parser.add_argument('--rnn_type', help='GRU or LSTM', choices=['LSTM', 'GRU'],\
            required=True)
    parser.add_argument('--dim', help='Dimension of RNN and MLPs',\
            type=check_positive, required=True)
    parser.add_argument('--n_rnn', help='Number of layers in the stacked RNN',
            type=check_positive, choices=xrange(1,6), required=True)
    parser.add_argument('--emb_size', help='Size of embedding layer (> 0)',
            type=check_positive, required=True)  # different than two_tier
###############SAMPLE_RNN################
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    #logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    logdir = os.path.join(logdir_root, 'train')
    return logdir

def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def create_model(args):
  # Create network.
  net = SpeechSeparation(
    batch_size=args.batch_size,
    big_frame_size=args.big_frame_size,
    frame_size=args.frame_size,
    q_levels=args.q_levels,
    rnn_type=args.rnn_type,
    dim=args.dim,
    n_rnn=args.n_rnn,
    seq_len=args.seq_len,
    num_of_frequency_points=args.num_of_frequency_points,
    emb_size=args.emb_size)
  return net 

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  #print("================================")
  #for name in tower_grads:  
  #  for name2 in name:  
  #   print(name2) 
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

# GENERATE 
def create_gen_wav_para(net):
  with tf.name_scope('infe_para'):
    infe_para = dict()
    infe_para['infe_big_frame_inp'] = \
	tf.get_variable("infe_big_frame_inp", 
		[net.batch_size, net.big_frame_size,1], dtype=tf.float32)
    infe_para['infe_big_frame_outp'] = \
	tf.get_variable("infe_big_frame_outp", 
		[net.batch_size, net.big_frame_size/net.frame_size, net.dim], dtype=tf.float32)

    infe_para['infe_big_frame_outp_slices'] = \
	tf.get_variable("infe_big_frame_outp_slices", 
		[net.batch_size, 1, net.dim], dtype=tf.float32)
    infe_para['infe_frame_inp'] = \
	tf.get_variable("infe_frame_inp", 
		[net.batch_size, net.frame_size,1], dtype=tf.float32)
    infe_para['infe_frame_outp'] = \
	tf.get_variable("infe_frame_outp", 
		[net.batch_size, net.frame_size, net.dim], dtype=tf.float32)

    infe_para['infe_frame_outp_slices'] = \
	tf.get_variable("infe_frame_outp_slices", 
		[net.batch_size, 1, net.dim], dtype=tf.float32)
    infe_para['infe_sample_inp'] = \
	tf.get_variable("infe_sample_inp", 
		[net.batch_size, net.frame_size,1], dtype=tf.int32)

    infe_para['infe_big_frame_state'] = net.cell.zero_state(net.batch_size, tf.float32)
    infe_para['infe_frame_state']     = net.cell.zero_state(net.batch_size, tf.float32)

	#net.big_frame_gen(infe_para['infe_big_frame_inp'])
    tf.get_variable_scope().reuse_variables()
    infe_para['infe_big_frame_outp'], \
    infe_para['infe_final_big_frame_state'] = \
        net._create_network_BigFrame(num_steps = 1,
		big_frame_state = infe_para['infe_big_frame_state'],
           	big_input_sequences = infe_para['infe_big_frame_inp'])

	#net.frame_gen(infe_para['infe_big_frame_outp'], infe_para['infe_frame_inp'])
    infe_para['infe_frame_outp'], \
    infe_para['infe_final_frame_state'] = \
        net._create_network_Frame(num_steps = 1,
        	big_frame_outputs = infe_para['infe_big_frame_outp_slices'],
		frame_state = infe_para['infe_frame_state'],
        	input_sequences = infe_para['infe_frame_inp'])

	#net.sample_gen(infe_para['infe_frame_outp'],infe_para ['infe_sample_inp'])
    sample_out = \
        net._create_network_Sample(frame_outputs=infe_para['infe_frame_outp_slices'],
        			sample_input_sequences = infe_para['infe_sample_inp'])
    sample_out = \
	tf.reshape(sample_out, [-1, net.q_levels])
    # Cast to float64 to avoid bug in TensorFlow
    infe_para['infe_sample_outp'] = tf.cast(
        tf.nn.softmax(tf.cast(sample_out, tf.float64)), tf.float32)

    infe_para['infe_sample_decode_inp'] = \
	tf.placeholder(tf.int32)
    infe_para['infe_decode'] = \
	mu_law_decode(infe_para['infe_sample_decode_inp'], net.q_levels)

    return infe_para


def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    print ("logdir",logdir)
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = args.silence_threshold if args.silence_threshold > \
                                                      EPSILON else None
        gc_enabled = args.gc_channels is not None
        reader = AudioReader(
            args.data_dir,
            args.test_dir,
            coord,
            sample_rate=args.sample_rate,
            gc_enabled=gc_enabled,
            sample_size=args.sample_size,
            silence_threshold=silence_threshold)
        audio_batch = reader.dequeue(args.batch_size)

    net =  create_model(args)
########Multi GPU###########
    #'''
    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    global_step = tf.get_variable('global_step', 
		[], initializer = tf.constant_initializer(0), trainable=False)

#train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    optim = optimizer_factory[args.optimizer](
                    learning_rate=args.learning_rate,
                    momentum=args.momentum)
    #optim = mpi.DistributedOptimizer(optim)
    tower_grads = []
    losses = []
    speech_inputs_mix = []
    speech_inputs_1 = []
    speech_inputs_2 = []
    speech_state = []
    train_input_batch_rnn = []
    train_big_frame_state = []
    train_frame_state = []
    final_big_frame_state = []
    final_frame_state = []
    for i in xrange(args.num_gpus):
      speech_inputs_2.append(tf.Variable(
          		tf.zeros([net.batch_size, net.seq_len,args.num_of_frequency_points]), 
                        trainable=False , 
          		name="speech_batch_inputs",
          		dtype=tf.float32))
      speech_inputs_1.append(tf.Variable(
          		tf.zeros([net.batch_size, net.seq_len,args.num_of_frequency_points]), 
                        trainable=False , 
          		name="speech_batch_inputs",
          		dtype=tf.float32))
      speech_inputs_mix.append(tf.Variable(
          		tf.zeros([net.batch_size, net.seq_len,args.num_of_frequency_points]), 
                        trainable=False , 
          		name="speech_batch_inputs",
          		dtype=tf.float32))
      train_input_batch_rnn.append(tf.Variable(
          		tf.zeros([net.batch_size, net.seq_len,1]), 
                        trainable=False , 
          		name="input_batch_rnn",
          		dtype=tf.float32))
      speech_state.append(net.cell.zero_state(net.batch_size, tf.float32))
      train_big_frame_state.append(net.cell.zero_state(net.batch_size, tf.float32))
      train_frame_state.append    (net.cell.zero_state(net.batch_size, tf.float32))
      final_big_frame_state.append(net.cell.zero_state(net.batch_size, tf.float32))
      final_frame_state.append    (net.cell.zero_state(net.batch_size, tf.float32))
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(args.num_gpus):
        #sys.stdout.write("FLAGS.num_gpus:",FLAGS.num_gpus)
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('TOWER_%d' % (i)) as scope:
            # Create model.
            print("Creating model On Gpu:%d." % (i))
            loss, mask_state, output1, output2 = net.loss_SampleRnn(
          		speech_inputs_1[i],
          		speech_inputs_2[i],
          		speech_inputs_mix[i],
			speech_state[i],
          		train_input_batch_rnn[i],
          		train_big_frame_state[i],
          		train_frame_state[i],
                      	l2_regularization_strength=args.l2_regularization_strength)
            tf.get_variable_scope().reuse_variables()
            losses.append(loss)
            # Reuse variables for the next tower.
            trainable = tf.trainable_variables()
            for name in trainable:  
              print(name) 
            '''
            del_trainable_list=tf.get_collection(\
			tf.GraphKeys.TRAINABLE_VARIABLES,
			scope='SEEPCH_RNN_LAYER')
            trainable = list(set(trainable) - set(del_trainable_list))
            '''
            gradients = optim.compute_gradients(loss,trainable)
      				#aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
      				#aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            print("==========================")
            for name in gradients:  
              print(name) 
            #gradients = tf.gradients(loss[i], trainable)
            #debug_print_op = tf.py_func(_debug_print_detail_func, [loss], [tf.bool])
            #with tf.control_dependencies(debug_print_op):
            tower_grads.append(gradients)
            #tower_grads.append(grads)
    #debug_print_op = tf.py_func(_debug_print_func, [tower_grads], [tf.bool])
    #with tf.control_dependencies(debug_print_op):
    grad_vars = average_gradients(tower_grads)
    grads, vars = zip(*grad_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
    grad_vars = zip(grads_clipped, vars)
    #for name in grads:  
    #  print(name) 
    apply_gradient_op = optim.apply_gradients(grad_vars, global_step=global_step) 
    #apply_gradient_op = optim.apply_gradients(grads) 
    #'''
###################
    #infe_para = create_gen_wav_para(net)

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    # Set up session
    #tf_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    tf_config = tf.ConfigProto(\
		allow_soft_placement=True,\
	 	log_device_placement=False,\
                inter_op_parallelism_threads = 1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    #sess = mpi.Session(0,config=tf_config)

    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    #init = tf.global_variables_initializer()
    #sess.run(init)
    sess.run(tf.initialize_all_variables())

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step
    try:
        loss_sum = 0;
        for step in range(saved_global_step + 1, args.num_steps):
            start_time = time.time()
            idx_begin=0
            #inputs = sess.run(audio_batch)
            inputslist = [sess.run(audio_batch) for i in xrange(args.num_gpus)]
            #print(inputslist[0].shape)
            if 0 != inputslist[0][0].shape[1]%3 :
                raise ValueError("0 != inputslist[0].shape%3,")
            s_len=inputslist[0][0].shape[1]/3
            speeker_1  =inputslist[0][0][:,:s_len,:]
            speeker_2  =inputslist[0][0][:,s_len:s_len*2,:]
            speeker_mix=inputslist[0][0][:,-s_len:,:]

            angle_1    =inputslist[0][1][:,:s_len,:]
            angle_2    =inputslist[0][1][:,s_len:s_len*2,:]
            angle_mix  =inputslist[0][1][:,-s_len:,:]

            speeker_test_1  =inputslist[0][2][:,:s_len,:]
            speeker_test_2  =inputslist[0][2][:,s_len:s_len*2,:]
            speeker_test_mix=inputslist[0][2][:,-s_len:,:]

            angle_test_1    =inputslist[0][3][:,:s_len,:]
            angle_test_2    =inputslist[0][3][:,s_len:s_len*2,:]
            angle_test_mix  =inputslist[0][3][:,-s_len:,:]
            #print(speeker_0.shape,speeker_1.shape,speeker_mix.shape)

            final_big_s = []
            final_s = []
            for g in xrange(args.num_gpus):
              final_big_s.append(sess.run(net.initial_state))
              final_s.append(sess.run(net.initial_state))
            testall=None
            #for i in range(0, (args.sample_size - args.big_frame_size)/
            #                  (args.seq_len - args.big_frame_size)):
            inp_dict={}
            angle= None
            for g in xrange(args.num_gpus):
              inp_dict[speech_inputs_1[g]]  =speeker_1[:, :args.seq_len, :]
              inp_dict[speech_inputs_2[g]]  =speeker_2[:, :args.seq_len, :]
              inp_dict[speech_inputs_mix[g]]=speeker_mix[:, :args.seq_len, :]
              angle     = angle_mix[:, :args.seq_len, :]
              angle_test= angle_test_mix[:, :args.seq_len, :]
            idx_begin += args.seq_len-args.big_frame_size

            duration1 = time.time() - start_time
            loss_value,_, mask_state_value = \
            	sess.run([losses, apply_gradient_op,mask_state], feed_dict=inp_dict)
            for g in xrange(args.num_gpus):
              loss_sum += loss_value[g]/args.num_gpus
            #    break
            duration = time.time() - start_time
            if(step<100):
              #print('step {:d} - loss = {:.3f}, ({:.3f} sec/step, ({:.3f} sec)'
              #    .format(step, loss_sum, duration, duration1))
              #rank = sess.run(mpi.rank())
              #log_str = ('rank {%d} step {%d} - loss = {%0.3f}, ({%0.3f} sec/step,{%0.3f} sec/step)')%(rank, step, loss_sum, duration,duration1)
              log_str = ('step {%d} - loss = {%0.3f}, ({%0.3f} sec/step,{%0.3f} sec/step)')%(step, loss_sum, duration,duration1)
              logging.warning(log_str)
              loss_sum = 0;
            elif(0==step % 100):
              #print('step {:d} - loss = {:.3f}, ({:.3f} sec/step , ({:.3f} sec)'
              #    .format(step, loss_sum/100, duration,duration1))
              #rank = sess.run(mpi.rank())
              #log_str = ('rank {%d} step {%d} - loss = {%0.3f}, ({%0.3f} sec/step,{%0.3f} sec/step)')%(rank, step, loss_sum/100, duration,duration1)
              log_str = ('step {%d} - loss = {%0.3f}, ({%0.3f} sec/step,{%0.3f} sec/step)')%(step, loss_sum/100, duration,duration1)
              logging.warning(log_str)
              loss_sum = 0;
              #========================
            #'''
            elif(0==step % 5001):
              outp1,outp2 = \
              sess.run([output1,output2], feed_dict=inp_dict)
              fs= 16000
              framesz= 0.032
              hop= framesz*0.5
              print(outp1.shape,outp2.shape,angle.shape)
              outp1=np.reshape(outp1, (outp1.shape[1], outp1.shape[2]))
              outp_angle=np.reshape(angle, (angle.shape[1], angle.shape[2]))
              outp1_re=outp1*np.cos(outp_angle) + 1j*outp1*np.sin(outp_angle)
              outp1_re=np.column_stack((outp1_re,np.conj(outp1_re[:,1:-1].T[::-1].T)))
              x_r=istft(outp1_re, fs, (outp1.shape[1]+1)*256, hop)
              scipy.io.wavfile.write("speeker1_"+str(step)+".wav", fs, x_r)

              outp2=np.reshape(outp2, (outp2.shape[1], outp2.shape[2]))
              outp_angle=np.reshape(angle, (angle.shape[1], angle.shape[2]))
              outp2_re=outp2*np.cos(outp_angle) + 1j*outp2*np.sin(outp_angle)
              outp2_re=np.column_stack((outp2_re,np.conj(outp2_re[:,1:-1].T[::-1].T)))
              x_r=istft(outp2_re, fs, (outp2.shape[1]+1)*256, hop)
              scipy.io.wavfile.write("speeker2_"+str(step)+".wav", fs, x_r)

              #========================
              inp_dict={}
              for g in xrange(args.num_gpus):
                inp_dict[speech_inputs_1[g]] = speeker_test_1[:,:args.seq_len,:]
                inp_dict[speech_inputs_2[g]] = speeker_test_2[:,:args.seq_len,:]
                inp_dict[speech_inputs_mix[g]] = speeker_test_mix[:,:args.seq_len,:]
              outp1,outp2 = \
              sess.run([output1,output2], feed_dict=inp_dict)

              outp1=np.reshape(outp1, (outp1.shape[1], outp1.shape[2]))
              outp_angle=np.reshape(angle_test,(angle_test.shape[1],angle_test.shape[2]))
              outp1_re=outp1*np.cos(outp_angle) + 1j*outp1*np.sin(outp_angle)
              outp1_re=np.column_stack((outp1_re,np.conj(outp1_re[:,1:-1].T[::-1].T)))
              x_r=istft(outp1_re, fs, (outp1.shape[1]+1)*256, hop)
              scipy.io.wavfile.write("speeker1_test_"+str(step)+".wav", fs, x_r)

              outp2=np.reshape(outp2, (outp2.shape[1], outp2.shape[2]))
              outp_angle=np.reshape(angle_test,(angle_test.shape[1],angle_test.shape[2]))
              outp2_re=outp2*np.cos(outp_angle) + 1j*outp2*np.sin(outp_angle)
              outp2_re=np.column_stack((outp2_re,np.conj(outp2_re[:,1:-1].T[::-1].T)))
              x_r=istft(outp2_re, fs, (outp2.shape[1]+1)*256, hop)
              scipy.io.wavfile.write("speeker2_test_"+str(step)+".wav", fs, x_r)

              #'''
              outp1=inputslist[0][0]
              angle1=inputslist[0][1]
              outp1=np.reshape(outp1, (outp1.shape[1], outp1.shape[2]))
              outp_angle1=np.reshape(angle1, (angle1.shape[1], angle1.shape[2]))
              outp1_re=outp1*np.cos(outp_angle1) + 1j*outp1*np.sin(outp_angle1)
              outp1_re=np.column_stack((outp1_re,np.conj(outp1_re[:,1:-1].T[::-1].T)))
              x_r=istft(outp1_re, fs, (outp1.shape[1]+1)*256*3, hop)
              scipy.io.wavfile.write("speeker_raw_train_"+str(step)+".wav", fs, x_r)

              outp2=inputslist[0][2]
              angle2=inputslist[0][3]
              outp2=np.reshape(outp2, (outp2.shape[1], outp2.shape[2]))
              outp_angle2=np.reshape(angle2, (angle2.shape[1], angle2.shape[2]))
              outp2_re=outp2*np.cos(outp_angle2) + 1j*outp2*np.sin(outp_angle2)
              outp2_re=np.column_stack((outp2_re,np.conj(outp2_re[:,1:-1].T[::-1].T)))
              x_r=istft(outp2_re, fs, (outp2.shape[1]+1)*256*3, hop)
              scipy.io.wavfile.write("speeker_raw_test_"+str(step)+".wav", fs, x_r)
 #             break
            #'''

            #''' 
            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step
            #''' 

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        #'''
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        #'''
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

