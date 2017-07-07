"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

from utils import get_arguments, save, load, validate_directories, average_gradients
from model import SpeechSeparation
from audio import AudioReader,mk_audio
from ops import optimizer_factory
from data_input import create_inputdict


import time
import logging
import numpy as np
import tensorflow as tf


EPSILON = 0.001
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
    # if the trained model is written into a location that's different from 
    # logdir.
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

##########Create model#########
    net =  SpeechSeparation(
    batch_size=args.batch_size,
    frame_size=args.frame_size,
    q_levels=args.q_levels,
    rnn_type=args.rnn_type,
    dim=args.dim,
    n_rnn=args.n_rnn,
    seq_len=args.seq_len,
    num_of_frequency_points=args.num_of_frequency_points,
    emb_size=args.emb_size)
########Multi GPU###########
    #'''
    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    # Create a variable to count the number of steps. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable('global_step',
        [], initializer = tf.constant_initializer(0), trainable=False)


    # Create optimizer (default is Adam)
    optim = optimizer_factory[args.optimizer](
                    learning_rate=args.learning_rate,
                    momentum=args.momentum)
    tower_grads = []
    
    
    losses = []
    speech_inputs_mix = []
    speech_inputs_1 = []
    speech_inputs_2 = []
    train_input_batch_rnn = []

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

    # Calculate the gradients for each model tower.
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(args.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('TOWER_%d' % (i)) as scope:
            # Create model.
            print("Creating model On Gpu:%d." % (i))
            
            loss, output1, output2 = net.loss_SampleRnn(
                speech_inputs_1[i],
                speech_inputs_2[i],
                speech_inputs_mix[i],
                train_input_batch_rnn[i],
                        l2_regularization_strength=args.l2_regularization_strength)
            
            # Reuse variables for the nect tower.
            tf.get_variable_scope().reuse_variables()

            # UNKNOWN
            losses.append(loss)
            trainable = tf.trainable_variables()
            for name in trainable:
              print(name)


            # Calculate the gradients for the batch of data on this tower.
            gradients = optim.compute_gradients(loss,trainable)
            print("==========================")
            for name in gradients:
              print(name)
            # Keep track of the gradients across all towers.
            tower_grads.append(gradients)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grad_vars = average_gradients(tower_grads)

    # UNKNOWN
    grads, vars = zip(*grad_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
    grad_vars = zip(grads_clipped, vars)

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = optim.apply_gradients(grad_vars, global_step=global_step)


    # Set up session
    tf_config = tf.ConfigProto(\
        # allow_soft_placement is set to True to build towers on GPU
        allow_soft_placement=True,\
        log_device_placement=False,\
                inter_op_parallelism_threads = 1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    sess.run(tf.global_variables_initializer())

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()



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


###################
    # Start the queue runners.
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step
    try:
        loss_sum = 0;
        for step in range(saved_global_step + 1, args.num_steps):
            loss_sum = 0
            start_time = time.time()

            inp_dict = create_inputdict(sess,audio_batch,args,speech_inputs_1,
                speech_inputs_2,speech_inputs_mix)
            ##########out results#########
            loss_value,_= sess.run([losses, apply_gradient_op], feed_dict=inp_dict)

            for g in xrange(args.num_gpus):
              loss_sum += loss_value[g]/args.num_gpus
            
            duration = time.time() - start_time

            if(step<100):
              log_str = ('step {%d} - loss = {%0.3f}, ({%0.3f} sec/step')%(step, loss_sum, duration)
              logging.warning(log_str)

            elif(0==step % 100):
              log_str = ('step {%d} - loss = {%0.3f}, ({%0.3f} sec/step')%(step, loss_sum/100, duration)
              logging.warning(log_str)

            if (0==step % 20):

              inp_dict={}
              inp_dict[speech_inputs_1[0]] = inputslist[0][2][:,:args.seq_len,:]
              inp_dict[speech_inputs_2[0]] = inputslist[0][2][:,s_len:s_len+args.seq_len,:]
              inp_dict[speech_inputs_mix[0]] = inputslist[0][2][:,-s_len:-s_len+args.seq_len,:]
              angle_test= inputslist[0][3][:,-s_len:-s_len+args.seq_len,:]

              outp1,outp2 = sess.run([output1,output2], feed_dict=inp_dict)

              x_r = mk_audio(outp1,angle_test,args.sample_rate,"spk1_test_"+str(step)+".wav")
              y_r = mk_audio(outp2,angle_test,args.sample_rate,"spk2_test_"+str(step)+".wav")
              merged = sess.run(tf.summary.merge(
                    [tf.summary.audio('speaker1_' + str(step), x_r[None, :], args.sample_rate),
                     tf.summary.audio('speaker2_' + str(step), y_r[None, :], args.sample_rate)]
                ))
              writer.add_summary(merged, step)

              outp2=inputslist[0][2]
              angle2=inputslist[0][3]              
              mk_audio(outp2,angle2,args.sample_rate,"raw_test_"+str(step)+".wav")

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

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

