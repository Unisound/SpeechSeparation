import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops

class SpeechSeparation(object):
    def _create_network_speechrnn(self,
			num_steps,
			speech_inputs_mix):
        with tf.variable_scope('SEEPCH_RNN_LAYER'):
          speech_outputs = []

          mlp1_weights = tf.get_variable(
            "mlp1", [self.dim, self.dim], dtype=tf.float32)
          mlp2_weights = tf.get_variable(
            "mlp2", [self.dim, self.dim], dtype=tf.float32)
          mlp3_weights = tf.get_variable(
            "mlp3", [self.dim, self.num_of_frequency_points*2], dtype=tf.float32)

          with tf.variable_scope("SEEPCH_RNN"):
	    input_list = tf.unstack(tf.transpose(speech_inputs_mix, perm=[1, 0, 2]), axis=0)
            fb_output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(self.f_cell, self.b_cell,\
		                        input_list, dtype=tf.float32, scope='bi_rnn')
            for speech_cell_output in fb_output: 
              out = math_ops.matmul(speech_cell_output, mlp1_weights)
              out = tf.nn.relu(out)
              out = math_ops.matmul(out, mlp2_weights)
              out = tf.nn.relu(out)
              out = math_ops.matmul(out, mlp3_weights)

              out = tf.nn.relu(out)
              speech_outputs.append(out)

          final_speech_outputs = tf.stack(speech_outputs) 
          final_speech_outputs = tf.transpose(final_speech_outputs, perm=[1, 0, 2])

          return final_speech_outputs


    def _create_network_SampleRnn(self,speech_inputs_1,speech_inputs_2,
	speech_inputs_mix):
        with tf.name_scope('SampleRnn_net'):
          #mask_num_steps = self.num_of_frequency_points-1
          mask_num_steps = 256
          mask_outputs = self._create_network_speechrnn(num_steps =  mask_num_steps, 
                         speech_inputs_mix = speech_inputs_mix)
          mask_1,mask_2=tf.split(mask_outputs,2, 2)
          output1 = speech_inputs_mix * mask_1
          output2 = speech_inputs_mix * mask_2

          return output1, output2


    def loss_SampleRnn(self,
						speech_inputs_1,
						speech_inputs_2,
						speech_inputs_mix,
						train_input_batch_rnn,
						l2_regularization_strength=None,
						name='sample'):
        with tf.name_scope(name):
            output1, output2 = self._create_network_SampleRnn( \
												speech_inputs_1,
												speech_inputs_2,
												speech_inputs_mix)

            with tf.name_scope('loss'):
                #mask_num_steps = self.num_of_frequency_points-1
                mask_num_steps = 256
                target_1    =tf.reshape(speech_inputs_1, [self.batch_size*mask_num_steps, -1])
                target_2    =tf.reshape(speech_inputs_2, [self.batch_size*mask_num_steps, -1])
                prediction_1=tf.reshape(output1, [self.batch_size*mask_num_steps,-1])
                prediction_2=tf.reshape(output2, [self.batch_size*mask_num_steps,-1])

                loss_1 = tf.losses.mean_squared_error(labels=target_1, predictions=prediction_1) 
                loss_2 = tf.losses.mean_squared_error(labels=target_2, predictions=prediction_2) 
                 
                reduced_loss_1 = tf.reduce_mean(loss_1)
                reduced_loss_2 = tf.reduce_mean(loss_2)
                reduced_loss =reduced_loss_1+reduced_loss_2

                if l2_regularization_strength is None:
                    tf.summary.scalar('reduced_loss', reduced_loss)
                    return reduced_loss , output1,output2
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_regularization_strength * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    return total_loss, output1,output2

    def __init__(self,
                 batch_size,
                 frame_size,
                 q_levels,
                 rnn_type,
                 dim,
                 n_rnn,
                 seq_len,
                 num_of_frequency_points,
                 emb_size):
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.q_levels = q_levels
        self.rnn_type = rnn_type
        self.dim = dim
        self.n_rnn = n_rnn
        self.seq_len=seq_len
        self.num_of_frequency_points=num_of_frequency_points
        self.emb_size=emb_size
        self.merged = tf.summary.merge_all()
 
        def single_cell():
          if 'LSTM' == self.rnn_type:
            return tf.contrib.rnn.BasicLSTMCell(self.dim/2)
          else:
            return tf.contrib.rnn.GRUCell(self.dim/2)    
        self.cell = single_cell()
        self.f_cell = single_cell()
        self.b_cell = single_cell()
        if self.n_rnn > 1:
          print("add rnn layer",self.n_rnn)
          self.cell = tf.contrib.rnn.MultiRNNCell(
                 [single_cell() for _ in range(self.n_rnn)])
          self.f_cell = tf.contrib.rnn.MultiRNNCell(
                 [single_cell() for _ in range(self.n_rnn)])
          self.b_cell = tf.contrib.rnn.MultiRNNCell(
                 [single_cell() for _ in range(self.n_rnn)])
