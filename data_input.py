import logging

def create_inputdict(sess,audio_batch,args,speech_inputs_1,
                speech_inputs_2,speech_inputs_mix):
    inp_dict={}

    inputslist = [sess.run(audio_batch) for i in xrange(args.num_gpus)]

    s_len=inputslist[0][0].shape[1]/3
    if(args.seq_len > s_len):
        logging.error("args.seq_len %d > s_len %d", args.seq_len, s_len)

    for g in xrange(args.num_gpus):
      inp_dict[speech_inputs_1[g]]  =inputslist[g][0][:,:args.seq_len,:]
      inp_dict[speech_inputs_2[g]]  =inputslist[g][0][:,s_len:s_len+args.seq_len,:]
      inp_dict[speech_inputs_mix[g]]=inputslist[g][0][:,-s_len:-s_len+args.seq_len,:]

    return inp_dict