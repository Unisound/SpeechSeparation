import logging
from audio import mk_audio

def create_inputdict(inputslist,args,speech_inputs_1,speech_inputs_2,speech_inputs_mix):
    inp_dict={}

    s_len=inputslist[0][0].shape[1]/3
    if(args.seq_len > s_len):
        logging.error("args.seq_len %d > s_len %d", args.seq_len, s_len)

    for g in xrange(args.num_gpus):
      inp_dict[speech_inputs_1[g]]  =inputslist[g][0][:,:args.seq_len,:]
      inp_dict[speech_inputs_2[g]]  =inputslist[g][0][:,s_len:s_len+args.seq_len,:]
      inp_dict[speech_inputs_mix[g]]=inputslist[g][0][:,-s_len:-s_len+args.seq_len,:]

    return inp_dict



def predict(sess,inputslist,args,step,output1,output2,speech_inputs_1,speech_inputs_2,speech_inputs_mix):
    inp_dict={}
    s_len=inputslist[0][0].shape[1]/3
    seq_len = args.seq_len

    inp_dict[speech_inputs_1[0]] = inputslist[0][2][:,:seq_len,:]
    inp_dict[speech_inputs_2[0]] = inputslist[0][2][:,s_len:s_len+seq_len,:]
    inp_dict[speech_inputs_mix[0]] = inputslist[0][2][:,-s_len:-s_len+seq_len,:]
    angle_test= inputslist[0][3][:,-s_len:-s_len+seq_len,:]

    outp1,outp2 = sess.run([output1,output2], feed_dict=inp_dict)

    x_r = mk_audio(outp1,angle_test,args.sample_rate,"spk1_test_"+str(step)+".wav")
    y_r = mk_audio(outp2,angle_test,args.sample_rate,"spk2_test_"+str(step)+".wav")

    outp2=inputslist[0][2]
    angle2=inputslist[0][3]              
    mk_audio(outp2,angle2,args.sample_rate,"raw_test_"+str(step)+".wav")

   
    return (x_r, y_r)
