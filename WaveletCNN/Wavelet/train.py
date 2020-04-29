import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
import soundfile as sf
from conf import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras import backend as K
K.clear_session()
from model import getModel
import librosa as lb
from data_io import batchGenerator, batchGenerator_val
from keras.callbacks import LearningRateScheduler
from keras.metrics import categorical_accuracy as accuracy

  
def lr_scheduler(epoch, lr):
    if epoch == 1:
        decay_rate = 1
        return lr * decay_rate
    elif epoch == 2:
        decay_rate = 10
        return lr*decay_rate
    elif epoch == 3:
        decay_rate = 10
        return lr*decay_rate
    elif epoch == 4:
        decay_rate = 0.1
        return lr*decay_rate
    elif epoch == 5:
        decay_rate = 1
        return lr*decay_rate
    else:
        decay_rate = 1
        return lr*decay_rate
    return lr




if __name__ == "__main__":

    saver = tf.train.Saver()

    print('N_filt ' + str(cnn_N_filt))
    print('N_filt len ' + str(cnn_len_filt))
    print('FS ' + str(fs))
    print('WLEN ' + str(wlen))
    print('learning_rate: ',lr)

    input_shape = (wlen, 1)
    out_dim = class_lay[0]

    logits = getModel(input_shape, out_dim)
    
    prediction = tf.nn.log_softmax(logits)
    
    
    labels = tf.placeholder(tf.float32, shape=(None,out_dim))

    with tf.name_scope('Accuracy'):
        acc= tf.reduce_mean(accuracy(labels, logits))

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        
        
    with tf.name_scope('RMSProp'):
    # Gradient Descent
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", acc)
    
    merged_summary_op = tf.summary.merge_all()
    
    # Initialize all variables
    init_op = tf.global_variables_initializer()
    
    batch_size = 32
    logs_path = os.path.join(output_folder, 'logs', 'Sincnet')
    checkpoints_path = os.path.join(output_folder, 'checkpoints','model.ckpt')
    # Run training loop
    with sess.as_default():
        if pt_file != 'none':
            load_path = saver.restore(sess, checkpoints_path)
            print("Model restored from file: %s" % checkpoints_path)
        sess.run(init_op)
        
        train_summary_writer = tf.summary.FileWriter(logs_path+'/Train',
                                                graph=tf.get_default_graph())
        val_summary_writer = tf.summary.FileWriter(logs_path+'/Val')
        for epoch in range(N_epochs):
            gen = batchGenerator(batch_size, train_data_folder, flac_lst_train, snt_train, wlen, out_dim)
            for i in range(N_batches):
                X_batch, y_batch = next(gen)
                if epoch == 0:
                    lr = 0.00001
                elif epoch == 1:
                    lr = 0.0001
                elif epoch == 2:
                    lr = 0.001
                elif epoch == 3:
                    lr = 0.0001
                else:
                    lr = 0.0001
                 
                feed_dict = {learning_rate: lr, inputs: X_batch, labels: y_batch, tf.keras.backend.learning_phase(): 1}
                sess.run(train_step,feed_dict)
                loss_train, acc_train, summary = (sess.run([loss, acc, merged_summary_op],feed_dict))
                train_summary_writer.add_summary(summary, epoch * total_batch_train + i)
                print("Epoch: "+str(epoch)+"step: "+str(i)+"Training loss: ",loss_train," ","Training accuracy"," ",acc_train)
                
                save_path = saver.save(sess, checkpoints_path)
                print("Model saved in file: %s" % save_path)
                
            if epoch == N_eval_epoch | epoch == 9:
            
                test_flag = 1
                loss_sum = 0
                err_sum = 0
                err_sum_snt = 0

            
                for i in range(snt_dev):
                    
                    file_id = flac_lst_dev.iloc[i, 1]
                    [signal, fs] = sf.read(str(dev_data_folder + file_id + '.wav'))
                    lab_batch = flac_lst_dev.iloc[i, -1]

                    # split signals into chunks
                    beg_samp = 0
                    end_samp = wlen

                    N_fr = int((signal.shape[0] - wlen) / (wshift))

                    sig_arr = tf.zeros([Batch_dev, wlen],dtype = tf.float32)
                    lab = Variable((torch.zeros(N_fr + 1) + lab_batch).cuda().contiguous().long())
                    pout = Variable(torch.zeros(N_fr + 1, class_lay[-1]).float().cuda().contiguous())
                    count_fr = 0
                    count_fr_tot = 0
                    while end_samp < signal.shape[0]:
                        sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                        beg_samp = beg_samp + wshift
                        end_samp = beg_samp + wlen
                        count_fr = count_fr + 1
                        count_fr_tot = count_fr_tot + 1
                        if count_fr == Batch_dev:
                            inp = Variable(sig_arr)
                            if which_model == 0:
                                pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))
                            else:
                                pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(WCNN_net(inp)))
                            count_fr = 0
                            sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()

                    if count_fr > 0:
                        inp = Variable(sig_arr[0:count_fr])
                        if which_model == 0:
                            pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(CNN_net(inp)))
                        else:
                            pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(WCNN_net(inp)))

                    pred = torch.max(pout, dim=1)[1]
                    loss = cost(pout, lab.long())
                    err = torch.mean((pred != lab.long()).float())
                    score = torch.sum((pout[:, 0] - pout[:, 1]), dim=0) / len(pout)
                    [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
                    err_sum_snt = err_sum_snt + (best_class != lab[0]).float()

                    loss_sum = loss_sum + loss.detach()
                    err_sum = err_sum + err.detach()

                    with open(output_folder + "/dev_scores_" + str(epoch) + ".txt", "a+") as score_file:
                        if flac_lst_dev.iloc[i, 2] == 0:
                            score_file.write("{} - bonafide {} \n".format(flac_lst_dev.iloc[i, 1], score))
                        else:
                            score_file.write(
                                "{} A0{} spoof {} \n".format(flac_lst_dev.iloc[i, 1], flac_lst_dev.iloc[i, 2],
                                                             score))

                err_tot_dev_snt = err_sum_snt / snt_dev
                loss_tot_dev = loss_sum / snt_dev
                err_tot_dev = err_sum / snt_dev

            print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
                epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

            with open(output_folder + "/res.res", "a+") as res_file:
                res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
                    epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        
        

#     optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-8)
#     model.compile(loss=categorical_crossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])

#     checkpoints_path = os.path.join(output_folder, 'checkpoints')

#     tb = TensorBoard(log_dir=os.path.join(output_folder, 'logs', 'Sincnet'))
#     lrate = LearningRateScheduler(lr_scheduler, verbose=1)
#     checkpointer = ModelCheckpoint(
#         filepath=os.path.join(checkpoints_path, 'Sincnet.hdf5'),
#         verbose=1,
#         save_best_only=False,
#         save_weights_only=True)

#     if not os.path.exists(checkpoints_path):
#         os.makedirs(checkpoints_path)

#     callbacks = [tb, checkpointer,lrate]

#     if pt_file != 'none':
#         model.load_weights(pt_file)

#     train_generator = batchGenerator(batch_size, train_data_folder, flac_lst_train, snt_train, wlen, out_dim)
#     validation_generator = batchGenerator_val(Batch_dev, dev_data_folder, flac_lst_dev, snt_dev, wlen, out_dim)
#     model.fit_generator(train_generator, steps_per_epoch=N_batches, epochs=N_epochs, verbose=1,
#                         validation_data=validation_generator, validation_steps=N_dev_batches, callbacks=callbacks)



