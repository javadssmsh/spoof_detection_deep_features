import os
import soundfile as sf
from conf import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras import backend as K
from model import getModel
import librosa as lb




def batchGenerator(batch_size, data_folder, flac_lst, N_snt, wlen, fact_amp, out_dim):
    """
        Batch generator for the train data

        Parameters:
        batch_size (int): size of batch
        data_folder : path where the flac audio files are present
        flac_lst (dataframe): protocol file with speaker_id, file_id, system_id, labels
        N_snt (int) : total number of samples
        wlen (int) : Length of the frame or chunk
        fact_amp (float) : amplitude for windowing
        out_dim (int) : output_dimension for the labels used to make the labels categorical

        Returns:
        sig_batch (np.array) : array containing random samples of given batch_size
        lab_batch (np.array) : array containing labels for the corresponding samples in sig_batch

        """
    while True:
        sig_batch, lab_batch = create_batches_rnd(batch_size, data_folder, flac_lst, N_snt, wlen, fact_amp,
                                                  out_dim)
        yield sig_batch, lab_batch


def create_batches_rnd(batch_size, data_folder, flac_lst, N_snt, wlen, fact_amp, out_dim):
    """
            Create random chunks form randomly chosen samples from the whole dataset.
            Each batch is structured such that for each speaker there are equal number of spoofed and human samples.

            Parameters:
            batch_size (int): size of batch
            data_folder : path where the flac audio files are present
            flac_lst (dataframe): protocol file with speaker_id, file_id, system_id, labels
            N_snt (int) : total number of samples
            wlen (int) : Length of the frame or chunk
            fact_amp (float) : amplitude for windowing
            out_dim (int) : output_dimension for the labels used to make the labels categorical

            Returns:
            sig_batch (np.array) : array containing random samples of given batch_size
            lab_batch (np.array) : array containing labels for the corresponding samples in sig_batch

            """
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch = []
    lab_batch = []
    snt_id_arr = np.random.randint(N_snt, size=batch_size)
    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

    for i in range(int(batch_size / 2)):
        # select a random sentence from the list
        # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        # signal=signal.astype(float)/32768
        file_id = flac_lst.iloc[snt_id_arr[i],1]
        [signal, fs] = sf.read(data_folder+file_id+'.flac')
        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg + wlen
        sig_batch.append(signal[snt_beg:snt_end] * rand_amp_arr[i])
        y = flac_lst.iloc[snt_id_arr[i], -1]
        lab_batch.append(y)
        speaker_id = flac_lst.iloc[snt_id_arr[i], 0]
        if flac_lst.iloc[snt_id_arr[i], -1] == 0:
            selected_labels = flac_lst.loc[(flac_lst.iloc[:, -1] == 1) & (flac_lst.iloc[:, 0] == speaker_id)]
        elif flac_lst.iloc[snt_id_arr[i], -1] == 1:
            selected_labels = flac_lst.loc[(flac_lst.iloc[:, -1] == 0) & (flac_lst.iloc[:, 0] == speaker_id)]
        label = selected_labels.sample()
        index = flac_lst.loc[(flac_lst.iloc[:, 1] == label.values[0][1])].index
        file_id = flac_lst.iloc[index,1].values[0]
        [signal, fs] = sf.read(str(data_folder+file_id+'.flac'))
        signal_len = signal.shape[0]
        signal_beg = np.random.randint(signal_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        signal_end = signal_beg + wlen
        sig_batch.append(signal[signal_beg:signal_end] * rand_amp_arr[i])
        y = flac_lst.iloc[index, -1].values[0]
        lab_batch.append(y)
    sig_batch = np.array(sig_batch)
    lab_batch = np.array(lab_batch)

    idx = np.random.permutation(len(sig_batch))
    x, y = sig_batch[idx], lab_batch[idx]
    a, b = np.shape(x)
    sig_batch = x.reshape((a, b, 1))
    #     print(sig_batch.shape)
    return sig_batch, to_categorical(np.array(y), num_classes=out_dim)


def batchGenerator_val(batch_size, data_folder, wav_lst, N_snt, wlen, fact_amp, out_dim):
    """
            Batch generator for the validation data

            Parameters:
            batch_size (int): size of batch
            data_folder : path where the flac audio files are present
            flac_lst (dataframe): protocol file with speaker_id, file_id, system_id, labels
            N_snt (int) : total number of samples
            wlen (int) : Length of the frame or chunk
            fact_amp (float) : amplitude for windowing
            out_dim (int) : output_dimension for the labels used to make the labels categorical

            Returns:
            sig_batch (np.array) : array containing random samples of given batch_size
            lab_batch (np.array) : array containing labels for the corresponding samples in sig_batch

            """
    while True:
        sig_batch, lab_batch = create_batches_rnd_val(batch_size, data_folder, wav_lst, N_snt, wlen, fact_amp,
                                                  out_dim)
        yield sig_batch, lab_batch


def create_batches_rnd_val(batch_size,data_folder,flac_lst, N_snt, wlen,fact_amp, out_dim):
    """
                Create random chunks form randomly chosen samples from the whole dataset.

                Parameters:
                batch_size (int): size of batch
                data_folder : path where the flac audio files are present
                flac_lst (dataframe): protocol file with speaker_id, file_id, system_id, labels
                N_snt (int) : total number of samples
                wlen (int) : Length of the frame or chunk
                fact_amp (float) : amplitude for windowing
                out_dim (int) : output_dimension for the labels used to make the labels categorical

                Returns:
                sig_batch (np.array) : array containing random samples of given batch_size
                lab_batch (np.array) : array containing labels for the corresponding samples in sig_batch

                """
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch=np.zeros([batch_size,wlen])
    lab_batch=[]
    snt_id_arr=np.random.randint(N_snt, size=batch_size)
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)
    for i in range(batch_size):
        # select a random sentence from the list
        #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        #signal=signal.astype(float)/32768
        file_id = flac_lst.iloc[snt_id_arr[i],1]
        [signal, fs] = sf.read(str(data_folder+file_id+'.flac'))
        # accesing to a random chunk
        snt_len=signal.shape[0]
        snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
        snt_end=snt_beg+wlen
        sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
        y= flac_lst.iloc[snt_id_arr[i],-1]
        yt = to_categorical(y, num_classes=out_dim)
        lab_batch.append(yt)
    a, b = np.shape(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
    return sig_batch, np.array(lab_batch)

if __name__ == "__main__":

    K.clear_session()


    print('N_filt ' + str(cnn_N_filt))
    print('N_filt len ' + str(cnn_len_filt))
    print('FS ' + str(fs))
    print('WLEN ' + str(wlen))

    input_shape = (wlen, 1)
    out_dim = class_lay[0]


    model = getModel(input_shape, out_dim)
    optimizer = RMSprop(lr=[0], rho=0.9, epsilon=1e-8)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    checkpoints_path = os.path.join(output_folder, 'checkpoints')

    tb = TensorBoard(log_dir=os.path.join(output_folder, 'logs', 'SincNet'))
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_path, 'SincNet.hdf5'),
        verbose=1,
        save_best_only=False)

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)


    callbacks = [tb, checkpointer]

    if pt_file != 'none':
        model.load_weights(pt_file)

    train_generator = batchGenerator(batch_size, train_data_folder, flac_lst_train, snt_train, wlen, 0.2, out_dim)
    validation_generator = batchGenerator_val(Batch_dev, dev_data_folder, flac_lst_dev, snt_dev, wlen, 0.2, out_dim)
    model.fit_generator(train_generator, steps_per_epoch=N_batches, epochs=N_epochs, verbose=1,
                        validation_data=validation_generator, validation_steps=N_dev_batches, callbacks=callbacks)
