"""
            File that is responsible for initializing the variables in conf.py file from the config file.

            Functions:
            ReadList(list_file) : To convert the protocol file into a dataframe to feed it into other functions like batchGenerator
            Read_conf(conf_file): To read the config file and initialize the conf.py file
            Str_to_bool():


"""

import configparser as ConfigParser
from optparse import OptionParser
import numpy as np
import pandas as pd
import soundfile as sf
import soundfile as sf
import os
import webrtcvad
import contextlib
import sys
import collections
import wave
from keras.utils import to_categorical
import os
from sklearn import preprocessing
mm_scaler = preprocessing.StandardScaler()
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# # The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";  



def ReadList(list_file):

    filehandle = open(list_file, 'r')

    protocol = []

    while True:

        # read a single line

        line = (filehandle.readline())

        protocol.append(line)

        if not line:

            break

    # close the pointer to that file
    filehandle.close()

    protocol = [s[:-1] for s in protocol]

    protocol = pd.DataFrame([s.split(' ') for s in protocol])

    protocol.columns = ['speaker_id', 'file_id', 'blah', 'system_id', 'label']

    protocol = protocol[['speaker_id', 'file_id', 'system_id', 'label']]

    protocol.dropna(inplace=True)

    protocol.drop_duplicates(subset="file_id", keep='first', inplace=True)

    for count, i in enumerate(protocol.iloc[:, -2]):
        if i == '-':
            protocol.iloc[count, -2] = 0
            protocol.iloc[count, -1] = 0
        elif i == 'A01':
            protocol.iloc[count, -2] = 1
            protocol.iloc[count, -1] = 1
        elif i == 'A02':
            protocol.iloc[count, -2] = 2
            protocol.iloc[count, -1] = 1
        elif i == 'A03':
            protocol.iloc[count, -2] = 3
            protocol.iloc[count, -1] = 1
        elif i == 'A04':
            protocol.iloc[count, -2] = 4
            protocol.iloc[count, -1] = 1
        elif i == 'A05':
            protocol.iloc[count, -2] = 5
            protocol.iloc[count, -1] = 1
        elif i == 'A06':
            protocol.iloc[count, -2] = 6
            protocol.iloc[count, -1] = 1

    return protocol


def read_conf(cfg_file=None):
    parser = OptionParser()
    parser.add_option("--cfg")  # Mandatory
    (options, args) = parser.parse_args()
    if cfg_file is None:
        cfg_file = options.cfg
    if options.cfg is None and cfg_file is None:
        cfg_file = r'/home/rohita/rohit/spoof/spoof_deep_features/spoof_detection_deep_features/WaveletCNN/Wavelet/cfg/Wavelet_ASV.cfg'
    Config = ConfigParser.ConfigParser()
    Config.read(cfg_file)

    # [data]
    options.train_lst = Config.get('data', 'train_lst')
    options.dev_lst = Config.get('data', 'dev_lst')
    # options.lab_dict = Config.get('data', 'lab_dict')
    options.train_data_folder = Config.get('data', 'train_data_folder')
    options.dev_data_folder = Config.get('data', 'dev_data_folder')
    options.output_folder = Config.get('data', 'output_folder')
    options.pt_file = Config.get('data', 'pt_file')

    # [windowing]
    options.fs = Config.get('windowing', 'fs')
    options.cw_len = Config.get('windowing', 'cw_len')
    options.cw_shift = Config.get('windowing', 'cw_shift')

    # [cnn]
    options.cnn_N_filt = Config.get('cnn', 'cnn_N_filt')
    options.cnn_len_filt = Config.get('cnn', 'cnn_len_filt')
    options.cnn_max_pool_len = Config.get('cnn', 'cnn_max_pool_len')
    options.cnn_use_laynorm_inp = Config.get('cnn', 'cnn_use_laynorm_inp')
    options.cnn_use_batchnorm_inp = Config.get('cnn', 'cnn_use_batchnorm_inp')
    options.cnn_use_laynorm = Config.get('cnn', 'cnn_use_laynorm')
    options.cnn_use_batchnorm = Config.get('cnn', 'cnn_use_batchnorm')
    options.cnn_act = Config.get('cnn', 'cnn_act')
    options.cnn_drop = Config.get('cnn', 'cnn_drop')

    # [dnn]
    options.fc_lay = Config.get('dnn', 'fc_lay')
    options.fc_drop = Config.get('dnn', 'fc_drop')
    options.fc_use_laynorm_inp = Config.get('dnn', 'fc_use_laynorm_inp')
    options.fc_use_batchnorm_inp = Config.get('dnn', 'fc_use_batchnorm_inp')
    options.fc_use_batchnorm = Config.get('dnn', 'fc_use_batchnorm')
    options.fc_use_laynorm = Config.get('dnn', 'fc_use_laynorm')
    options.fc_act = Config.get('dnn', 'fc_act')

    # [class]
    options.class_lay = Config.get('class', 'class_lay')
    options.class_drop = Config.get('class', 'class_drop')
    options.class_use_laynorm_inp = Config.get('class', 'class_use_laynorm_inp')
    options.class_use_batchnorm_inp = Config.get('class', 'class_use_batchnorm_inp')
    options.class_use_batchnorm = Config.get('class', 'class_use_batchnorm')
    options.class_use_laynorm = Config.get('class', 'class_use_laynorm')
    options.class_act = Config.get('class', 'class_act')

    # [optimization]
    options.lr = Config.get('optimization', 'lr')
    options.batch_size = Config.get('optimization', 'batch_size')
    options.N_epochs = Config.get('optimization', 'N_epochs')
    options.N_batches = Config.get('optimization', 'N_batches')
    options.N_dev_batches = Config.get('optimization', 'N_dev_batches')
    options.N_eval_epoch = Config.get('optimization', 'N_eval_epoch')
    options.seed = Config.get('optimization', 'seed')

    return options


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError
        
        
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
        file_id = flac_lst.iloc[snt_id_arr[i], 1]
        [signal,fs] = sf.read(data_folder + file_id + '.wav')
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
        file_id = flac_lst.iloc[index, 1].values[0]
        [signal,fs] = sf.read(str(data_folder + file_id + '.wav'))
        signal_len = signal.shape[0]
        signal_beg = np.random.randint(signal_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        signal_end = signal_beg + wlen
        sig_batch.append(signal[signal_beg:signal_end] * rand_amp_arr[i])
        y = flac_lst.iloc[index, -1].values[0]
        lab_batch.append(y)
    sig_batch = np.array(sig_batch)
    lab_batch = np.array(lab_batch)
    sig_batch = mm_scaler.fit_transform(sig_batch)
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


def create_batches_rnd_val(batch_size, data_folder, flac_lst, N_snt, wlen, fact_amp, out_dim):
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
    sig_batch = np.zeros([batch_size, wlen])
    lab_batch = []
    snt_id_arr = np.random.randint(N_snt, size=batch_size)
    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)
    for i in range(batch_size):
        file_id = flac_lst.iloc[snt_id_arr[i], 1]
        [signal,fs] = sf.read(str(data_folder + file_id + '.wav'))
        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg + wlen
        sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
        y = flac_lst.iloc[snt_id_arr[i], -1]
        yt = to_categorical(y, num_classes=out_dim)
        lab_batch.append(yt)
    a, b = np.shape(sig_batch)
    sig_batch = mm_scaler.fit_transform(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
    return sig_batch, np.array(lab_batch)


def batchGenerator_scores(batch_size, data_folder, wav_lst, N_snt, wlen, fact_amp, out_dim):
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
        sig_batch, lab_batch = create_batches_scores(batch_size, data_folder, wav_lst, N_snt, wlen, fact_amp,
                                                      out_dim)
        yield sig_batch, lab_batch

def create_batches_scores(batch_size, data_folder, flac_lst, N_snt, wlen, fact_amp, out_dim):
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
    sig_batch = np.zeros([batch_size, wlen])
    lab_batch = []
    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)
    for i in range(batch_size):
        file_id = flac_lst.iloc[i, 1]
        [signal,fs] = sf.read(str(data_folder + file_id + '.wav'))
        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg + wlen
        sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
        y = flac_lst.iloc[i, -1]
        yt = to_categorical(y, num_classes=out_dim)
        lab_batch.append(yt)
    a, b = np.shape(sig_batch)
    sig_batch = mm_scaler.fit_transform(sig_batch)
    sig_batch = sig_batch.reshape((a, b, 1))
    return sig_batch, np.array(lab_batch)

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []

    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def vad(data_folder, flac_lst):
    for count,i in enumerate(flac_lst.iloc[:,1]):
        if count%100==0:
            print("count: ",count)
        data, fs = sf.read(str(data_folder + i + '.flac'))
        sf.write(str(data_folder + i + '.wav'), data, fs)
        audio, fs = read_wave(str(data_folder + i + '.wav'))
        os.remove(str(data_folder + i + '.flac'))
        vad = webrtcvad.Vad(3)
        frames = frame_generator(10, audio, fs)
        frames = list(frames)
        segments = vad_collector(fs, 30, 300, vad, frames)
        audio = b""
        for j, segment in enumerate(segments):
            if audio == False:
                audio = segment
            else:
                audio = b"".join([audio,segment])
# Save as FLAC file at correct sampling rate
        write_wave(str(data_folder + i + '.wav'), audio, fs) 
       