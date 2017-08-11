import tensorflow as tf
import librosa
import os
from IPython.display import Audio, display
import numpy as np
import matplotlib.pyplot as plt

from progressbar import ETA, Bar, Percentage, ProgressBar
from sys import stderr
from vae import VAE
import pickle

# Reads wav file and produces spectrum
# Fourier phases are ignored
N_FFT = 2048
fs = 22050
def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    
    S = np.log1p(np.abs(S[:430,:430])) 
    return S, fs

def produce(result, filename):
    '''
    result: of shape height * width * channels (1 * 430 * 430)
    '''
    a = np.zeros(shape=(1025, 430))
    a[:N_CHANNELS, :] = np.exp(result[0].T) - 1

    # This code is supposed to do phase reconstruction
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(500):
        S = a * np.exp(1j*p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))

    librosa.output.write_wav(filename, x, fs)

# src and tgt should have 1-to-1 correspondance in filename
SRC_FILEROOT = "inputs/source-226/"
TGT_FILEROOT = "inputs/target-238/"
TEST_FILEROOT = "inputs/test-226/"

N_SAMPLES = 430
N_CHANNELS = 430

lookup = {}
test_list = []
for subdir, dirs, files in os.walk(SRC_FILEROOT):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".wav"):
            print (filepath)
            a, fs = read_audio_spectum(filepath)
            a = a[:N_CHANNELS, :N_SAMPLES]
            a_tf = np.ascontiguousarray(a.T[None,None,:,:])
            key = file[:-4][5:]
            lookup[key] = {}
            lookup[key]['src'] = a_tf

for subdir, dirs, files in os.walk(TGT_FILEROOT):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".wav"):
            print (filepath)
            a, fs = read_audio_spectum(filepath)
            a = a[:N_CHANNELS, :N_SAMPLES]
            a_tf = np.ascontiguousarray(a.T[None,None,:,:])
            key = file[:-4][5:]
            lookup[key]['tgt'] = a_tf

for subdir, dirs, files in os.walk(TEST_FILEROOT):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".wav"):
            print (filepath)
            a, fs = read_audio_spectum(filepath)
            a = a[:N_CHANNELS, :N_SAMPLES]
            a_tf = np.ascontiguousarray(a.T[None,None,:,:])
            test_list.append(a_tf)

with open("./lookup.p", "w") as outputfile:
    pickle.dump(lookup, outputfile)

with open("./test_list.p", "w") as outputfile:
    pickle.dump(test_list, outputfile)
                          
flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "gan", "gan or vae")

FLAGS = flags.FLAGS

model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

for epoch in range(FLAGS.max_epoch):
    training_loss = 0.0

    pbar = ProgressBar()
    for i in pbar(range(FLAGS.updates_per_epoch)):
        items = numpy.random.choice(lookup.items(), size=FLAGS.batch_size)
        inputs = [item['src'] for item in items]
        inputs = np.concatenate(inputs, axis=0)
        print inputs.shape
        targets = [item['tgt'] for item in items]
        targets = np.concatenate(targets, axis=0)
        print targets.shape
        loss_value = model.update_params(inputs, targets)
        training_loss += loss_value

    training_loss = training_loss / \
        (FLAGS.updates_per_epoch * FLAGS.batch_size)

    print("Loss %f" % training_loss)

    test_items = numpy.random.choice(test_list, size=FLAGS.batch_size)
    test_inputs = np.concatenate(test_items, axis=0)
    results = model.evaluate(test_inputs)
    print results.shape
    for i in range(results.shape[0]):
        produce(results[i], "outputs/test-226/" + str(epoch)+"_"+str(i)+".wav")

# # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
# std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
# kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS)*std
    
# g = tf.Graph()
# with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
#     # data shape is "[batch, in_height, in_width, in_channels]",
#     x = tf.placeholder('float32', [1,1,N_SAMPLES,N_CHANNELS], name="x")

#     kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
#     conv = tf.nn.conv2d(
#         x,
#         kernel_tf,
#         strides=[1, 1, 1, 1],
#         padding="VALID",
#         name="conv")
    
#     net = tf.nn.relu(conv)

#     content_features = net.eval(feed_dict={x: a_content_tf})
#     style_features = net.eval(feed_dict={x: a_style_tf})
    
#     features = np.reshape(style_features, (-1, N_FILTERS))
#     style_gram = np.matmul(features.T, features) / N_SAMPLES

# ALPHA= 1e-2
# learning_rate= 1e-2
# iterations = 10000

# result = None
# with tf.Graph().as_default():

#     # Build graph with variable input
#     # x = tf.Variable(np.zeros([1,1,N_SAMPLES,N_CHANNELS], dtype=np.float32), name="x")
#     x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, name="x")

#     kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
#     conv = tf.nn.conv2d(
#         x,
#         kernel_tf,
#         strides=[1, 1, 1, 1],
#         padding="VALID",
#         name="conv")
    
#     net = tf.nn.relu(conv)

#     content_loss = ALPHA * 2 * tf.nn.l2_loss(
#             net - content_features)

#     style_loss = 0

#     _, height, width, number = map(lambda i: i.value, net.get_shape())

#     size = height * width * number
#     feats = tf.reshape(net, (-1, number))
#     gram = tf.matmul(tf.transpose(feats), feats)  / N_SAMPLES
#     style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

#      # Overall loss
#     loss = content_loss + style_loss

#     opt = tf.contrib.opt.ScipyOptimizerInterface(
#           loss, method='L-BFGS-B', options={'maxiter': 10000})
        
#     # Optimization
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
       
#         print('Started optimization.')
#         opt.minimize(sess)
    
#         print 'Final loss:', loss.eval()
#         result = x.eval()


# print result.shape
# a = np.zeros_like(a_test)
# a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1

# # This code is supposed to do phase reconstruction
# p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
# for i in range(500):
#     S = a * np.exp(1j*p)
#     x = librosa.istft(S)
#     p = np.angle(librosa.stft(x, N_FFT))

# OUTPUT_FILENAME = 'outputs/out.wav'
# librosa.output.write_wav(OUTPUT_FILENAME, x, fs)