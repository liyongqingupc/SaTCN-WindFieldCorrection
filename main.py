
from model import WindFieldCorrection
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_integer("train_batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("test_batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("checkpoint_file", None, "The checkpoint file name")
flags.DEFINE_integer("lambdl1", 1, "l1 norm lambda hyperparameter")
flags.DEFINE_integer("save_freq", 5, "Freq to save")    # save_freq
flags.DEFINE_integer("histlen", 2, "History seq length")
flags.DEFINE_integer("futulen", 1, "Future seq length")
flags.DEFINE_float("learn_rate", 5e-4, "learn_rate")  # lyq 5e-4 -> 5e-3,integer->float
FLAGS = flags.FLAGS

def main(_):
    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config = run_config) as sess:
        wrwgan = WindFieldCorrection(sess,train_batch_size = FLAGS.train_batch_size,\
                            test_batch_size = FLAGS.test_batch_size,\
                            epochs=FLAGS.epoch,checkpoint_file = FLAGS.checkpoint_file,\
                            lambdl1 = FLAGS.lambdl1, save_freq = FLAGS.save_freq,\
                            histlen = FLAGS.histlen, futulen = FLAGS.futulen, learn_rate = FLAGS.learn_rate)
        wrwgan.build_model()
        wrwgan.train()

if __name__ == '__main__':
    tf.compat.v1.app.run()