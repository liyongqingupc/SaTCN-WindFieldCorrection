import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
from utils import *
import constants as c
from tensorflow.python.keras.layers.convolutional import Conv3D

def SelfAttention0(input_tensor):
    xx = (input_tensor[0, :, :, :, :])  # (1, h, w, 2)
    n, h, w, ch = xx.shape  #
    # conv2d input: [batch,in_height,in_width,in_channels]
    theta0 = tf.layers.conv2d(xx, filters=1, kernel_size=1, strides=(1, 1), padding='SAME', name='theta0')  # self.conv_theta(x)
    print('theta:', theta0.shape)  # (1, h, w, 1)
    theta0 = tf.reshape(theta0, (-1, h * w, theta0.shape[-1]))
    print('theta_reshape:', theta0.shape)  # (1, h*w, 1)

    phi0 = tf.layers.conv2d(xx, filters=1, kernel_size=1, strides=(1, 1), padding='SAME', name='phi0')  # self.conv_phi(x)
    phi0 = tf.reshape(phi0, (-1, phi0.shape[-1], h * w))
    print('phi_reshape:', phi0.shape)  # (1, 1, h*w)

    g0 = tf.layers.conv2d(xx, filters=2, kernel_size=1, strides=(1, 1), padding='SAME', name='g0')  # self.conv_g(x)
    g0 = tf.reshape(g0, (-1, g0.shape[-1], h * w))
    print('g_reshape:', g0.shape)  # (1, 2, h*w)

    attn = tf.matmul(theta0, phi0)
    print('attn:', attn.shape)  # (1, h*w, h*w)
    attn = tf.nn.softmax(attn)

    attn_g = tf.matmul(g0, attn, transpose_b=True)
    print('attn_g:', attn_g.shape)  # (1, 2, h*w)

    attn_g = tf.reshape(attn_g, (-1, 2, h, w))
    attn_g = tf.transpose(attn_g, [0, 2, 3, 1])
    print('attn_g_reshape:', attn_g.shape)  # (1, h, w, 2)

    attn_g_sig0 = tf.layers.conv2d(attn_g, filters=2, kernel_size=1, strides=(1, 1), padding='SAME', name='attn_g_sig0')
    print('attn_g_sig:', attn_g_sig0.shape)  # (1, h, w, 2)
    attn_g_out = xx + attn_g_sig0

    attn_g_out = attn_g_out[np.newaxis, :, :, :, :]
    print('SA_output:', attn_g_out.shape)  # SA_output: (1, 1, h, w, 2)

    return attn_g_out

def SelfAttention1(input_tensor):
    xx = (input_tensor[0, :, :, :, :])
    n, h, w, ch = xx.shape  #

    theta1 = tf.layers.conv2d(xx, filters=1, kernel_size=1, strides=(1, 1), padding='SAME', name='theta1')
    theta1 = tf.reshape(theta1, (-1, h * w, theta1.shape[-1]))

    phi1 = tf.layers.conv2d(xx, filters=1, kernel_size=1, strides=(1, 1), padding='SAME', name='phi1')
    phi1 = tf.reshape(phi1, (-1, phi1.shape[-1], h * w))
    print('phi_reshape:', phi1.shape)

    g1 = tf.layers.conv2d(xx, filters=2, kernel_size=1, strides=(1, 1), padding='SAME', name='g1')  # self.conv_g(x)
    g1 = tf.reshape(g1, (-1, g1.shape[-1], h * w))

    attn = tf.matmul(theta1, phi1)
    print('attn:', attn.shape)  # (1, h*w, h*w)
    attn = tf.nn.softmax(attn)

    attn_g = tf.matmul(g1, attn, transpose_b=True)
    print('attn_g:', attn_g.shape)  #

    attn_g = tf.reshape(attn_g, (-1, 2, h, w))
    attn_g = tf.transpose(attn_g, [0, 2, 3, 1])
    print('attn_g_reshape:', attn_g.shape)  # (1, h, w, 2)

    attn_g_sig1 = tf.layers.conv2d(attn_g, filters=2, kernel_size=1, strides=(1, 1), padding='SAME', name='attn_g_sig1')
    print('attn_g_sig:', attn_g_sig1.shape)  # (1, h, w, 2)
    attn_g_out = xx + attn_g_sig1

    attn_g_out = attn_g_out[np.newaxis, :, :, :, :]
    print('SA_output:', attn_g_out.shape)

    return attn_g_out


def SelfAttention2(input_tensor):
    xx = (input_tensor[0, :, :, :, :])
    n, h, w, ch = xx.shape  #

    theta2 = tf.layers.conv2d(xx, filters=1, kernel_size=1, strides=(1, 1), padding='SAME', name='theta2')
    theta2 = tf.reshape(theta2, (-1, h * w, theta2.shape[-1]))

    phi2 = tf.layers.conv2d(xx, filters=1, kernel_size=1, strides=(1, 1), padding='SAME', name='phi2')
    phi2 = tf.reshape(phi2, (-1, phi2.shape[-1], h * w))

    g2 = tf.layers.conv2d(xx, filters=2, kernel_size=1, strides=(1, 1), padding='SAME', name='g2')
    g2 = tf.reshape(g2, (-1, g2.shape[-1], h * w))

    attn = tf.matmul(theta2, phi2)
    print('attn:', attn.shape)  # (1, h*w, h*w)
    attn = tf.nn.softmax(attn)

    attn_g = tf.matmul(g2, attn, transpose_b=True)
    attn_g = tf.reshape(attn_g, (-1, 2, h, w))
    attn_g = tf.transpose(attn_g, [0, 2, 3, 1])

    attn_g_sig2 = tf.layers.conv2d(attn_g, filters=2, kernel_size=1, strides=(1, 1), padding='SAME', name='attn_g_sig2')
    print('attn_g_sig:', attn_g_sig2.shape)  # (1, h, w, 2)
    attn_g_out = xx + attn_g_sig2

    attn_g_out = attn_g_out[np.newaxis, :, :, :, :]
    print('SA_output:', attn_g_out.shape)

    return attn_g_out


def SelfAttention3(input_tensor):
    xx = (input_tensor[0, :, :, :, :])
    n, h, w, ch = xx.shape  #

    theta3 = tf.layers.conv2d(xx, filters=1, kernel_size=1, strides=(1, 1), padding='SAME', name='theta3')  # self.conv_theta(x)
    theta3 = tf.reshape(theta3, (-1, h * w, theta3.shape[-1]))

    phi3 = tf.layers.conv2d(xx, filters=1, kernel_size=1, strides=(1, 1), padding='SAME', name='phi3')  # self.conv_phi(x)
    phi3 = tf.reshape(phi3, (-1, phi3.shape[-1], h * w))

    g3 = tf.layers.conv2d(xx, filters=2, kernel_size=1, strides=(1, 1), padding='SAME', name='g3')  # self.conv_g(x)
    g3 = tf.reshape(g3, (-1, g3.shape[-1], h * w))

    attn = tf.matmul(theta3, phi3)
    attn = tf.nn.softmax(attn)

    attn_g = tf.matmul(g3, attn, transpose_b=True)
    attn_g = tf.reshape(attn_g, (-1, 2, h, w))
    attn_g = tf.transpose(attn_g, [0, 2, 3, 1])

    attn_g_sig3 = tf.layers.conv2d(attn_g, filters=2, kernel_size=1, strides=(1, 1), padding='SAME', name='attn_g_sig3')
    print('attn_g_sig:', attn_g_sig3.shape)  # (1, h, w, 2)
    attn_g_out = xx + attn_g_sig3

    attn_g_out = attn_g_out[np.newaxis, :, :, :, :]
    print('SA_output:', attn_g_out.shape)

    return attn_g_out


class WindFieldCorrection():

    def __init__(self,sess,train_batch_size,test_batch_size,epochs,checkpoint_file,lambdl1,save_freq,histlen,futulen,learn_rate):
        self.bs1 = batch_norm(name = "genb1")
        self.bs2 = batch_norm(name = "genb2")
        self.bs3 = batch_norm(name = "genb3")
        self.bs4 = batch_norm(name = "genb4")
        self.bs5 = batch_norm(name = "genb5")

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lambdl1 = lambdl1   # lyq add: used for l1 norm term
        self.checkpoint_file = checkpoint_file
        self.sess = sess
        self.save_freq = save_freq   # lyq add
        self.histlen = histlen   # lyq add
        self.futulen = futulen  # lyq add
        self.learn_rate = learn_rate   # lyq add
        #self.g_cost = 0 # lyq add 220614

    def build_model(self):
        m = 2  # the num of input channel
        self.train_original = tf.placeholder(tf.float32, [None, self.histlen + 1 + self.futulen, c.data_height, c.data_width, m])
        self.train_revised = tf.placeholder(tf.float32, [None, 1, c.data_height, c.data_width, 1])

        self.train_revised_fake = self.generator(self.train_original)

        self.test_original = tf.placeholder(tf.float32, [None, self.histlen + 1 + self.futulen, c.data_height, c.data_width, m])
        self.test_revised = tf.placeholder(tf.float32, [None, 1, c.data_height, c.data_width, 1])
        self.test_revised_fake = self.revise_data(self.test_original)

        self.g_cost = self.lambdl1 * tf.reduce_mean(tf.abs(self.train_revised - self.train_revised_fake))


    def train(self):

        gen_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope = "generator")

        ##########  AdamOptimizer for  #############
        self.g_opt = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learn_rate, beta1 = 0.9).minimize(self.g_cost, var_list = gen_var)  #lyq 0.5->0.9 #220615

        saver = tf.compat.v1.train.Saver()  # used to save model

        if self.checkpoint_file == "None":
            self.ckpt_file = None
        if self.checkpoint_file:  # checkpoint_file
            saver_ = tf.train.import_meta_graph('../save/models/' + self.checkpoint_file + '.meta')  # meta_graph
            saver_.restore(self.sess, tf.train.latest_checkpoint(c.save_models_dir))
            print ("Restored model")
        else:
            tf.compat.v1.global_variables_initializer().run()

        train_data_original = glob.glob(os.path.join(c.train_original_dir, '*'))
        train_data_original.sort(key=lambda x:int(x.split('201802_201812_')[1].split('.npy')[0]))
        train_data_observe = glob.glob(os.path.join(c.train_observe_dir, '*'))
        train_data_observe.sort(key=lambda x:int(x.split('0125_')[1].split('.')[0]))
        train_data_revised = glob.glob(os.path.join(c.train_revised_dir, '*'))
        train_data_revised.sort(key=lambda x:int(x.split('201802_201812_')[1].split('.npy')[0]))

        test_data_original = glob.glob(os.path.join(c.test_original_dir, '*'))
        test_data_original.sort(key=lambda x:int(x.split('201801_')[1].split('.npy')[0]))
        test_data_observe = glob.glob(os.path.join(c.test_observe_dir, '*'))
        test_data_observe.sort(key=lambda x:int(x.split('0125_')[1].split('.npy')[0]))
        test_data_revised = glob.glob(os.path.join(c.test_revised_dir, '*'))
        test_data_revised.sort(key=lambda x:int(x.split('201801_')[1].split('.npy')[0]))

        for epoch in range(self.epochs):
            start_time = time.time()
            #################### cut_num ##################
            cut_num = 100  # num of deleted samples
            #################### cut_num ##################
            f = open(c.save_name + ".txt", "a")  # open a txt file to save results
            print('batch_size:', self.train_batch_size, 'learn_rate:', self.learn_rate, file=f)
            print("......Epoch_", epoch, "......", file=f)
            print("......Epoch_", epoch, "......")

            for counter in range(0, int((len(train_data_original) - self.histlen) / self.train_batch_size) - cut_num, 1):
                train_num = int((len(train_data_original) - self.histlen) / self.train_batch_size) - cut_num

                if np.mod(counter, 0.5 * train_num - 1) == 0:
                    print("....Iteration....:", counter, '/', train_num)

                ###################
                batch_original_path = train_data_original[counter * self.train_batch_size : \
                                     self.histlen + self.futulen + (counter + 1) * self.train_batch_size]
                input_original_for = read_data(batch_original_path, self.train_batch_size, self.histlen, self.futulen)

                batch_observe_path = train_data_observe[counter * self.train_batch_size + 1: \
                                     self.histlen + self.futulen + (counter + 1) * self.train_batch_size + 1]
                input_original_obs = read_observe_data(batch_observe_path, self.train_batch_size, self.histlen, self.futulen)
                input_original_for[np.where(input_original_obs == 0)] = 0

                input_original = np.concatenate((input_original_for, input_original_obs), axis = 4) # concatenate

                ###################
                batch_revised_path = train_data_revised[self.histlen + counter * self.train_batch_size + 1: \
                                     self.histlen + (counter + 1) * self.train_batch_size + 1]
                truth_revised = read_data(batch_revised_path, self.train_batch_size, 0, 0)
                truth_revised[np.where(input_original_obs[:, 0:1, :, :, :] == 0)] = 0

                ################### Print error ##################
                #print('input_original-truth_revised:', np.mean(abs(input_original[:, :, :, 4:5] - truth_revised)))

                _, g_cost = self.sess.run([self.g_opt, self.g_cost], \
                            feed_dict = {self.train_original: input_original, self.train_revised: truth_revised})

                if np.mod(counter, int(0.3 * train_num)) == 0:  # print g_cost_diff
                    print("g_cost: ", g_cost, file=f)
                    print("g_cost: ", g_cost)

            ################# Print time of each epoch ################
            print(('time', time.time() - start_time))


            '''test and save the model every save_freq epoches'''
            #####################################################
            if np.mod(epoch + 1, self.save_freq) == 0:
                print("......Testing.....")
                for tcounter in range(0, int((len(test_data_original) - self.histlen - self.futulen) / self.test_batch_size) - 2, 1):

                    test_num = int((len(test_data_original) - self.histlen) / self.test_batch_size) -1

                    ####################
                    tbatch_original_path = test_data_original[tcounter * self.test_batch_size: \
                                           self.histlen + self.futulen + (tcounter + 1) * self.test_batch_size]
                    tinput_original_for = read_data(tbatch_original_path, self.test_batch_size, self.histlen, self.futulen)

                    tbatch_observe_path = test_data_observe[tcounter * self.test_batch_size + 1: \
                                          self.histlen + self.futulen + (tcounter + 1) * self.test_batch_size + 1]
                    tinput_original_obs = read_observe_data(tbatch_observe_path, self.test_batch_size, self.histlen, self.futulen)

                    ####### land = 0 #######
                    tinput_original_for[np.where(tinput_original_obs == 0)] = 0
                    tinput_original = np.concatenate((tinput_original_for, tinput_original_obs), axis=4)  # concatenate
                    ####################
                    tbatch_revised_path = test_data_revised[self.histlen + tcounter * self.test_batch_size + 1: \
                                          self.histlen + (tcounter + 1) * self.test_batch_size + 1]
                    ttruth_revised = read_data(tbatch_revised_path, self.test_batch_size, 0, 0)
                    ttruth_revised[np.where(tinput_original_obs[:, 0:1, :, :, :] == 0)] = 0  # lyq 220610add

                    ################### Print error ##################
                    #print('tinput_original-ttruth_revised:', np.mean(abs(tinput_original[:, :, :, self.histlen : self.histlen+1] - ttruth_revised)))

                    ########## trevised_fake ##########
                    trevised_fake = self.sess.run([self.test_revised_fake],
                                    feed_dict = {self.test_original: tinput_original, self.test_revised: ttruth_revised})

                    trevised_fake = np.array(trevised_fake)
                    #print('trevised_fake:', trevised_fake.shape)
                    trevised_fake = trevised_fake[0,:,:,:,:,:]
                    trevised_fake[np.where(tinput_original_obs[:, 0:1, :, :, :] == 0)] = 0
                    test_diff = np.mean(np.abs(ttruth_revised - trevised_fake))

                    #print('ttruth_revised:',ttruth_revised.shape,'tinput_original:', tinput_original[:, self.histlen : self.histlen + 1, :, :, 0:1].shape)
                    truth_diff = np.mean(np.abs(ttruth_revised - tinput_original[:, self.histlen : self.histlen + 1, :, :, 0:1]))

                    if np.mod(tcounter, int(0.05 * test_num)) == 0:  # print error every some test samples int(0.1 * test_num)
                        print("Truth error: ", truth_diff, "Test error: ", test_diff, "Truth-Test: ", truth_diff - test_diff, file=f)
                        print("Truth error: ", truth_diff, "Test error: ", test_diff, "Truth-Test: ", truth_diff - test_diff)

                    test_save_dir = c.get_dir(os.path.join(c.test_save_dir, 'train' + str(train_num) + \
                                    '_histlen'+ str(self.histlen) + '_epoch'+ str(epoch)))
                    '''save generated test data'''
                    np.save(os.path.join(test_save_dir, str(epoch) + '_' + str(tcounter) + '.npy'), trevised_fake)

                '''save the trained model every save_freq epoches'''
                saver.save(self.sess, os.path.join(c.save_models_dir, 'WindPred' + str(epoch) + '.ckpt'))
                print ('Saved models {}'.format(epoch))

            f.close()  # Close

    def generator(self, train_original, reuse = False):
        with tf.variable_scope("generator") as scope:
             print('train_original:', train_original.shape)
             x = (train_original[0, :, :, :, :])
             x = x[np.newaxis, :, :, :, :]
             x0 = SelfAttention0(x[:, 0:1, :, :, :])
             print('x0:', x0.shape)
             x1 = SelfAttention1(x[:, 1:2, :, :, :])
             x2 = SelfAttention2(x[:, 2:3, :, :, :])
             x3 = SelfAttention3(x[:, 3:4, :, :, :])
             x = tf.concat([x0, x1, x2, x3], 1)
             print('x:', x.shape)

             m = 8
             #### tf.pad(tensor, paddings, mode='CONSTANT', constant_values=0, name=None) ####
             train_original1 = tf.pad(x, [[0, 0],[1, 0],[1, 1],[1, 1],[0, 0]])

             convb1 = Conv3D(filters=32*m, kernel_size=(2,3,3), strides=(1,1,1), padding='valid', dilation_rate=(1,1,1))(train_original1) #same->valid 220801
             convb1 = lrelu(self.bs1(convb1))
             convb1 = tf.pad(convb1, [[0, 0], [1, 0], [1, 1], [1, 1], [0, 0]])
             print('convb1:', convb1.shape)
             convb2 = Conv3D(filters=16*m, kernel_size=(2,3,3), strides=(1,1,1), padding='valid', dilation_rate=(1,1,1))(convb1)
             convb2 = lrelu(self.bs2(convb2))
             convb2 = tf.pad(convb2, [[0, 0], [1, 0], [1, 1], [1, 1], [0, 0]])
             print('convb2:', convb2.shape)
             convb3 = Conv3D(filters=8*m, kernel_size=(2,3,3), strides=(1,1,1), padding='valid', dilation_rate=(1,1,1))(convb2)
             convb3 = lrelu(self.bs3(convb3))
             convb3 = tf.pad(convb3, [[0, 0], [1, 0], [1, 1], [1, 1], [0, 0]])
             print('convb3:', convb3.shape)
             convb4 = Conv3D(filters=1,  kernel_size=(2,3,3), strides=(1,1,1), padding='valid', dilation_rate=(1,1,1))(convb3)
             convb4 = lrelu(self.bs4(convb4))
             print('convb4:', convb4.shape)

             # integrate 4 channel
             convb4 = tf.transpose(convb4, [0, 4, 2, 3, 1])  # (1, 1, 121, 121, 4)
             print('convb4_transpose:', convb4.shape)
             convb5 = tf.layers.conv2d(convb4[0, :, :, :, :], 1, kernel_size=[1, 1], strides=[1, 1],padding='SAME')
             convb5 = lrelu(self.bs5(convb5))
             print('convb5:', convb5.shape)
             convb5 = convb5[np.newaxis, :, :, :, :]
             revised_data = convb5 + train_original[:, self.histlen:self.histlen+1, :, :, 0:1]  #

             print('Train_revised_data:', revised_data.shape)
             return revised_data

    def revise_data(self, test_original):
        with tf.variable_scope("generator") as scope:
             scope.reuse_variables()
             x = (test_original[0, :, :, :, :])
             x = x[np.newaxis, :, :, :, :]
             x0 = SelfAttention0(x[:, 0:1, :, :, :])
             x1 = SelfAttention1(x[:, 1:2, :, :, :])
             x2 = SelfAttention2(x[:, 2:3, :, :, :])
             x3 = SelfAttention3(x[:, 3:4, :, :, :])
             x = tf.concat([x0, x1, x2, x3], 1)

             m = 8
             test_original1 = tf.pad(x, [[0, 0], [1, 0], [1, 1], [1, 1], [0, 0]])
             convb1 = Conv3D(filters=32*m, kernel_size=(2,3,3), strides=(1,1,1), padding='valid',dilation_rate=(1,1,1))(test_original1)
             convb1 = lrelu(self.bs1(convb1))
             convb1 = tf.pad(convb1, [[0, 0], [1, 0], [1, 1], [1, 1], [0, 0]])
             convb2 = Conv3D(filters=16*m, kernel_size=(2,3,3), strides=(1,1,1), padding='valid',dilation_rate=(1,1,1))(convb1)
             convb2 = lrelu(self.bs2(convb2))
             convb2 = tf.pad(convb2, [[0, 0], [1, 0], [1, 1], [1, 1], [0, 0]])
             convb3 = Conv3D(filters=8*m, kernel_size=(2,3,3), strides=(1,1,1), padding='valid',dilation_rate=(1,1,1))(convb2)
             convb3 = lrelu(self.bs3(convb3))
             convb3 = tf.pad(convb3, [[0, 0], [1, 0], [1, 1], [1, 1], [0, 0]])
             convb4 = Conv3D(filters=1,  kernel_size=(2,3,3), strides=(1,1,1), padding='valid',dilation_rate=(1,1,1))(convb3)
             convb4 = lrelu(self.bs4(convb4))

             # integrate 4 channel
             convb4 = tf.transpose(convb4, [0, 4, 2, 3, 1])  # (1, 1, 121, 121, 4)
             print('convb4_transpose:', convb4.shape)
             convb5 = tf.layers.conv2d(convb4[0, :, :, :, :], 1, kernel_size=[1, 1], strides=[1, 1], padding='SAME')
             convb5 = lrelu(self.bs5(convb5))
             print('convb5:', convb5.shape)
             convb5 = convb5[np.newaxis, :, :, :, :]
             revised_data = convb5 + test_original[:, self.histlen:self.histlen+1, :, :, 0:1]  #

             print('Test_revised_data:', revised_data.shape)
             return revised_data