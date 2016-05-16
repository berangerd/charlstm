import sys
import numpy as np
import theano
import theano.tensor as T

class CharLSTM:
    """
        theano implementation of a character-level recurrent neural network
        2-layer LSTM, mini-batch learning using rmsprop
        supports L1 and/or L2 regularization
    """

    def __init__(self):
        # LSTM parameters
        self.p = {}
        # learning parameters
        self.lp = {}

        self.batch_size = 1
        self.uchar = [] # list of unique characters found in the input file
                        # determines the size of the input 1-hot vector

        # sequences (divided into training and validation sets) put into batches
        self.X = {}
        self.Y = {}
        self.n_batches = {} # number of minibatches for the training and
                            # validation sets

    def prepare_input(self, filepath, ratio_valid=0.30, batch_size=50,
                      seq_length=50, verbose=True):
        """transformation of the input into numerical arrays
           to be used for mini-batch learning"""

        self.batch_size = batch_size

        with open(filepath, 'r') as f:
            data = f.read()

        self.uchar = list(set(data)) # list of all unique characters
        char_dim = len(self.uchar)

        seqs = [data[i:i+seq_length] for i in range(0, len(data), seq_length)]
        if len(seqs[-1]) != seq_length:
            # mini-batch learning requires sequences of the same length
            # so remove the last sequence if its length is not seq_length
            seqs = seqs[:-1]

        # separate training from validation data according to ratio_valid
        boundary_index = int(round((1. - ratio_valid) * len(seqs)))
        sequences = {'train': seqs[:boundary_index],
                     'valid': seqs[boundary_index:]}

        self.n_batches['train'] = int(np.ceil(
                                      len(sequences['train'])/batch_size))
        self.n_batches['valid'] = int(np.ceil(
                                      len(sequences['valid'])/batch_size))

        if verbose:
            print('* ' + str(char_dim - 1) + ' unique characters' +
                  '\n* ' + str(len(seqs)) + ' sequences provided in ' +
                  str(self.n_batches['train']) + ' training batches and ' +
                  str(self.n_batches['valid']) + ' validation batches\n' +
                  '  (batch_size=' + str(batch_size) + ')\n')

        # pre-processing of the dataset for mini-batch learning
        self.X = {'train': [], 'valid': []}
        self.Y = {'train': [], 'valid': []}

        for sample in self.n_batches: # 'train' and 'valid'
            for n in xrange(self.n_batches[sample]): # loop over batches
                if batch_size > 1:
                    arrayX_length = (seq_length - 1, batch_size)
                    arrayY_length = (seq_length - 1, batch_size, char_dim)
                else: # online learning is a special case
                    arrayX_length = seq_length - 1
                    arrayY_length = (seq_length - 1, char_dim)
                cur_X = np.zeros(arrayX_length, dtype=np.int32)
                cur_Y = np.zeros(arrayY_length, dtype=np.int8)

                for m in xrange(batch_size): # loop over sequences
                    s = sequences[sample][n * batch_size + m]
                    for i in xrange(len(s) - 1): # loop over characters
                        if batch_size > 1:
                            cur_X[i,m] = self.uchar.index(s[i])
                            cur_Y[i,m,self.uchar.index(s[i+1])] = 1
                        else:
                            cur_X[i] = self.uchar.index(s[i])
                            cur_Y[i,self.uchar.index(s[i+1])] = 1
                self.X[sample].append(cur_X)
                self.Y[sample].append(cur_Y)

    def model_setup(self, mfile=None, num_units1=128, num_units2=128,
                          lrate=2e-3, drate=0.95, eps=1e-8, bptt_maxdepth=50,
                          l1=0, l2=0, char_dim=None):
        """initialization of the 2-layer LSTM model for learning or for
           the generation of sequences"""

        # the default parameters are identical to Andrej Karpathy's
        # (see https://github.com/karpathy/char-rnn)

        # 2-layer LSTM parameters
        self.p = {'U1': None, 'W1': None, 'b1': None,
                  'U2': None, 'W2': None, 'b2': None,
                  'V': None, 'c': None}
        # learning parameters
        self.lp = {'lrate': lrate, # learning rate
                   'drate': drate, # decay rate for rmsprop
                   'eps': eps, # epsilon parameter for rmsprop
                   'bptt_maxdepth': bptt_maxdepth, # backpropagation cutoff
                   'l1': l1, # L1 regularization parameter
                   'l2': l2 # L2 regularization parameter
                  }

        if mfile is not None: # loading parameters from an npz file
            np_init = self.load_params(mfile)
            num_units1 = np_init['b1'].shape[1]
            num_units2 = np_init['b2'].shape[1]
        else:
            if char_dim is None:
                if self.uchar:
                    char_dim = len(self.uchar)
                else:
                    raise Exception('prepare_input() should be run before ' +
                                    'model_setup() unless mfile is provided')

            # initialize small random weights
            r_char_dim = np.sqrt(1./(char_dim))
            r_units1 = np.sqrt(1./(num_units1))
            r_units2 = np.sqrt(1./(num_units2))

            def uniform(rng, shape):
                return np.random.uniform(-rng, rng,
                                         shape).astype(theano.config.floatX)

            def randn(rng, shape):
                return np.random.uniform(-rng, rng,
                                         shape).astype(theano.config.floatX)

            def bias_hack(num_units):
                b = np.zeros((4, num_units))
                b[0] = 1. # forget gate hack
                          # helps the network remember information
                return b.astype(theano.config.floatX)

            def zeros(shape):
                return np.zeros(shape).astype(theano.config.floatX)

            def ones(shape):
                return np.ones(shape).astype(theano.config.floatX)

            # parameters for the gates
            # [0]: forget
            # [1]: input
            # [2]: output
            # [3]: cell state update
            np_init = {}

            # first layer
            np_init['U1'] = uniform(r_char_dim, (4, num_units1, char_dim))
            np_init['W1'] = uniform(r_units1, (4, num_units1, num_units1))
            np_init['b1'] = bias_hack(num_units1)

            # second layer
            np_init['U2'] = uniform(r_units1, (4, num_units2, num_units1))
            np_init['W2'] = uniform(r_units2, (4, num_units2, num_units2))
            np_init['b2'] = bias_hack(num_units2)

            # parameters for the last layer (cell output -> network output)
            np_init['V'] = uniform(r_units2, (char_dim, num_units2))
            np_init['c'] = zeros(char_dim)

            # dynamical learning rate (in case the user wants to modify it
            # during the learning process)
            if theano.config.floatX == 'float32':
                dyn_lrate_init = np.float32(self.lp['lrate'])
            else:
                dyn_lrate_init = np.float64(self.lp['lrate'])
            self.dyn_lrate = theano.shared(dyn_lrate_init, name='dyn_lrate')

            # parameters for rmsprop (running average of gradients)
            msq_g = {}
            for param in self.p:
                msq_g[param] = theano.shared(zeros(np_init[param].shape),
                                             name='msq_g'+param)

        for param in self.p:
            self.p[param] = theano.shared(np_init[param], name=param)

        if self.batch_size > 1:
            x = T.imatrix('x')
            y = T.btensor3('y')
        else:
            x = T.ivector('x')
            y = T.bmatrix('y')

        def forward_prop(x, ht1m1, Ct1m1, ht2m1, Ct2m1,
                         U1, W1, b1, U2, W2, b2, V, c):
            # defines each time step of the RNN model

            if self.batch_size > 1: # transform into column vectors
                col_b1 = b1.dimshuffle((0,1,'x'))
                col_b2 = b2.dimshuffle((0,1,'x'))
                col_c = c.dimshuffle((0,'x'))
            else:
                col_b1 = b1
                col_b2 = b2
                col_c = c

            # layer 1
            gates1 = []
            for i in xrange(3): # forget, input and output gates
                gates1.append(T.nnet.sigmoid(U1[i][:,x] +
                                             W1[i].dot(ht1m1) +
                                             col_b1[i]))
            tentative_Ct1 = T.tanh(U1[3][:,x] + W1[3].dot(ht1m1) + col_b1[3])

            Ct1 = Ct1m1 * gates1[0] + tentative_Ct1 * gates1[1]
            ht1 = gates1[2] * T.tanh(Ct1)

            # layer 2
            gates2 = []
            for i in xrange(3): # forget, input and output gates
                gates2.append(T.nnet.sigmoid(U2[i].dot(ht1) +
                                             W2[i].dot(ht2m1) +
                                             col_b2[i]))
            tentative_Ct2 = T.tanh(U2[3].dot(ht1) + W2[3].dot(ht2m1) +
                                   col_b2[3])

            Ct2 = Ct2m1 * gates2[0] + tentative_Ct2 * gates2[1]
            ht2 = gates2[2] * T.tanh(Ct2)

            # final layer
            o = T.nnet.softmax((V.dot(ht2) + col_c).T)

            return [o, ht1, Ct1, ht2, Ct2]

        if self.batch_size > 1:
            ht1_Ct1_size = (num_units1, self.batch_size)
            ht2_Ct2_size = (num_units2, self.batch_size)
        else:
            ht1_Ct1_size = num_units1
            ht2_Ct2_size = num_units2

        [o, ht1, Ct1, ht2, Ct2], updates = theano.scan(
            fn=forward_prop,
            sequences=x,
            outputs_info=[None,
                            T.zeros(ht1_Ct1_size),
                            T.zeros(ht1_Ct1_size),
                            T.zeros(ht2_Ct2_size),
                            T.zeros(ht2_Ct2_size)
                           ],
            non_sequences=[self.p['U1'], self.p['W1'], self.p['b1'],
                           self.p['U2'], self.p['W2'], self.p['b2'],
                           self.p['V'], self.p['c']],
            truncate_gradient=self.lp['bptt_maxdepth'],
            strict=True)

        # o is a (seq_len, batch_size, char_dim) tensor---even if batch_size=1
        prediction = T.argmax(o, axis=2)

        self.theano_predict = theano.function(
            inputs=[x],
            outputs=[o, prediction],
        )

        if mfile is not None: # not here for learning; we can stop here
            return

        # compute the cross-entropy loss
        xent = (-y*T.log(o)).sum(axis=2) # (string_len, batch_size) matrix
        cost = T.mean(xent)

        # regularization using L1 and/or L2 norms
        reg_cost = cost

        # cast into theano.config.floatX is a trick to avoid float64 below
        tot_shape = (xent.shape[0] * xent.shape[1]).astype(theano.config.floatX)

        for param in self.p:
            if l1 > 0: # L1 regularization
                reg_cost += l1 * T.sum(abs(self.p[param])) / tot_shape
            if l2 > 0: # L2 regularization
                reg_cost += l2 * T.sum(self.p[param] ** 2) / tot_shape

        g = {}
        for param in self.p:
            g[param] = T.grad(reg_cost, self.p[param])

        # for rmsprop
        new_msq_g = {}
        updates = {}
        rmsprop_updates = []
        sgd_updates = []
        ratios = {}
        for param in self.p:
            new_msq_g[param] = (self.lp['drate'] * msq_g[param] +
                               (1. - self.lp['drate']) * g[param]**2)

            updates[param] = (self.dyn_lrate * g[param] /
                             (T.sqrt(new_msq_g[param]) + self.lp['eps']))

            # update to parameter scale ratio
            ratios[param] = (T.flatten(updates[param]).norm(2) /
                             T.flatten(self.p[param]).norm(2))

            sgd_updates.append((self.p[param],
                                self.p[param] - self.dyn_lrate * g[param]))

            rmsprop_updates.append((self.p[param],
                                    self.p[param] - updates[param]))
            rmsprop_updates.append((msq_g[param], new_msq_g[param]))

            # todo: add possibility to clip gradients to some value

        f_out = [cost, prediction]

        # compute cost and prediction but do not update the weights
        self.theano_check = theano.function(
            inputs=[x, y],
            outputs=f_out,
        )

        f_out.extend([ratios['U1'], ratios['W1'], ratios['b1'],
                      ratios['U2'], ratios['W2'], ratios['b2'],
                      ratios['V'], ratios['c']])

        # mini-batch training with rmsprop
        self.theano_train_rmsprop = theano.function(
            inputs=[x, y],
            outputs=f_out,
            updates=rmsprop_updates
        )

        # mini-batch training with stochastic gradient descent
        self.theano_train_sgd = theano.function(
            inputs=[x, y],
            outputs=f_out,
            updates=sgd_updates
        )

    def generate_text(self, seed='', text_length=100, method='normal'):
        """generate text given previously learned model parameters"""

        if seed == '':
            # uniform selsection from existing characters
            seed = np.random.choice(self.uchar)

        cur_array = np.zeros(len(seed)).astype('int32')
        for i in xrange(len(seed)):
            cur_array[i] = self.uchar.index(seed[i])

        text = seed
        for i in xrange(text_length-len(seed)):
            # cannot do a step-by-step foward pass of the RNN model because
            # of the scan() structure in theano. so do forward pass
            # text_length - len(seed) times with increasing sequence length
            o, pred = self.theano_predict(cur_array)
            # consider only the output and prediction for the last character
            o = o[-1][0]
            pred = pred[-1][0]

            o /= (o.sum() + 1e-6) # without this eps=1e-6, np.random.multinomial
                                  # may return the following error:
                                  # "ValueError: sum(pvals[:-1]) > 1.0"

            if method == 'normal':
                guess = np.argmax(np.random.multinomial(1, o))
            else: # method=argmax
                guess = pred
            cur_array = np.append(cur_array, np.int32(guess))
            text += self.uchar[guess]
        return text


    def save_params(self, filename):
        """save the learned parameters of the model into an npz file"""
        np.savez(filename,
                 U1=self.p['U1'].get_value(),
                 W1=self.p['W1'].get_value(),
                 b1=self.p['b1'].get_value(),
                 U2=self.p['U2'].get_value(),
                 W2=self.p['W2'].get_value(),
                 b2=self.p['b2'].get_value(),
                 V=self.p['V'].get_value(),
                 c=self.p['c'].get_value(),
                 uchar=self.uchar
                )

    def load_params(self, filename):
        """load the learned parameters of the model stored in an npz file"""
        res = np.load(filename)
        np_init = {}
        for param in res:
            if param in self.p:
                np_init[param] = res[param]
            elif param == 'uchar':
                self.uchar = list(res['uchar'])
        return np_init

