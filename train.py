#! /usr/bin/env python

# train the LSTM model based on an input textfile (input_filepath)
# results (.out and .npz files) are stored in ./training_results
# -- the .out file contains useful information (training and validation costs,
#    as well as update/parameter ratios) step by step
# -- the .npz files store the parameters of the model at a given point of
#    learning

from charlstm import *
import numpy as np
import sys

if len(sys.argv) < 2:
    # by default, learn model based on Jane Austen's Pride and Prejudice
    input_filepath = 'text_input/prideandprejudice.txt'
else:
    input_filepath = sys.argv[1]
if len(sys.argv) < 3:
    # name of the run, to keep track of output files in case of several runs
    run_name = 'run0'
else:
    run_name = sys.argv[2]

input_filename = input_filepath.split('/')[-1]
output_filepath = 'training_results/' + run_name + '_' + input_filename + '.out'

rnn = CharLSTM()
print '* preparing the input file ' + input_filepath + '...'
rnn.prepare_input(input_filepath)
print '* initialization of the LSTM model...'
rnn.model_setup(num_units1=256, num_units2=256)

print('Summary of the hyperparameters (2-layer LSTM):' +
      '\n* num_units1=' + str(rnn.p['b1'].get_value().shape[1]) +
      '\n* num_units2=' + str(rnn.p['b2'].get_value().shape[1]) +
      '\n* lrate=' + str(rnn.lp['lrate']) +
      '\n* drate=' + str(rnn.lp['drate']) +
      '\n* eps=' + str(rnn.lp['eps']) +
      '\n* bptt_maxdepth=' + str(rnn.lp['bptt_maxdepth']) +
      '\n* l1=' + str(rnn.lp['l1']) +
      '\n* l2=' + str(rnn.lp['l2']) +
      '\n')

print 'Start learning for run "' + run_name + '"'
print 'Results are stored in ./training_results\n'

f = open(output_filepath, 'w', 0)

nepoch = 75 # total number of epochs (i.e. number of passes through the input
            # file) to be considered when learning

# dynamic learning rate: decrease the learning rate by a factor of lratedecay
# at every epoch starting at epoch number lratedecay_nepoch
lratedecay = 0.97
lratedecay_nepoch = 10

for n in xrange(nepoch):
    if n+1 >= lratedecay_nepoch:
        rnn.dyn_lrate.set_value(np.float32(rnn.dyn_lrate.get_value() *
                                           lratedecay))
    current_lrate = rnn.dyn_lrate.get_value()
    for i in xrange(rnn.n_batches['train']): # loop over training data batches
        cur_epoch = n + float(i) / rnn.n_batches['train']

        # first, check the cost for a random batch from the validation data
        if rnn.n_batches['valid'] > 1:
            rand_j = np.random.randint(0, rnn.n_batches['valid'])
            v_cost, v_prediction = rnn.theano_check(rnn.X['valid'][rand_j],
                                                    rnn.Y['valid'][rand_j])
        else: # no validation data; set v_cost to a default value
            v_cost = -1

        # then, do a step of learning using the training data
        t_cost, t_prediction, r_U1, r_W1, r_b1, r_U2, r_W2, r_b2, r_V, r_c = \
            rnn.theano_train_rmsprop(rnn.X['train'][i], rnn.Y['train'][i])

        print('Epoch #' + str(n+1) + ', training batch ' + str(i+1) + '/' +
              str(rnn.n_batches['train']) + ', lrate=' + str(current_lrate))
        print '(t_cost, v_cost) = (' + str(t_cost) + ', ' + str(v_cost) + ')'
        print(('ratios: {:.2e} {:.2e} {:.2e} '
               '{:.2e} {:.2e} {:.2e} {:.2e} {:.2e}\n').format(
                   float(r_U1), float(r_W1), float(r_b1), float(r_U2),
                   float(r_W2), float(r_b2), float(r_V), float(r_c)))

        f.write(str(cur_epoch) + ' ' +
                str(t_cost) + ' ' +
                str(v_cost) + ' ' +
                str(r_U1) + ' ' +
                str(r_W1) + ' ' +
                str(r_b1) + ' ' +
                str(r_U2) + ' ' +
                str(r_W2) + ' ' +
                str(r_b2) + ' ' +
                str(r_V) + ' ' +
                str(r_c) + '\n')

    if n > 0 and n % 10 == 0:
        # for instance, save model parameters every ten epochs
        npzpath = ('training_results/' + run_name + '_' + input_filename +
                   '_epoch' + str(n) + '.npz')
        print 'Saving model parameters into file ' + npzpath + '\n'
        rnn.save_params(npzpath)

f.close()

