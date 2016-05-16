#! /usr/bin/env python

# generate sequences from a previously trained LSTM model for which parameters
# were stored in a .npz file (expected to be in ./training_results)
# the generated sequences are printed on the screen and stored in ./text_output

from charlstm import *
import sys

if len(sys.argv) < 3:
    if len(sys.argv) < 2:
        print 'The path to a .npz result file must be provided as argument'
        print('This file should be located in ./training_results after ' +
              'training of the model with train.py')
        sys.exit()
    else:
        input_filepath = sys.argv[1]
    output_filepath = 'text_output/' + input_filepath.split('/')[-1] + '.out'
else:
    output_filepath = sys.argv[2]

rnn = CharLSTM()
print '* initialization of the LSTM model based on ' + input_filepath
rnn.model_setup(mfile=input_filepath)

with open(output_filepath, 'a', 0) as f: # append results in output_filepath
    for i in xrange(5):
        # generate, for example, 5 sequences of 100 characters each
        s = rnn.generate_text('', method='normal', text_length=100)
        print 'Sequence ' + str(i+1) + ': "' + s + '"'
        f.write('# sequence ' + str(i+1) + ':\n' + s + '\n\n')

