import numpy as np

import fixed_ensemble


class FixedEnsembleLearning(fixed_ensemble.FixedEnsemble):
    def __init__(self, learning_rate=5e-3, **kwargs):
        super(FixedEnsembleLearning, self).__init__(**kwargs)
        self.learning_rate_shift=(int(round(np.log2(1.0/learning_rate))) -
                                  self.decoder_offset)

    def step(self, input, error):
        # the learning version also gets an error signal
        self.error = np.array(error.clip(-self.input_max,
                                         self.input_max), dtype='int64')
        return super(FixedEnsembleLearning, self).step(input)

    def neuron(self, current):
        activity = super(FixedEnsembleLearning, self).neuron(current)

        # use the error signal and the activity to adjust the decoders
        index = np.where(activity>0)[0]
        if len(index) > 0:
            delta = self.error >> self.learning_rate_shift
            self.decoder[:,index] -= delta[:, None]

            dec_max = 1<<(self.decoder_bits-1)-1
            dec_min = -(1<<(self.decoder_bits-1))
            self.decoder = np.clip(self.decoder, dec_min, dec_max)

        return activity
