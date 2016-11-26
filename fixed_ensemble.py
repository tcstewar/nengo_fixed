import numpy as np


class FixedEnsemble(object):
    def __init__(self,
                 n_inputs,          # dimensionality of input
                 n_outputs,         # dimensionality of output
                 n_neurons,         # number of neurons
                 input_bits=8,      # number of bits for input (and output)
                 state_bits=8,      # number of bits for internal state
                 extra_bits=4,      # number of extra bits used in internal
                                    # computation of the encoder
                 decoder_offset=4,  # number of extra bits used in the decoder
                 decoder_bits=16,   # total number of bits for decoder
                                    #  (including offset)
                 seed=None,         # random number seed
                 has_neuron_state=True, # whether to have state (or an LFSR)
                 smoothing=10       # compute a low-pass filter of this many
                                    #  time steps on the output
                 ):
        self.input_bits = input_bits
        self.state_bits = state_bits
        self.extra_bits = extra_bits
        self.decoder_offset = decoder_offset
        self.decoder_bits = decoder_bits

        # we're using an int64 in this implementation, so make sure we're
        #  not outside of the range provided by this
        assert input_bits + state_bits + extra_bits < 64

        self.rng = np.random.RandomState(seed=seed)
        self.compute_encoders(n_inputs, n_neurons)
        self.decoder = np.zeros((n_outputs, n_neurons), dtype='int64')
        self.has_neuron_state = has_neuron_state
        self.input_max = (1<<self.input_bits) - 1
        if has_neuron_state:
            self.state = np.zeros(n_neurons, dtype='int64')
        self.smoothing = smoothing
        if smoothing > 0:
            smoothing_decay = np.exp(-1.0/smoothing)
            self.smoothing_shift = -int(np.round(np.log2(1-smoothing_decay)))
            self.smoothing_state = np.zeros(n_outputs, dtype='int64')

    def compute_encoders(self, n_inputs, n_neurons):
        # generate the static synapses
        # NOTE: this algorithm could be changed, and just needs to produce a
        # similar distribution of connection weights.  Changing this
        # distribution slightly changes the class of functions the neural
        # network will be good at learning
        max_rates = self.rng.uniform(0.5, 1, n_neurons)
        intercepts = self.rng.uniform(-1, 1, n_neurons)

        gain = max_rates / (1 - intercepts)
        bias = -intercepts * gain

        enc = self.rng.randn(n_neurons, n_inputs)
        enc /= np.linalg.norm(enc, axis=1)[:,None]

        encoder = enc * gain[:, None]
        self.bias = (bias*(1<<(self.state_bits+self.extra_bits))).astype('int64')

        # store sign and shift rather than the encoder
        self.sign = np.where(encoder>0, 1, -1)
        self.shift1 = np.log2(encoder*(1<<self.extra_bits)*self.sign).astype(int)
        self.shift1 += self.state_bits - self.input_bits

    def set_decoders(self, decoders):
        # if we have externally created decoders, apply them here
        self.decoder[:] = decoders.T * self.input_max * (1<<self.decoder_offset)
        dec_max = 1<<(self.decoder_bits-1)-1
        dec_min = -(1<<(self.decoder_bits-1))
        self.decoder = np.clip(self.decoder, dec_min, dec_max)

    def step(self, input):
        # the main neuron update loop

        # first, make sure the state is integers in a fixed range
        input = np.array(input.clip(-self.input_max,
                                    self.input_max), dtype='int64')

        # feed input over the encoders
        current = self.compute_neuron_input(input)
        # do the neural nonlinearity
        activity = self.neuron(current)

        self.spikes = activity  # used for offline learning purposes
                                # (the real system would not need to store this)

        # apply the learned synapses
        value = self.compute_output(activity)

        return value

    def compute_neuron_input(self, state):
        result = self.bias.copy()
        for i, s in enumerate(state):
            result += (self.sign[:,i]*(s<<(self.shift1[:,i])))
        return result>>self.extra_bits
        # the above code approximates the following multiply using shifts
        #return np.dot(encoder, state) + bias

    def neuron(self, current):
        # this implements two different neuron models: one with state and
        #  one without state.  We'd only want one or the other on a chip.
        if self.has_neuron_state:
            # this is the accumulator implementation for a spike
            self.state = self.state + current
            self.state = np.where(self.state<0, 0, self.state)
            spikes = np.where(self.state>=(1<<self.state_bits), 1, 0)
            self.state[spikes>0] -= 1<<self.state_bits
        else:
            # this is the rng implementation for a spike
            #  (in a real chip, we'd use an LFSR for random number generation)
            rnd = self.rng.randint(0,1<<self.state_bits,len(current))
            spikes = np.where(rnd < current, 1, 0)
        return spikes

    def compute_output(self, activity):
        # given the spiking activity, apply the decoders and provide an output
        decoder_access = self.decoder[:,np.where(activity>0)[0]]
        if decoder_access.shape[1]>0:
            value = np.sum(decoder_access, axis=1)
        else:
            value = np.zeros(decoder_access.shape[0], dtype=int)

        if self.smoothing:
            # apply a low-pass filter
            dv = (value - self.smoothing_state) >> self.smoothing_shift
            self.smoothing_state += dv
            value = self.smoothing_state
        return value.copy() >> self.decoder_offset

