"""
RNN model used for language models.

Model components vary slightly for train vs validation/test, e.g. dropout
only applied during training, no optimizer for validation/test models.

Model can be either:
1. specific to an individual character
2. shared by multiple characters. in this case, the embedding and RNN
parameters are shared but the projection layer parameters are specific
(different) for each character.
"""


import tensorflow as tf
from tensorflow.python.training.training_util import get_or_create_global_step


class Model(object):
    """The language model.

    1. Initialize parameters
    2. Create placeholders for input/output data
    2. Create word embedding matrix w/ dropout
    3. Create RNN cell w/ multi-layers + dropout + masking
    4. Create projection layer for each character
    5. Define cross entropy loss function
    6. Optimization (apply/clip gradients, minimize loss)
    """

    def __init__(self, args, is_training, config, init_embed, name):
        self.args = args
        self.is_training = is_training
        self.config = config
        self.init_embed = init_embed
        self.name = name
        self.lr = tf.Variable(self.config['lr'], trainable=False)
        self.logits, self.cost, self.train_op = [], [], []

        # only create TensorBoard summaries for train/test models
        if self.args.save_path and self.name != 'Test' and self.name != 'Generate':
            self.file_writer = tf.summary.FileWriter(self.args.save_path + '/'
                                                     + self.name,
                                                     graph=tf.get_default_graph())

    def _create_placeholders(self):
        """Define placeholders for input and output."""
        self.input_data = tf.placeholder(tf.int32,
                                         [self.config['batch_size'], None])
        self.targets = tf.placeholder(tf.int32,
                                      [self.config['batch_size'], None])

    def _create_embedding(self):
        """Create word embeddings for model which are either initialized
        randomly or with GLoVe vectors.
        """
        # words represented as dense vector (embedding)
        with tf.device('/cpu:0'):
            # embedding initialized randomly or with GLoVe vectors
            embedding = tf.get_variable('embedding',
                                        shape=[self.config['vocab_size'],
                                               self.config['embed_size']],
                                        initializer=tf.constant_initializer(
                                            self.init_embed),
                                        dtype=tf.float32)
            # select vectors corresponding to input words
            self.inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # only apply dropout during training (not during Valid/Test)
        if self.is_training and self.config['keep_prob'] < 1:
            # dropout applied to inputs
            self.inputs = tf.nn.dropout(self.inputs, self.config['keep_prob'])

    def _create_rnn(self):
        """Create RNN cell which includes basicrnn/basiclstm/gru, dropout,
        multiple layers, masking (for variable length sequences).
        """
        # must define functions so it works with MultiRNNCell
        if self.args.rnn_type == 'rnn':
            def rnn_cell():
                return tf.contrib.rnn.BasicRNNCell(self.config['embed_size'],
                                                   reuse=tf.get_variable_scope().reuse)
        elif self.args.rnn_type == 'lstm':
            def rnn_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.config['embed_size'],
                                                    forget_bias=1.0,
                                                    state_is_tuple=True,
                                                    reuse=tf.get_variable_scope().reuse)
        else:
            def rnn_cell():
                return tf.contrib.rnn.GRUCell(self.config['embed_size'],
                                              reuse=tf.get_variable_scope().reuse)
        attn_cell = rnn_cell
        # only apply dropout during training
        if self.is_training and self.config['keep_prob'] < 1:
            # DropoutWrapper applies dropout between layers
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(rnn_cell(),
                                                     output_keep_prob=self.config['keep_prob'])
        # create multi-layer network with num_layers
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(self.config['num_layers'])],
            state_is_tuple=True)

        # initialize cell state to zero
        self.initial_state = cell.zero_state(self.config['batch_size'],
                                             dtype=tf.float32)

        # create mask which contains 1's for actual words, 0's for padded words
        self.mask = tf.sign(tf.to_float(self.input_data))
        self.seq_len = tf.reduce_sum(self.mask, reduction_indices=1)

        # create RNN unrolled for num_steps
        with tf.variable_scope('RNN'):
            outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=self.inputs,
                                               initial_state=self.initial_state,
                                               sequence_length=self.seq_len)

        self.mask = tf.reshape(self.mask, [-1])
        self.output = tf.reshape(outputs, [-1, self.config['embed_size']])
        self.final_state = state

    def _create_projection(self):
        """Create projection layer to map from outputs to words.

        Separate projection layers defined for each character/speaker. This
        works the same if model only trained for one character. Variables
        will be a list with an entry for each character.
        """
        for speaker in self.args.speakers:
            softmax_w = tf.get_variable('softmax_w_' + speaker,
                                        [self.config['embed_size'],
                                        self.config['vocab_size']],
                                        dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b_' + speaker,
                                        [self.config['vocab_size']],
                                        dtype=tf.float32)
            self.logits.append(tf.matmul(self.output, softmax_w) + softmax_b)

    def _create_loss(self):
        """Create loss to minimize -log probability of target words"""
        for i, speaker in enumerate(self.args.speakers):
            # loss is weighted by mask, and mask = 0 for padded words
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits[i]],
                [tf.reshape(self.targets, [-1])],
                [self.mask])

            self.cost.append(tf.reduce_sum(loss,
                                           name='cost/' + speaker))

    def _create_optimizer(self):
        """Define training ops (optimizer, gradients w/ clipping)

        Create separate train op for each speaker, so projection layer
        variables specific to each speaker will only update when that speaker
        is training.
        """
        # only update variables defined as trainable
        optimizer = tf.train.AdamOptimizer(self.config['lr'])
        for i, speaker in enumerate(self.args.speakers):
            # get trainable variables (embedding, RNN, speaker
            # specific projection layer)
            scope = '.*embedding|.*RNN|.*' + speaker
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope=scope)
            # compute gradients of cost w.r.t. tvars, cap them at max_grad_norm
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost[i], tvars),
                                              self.config['max_grad_norm'])
            # create optimizer using gradients, vars for each speaker
            self.train_op.append(optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=get_or_create_global_step(),
                name='train_op/' + speaker))

    def build_graph(self):
        """Build graph with all components for the model."""
        self._create_placeholders()
        self._create_embedding()
        self._create_rnn()
        self._create_projection()
        self._create_loss()
        if self.is_training:
            self._create_optimizer()
