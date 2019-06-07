import tensorflow as tf


# https://github.com/tensorflow/tensorflow/issues/4814
def create_reset_metric(metric, scope='reset_metrics', **metric_args):
  with tf.variable_scope(scope) as scope:
    metric_op, update_op = metric(**metric_args)
    vars = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
    reset_op = tf.variables_initializer(vars)
  return metric_op, update_op, reset_op


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


class Model:
    def __init__(self, enc_hidden_size, dec_hidden_size,
                 src_vocab_size, tgt_vocab_size, emb_size, learning_rate,
                 src, tgt, src_len, tgt_len, hter,):
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.learning_rate = learning_rate
        self.training = tf.placeholder(dtype=tf.bool, name='isTraining')  # 是否正在训练
        self.all_pred = tf.get_variable(name='all_pred', dtype=tf.float32, shape=[0, 1])
        l2 = tf.contrib.layers.l2_regularizer(scale=1e-4)
        with tf.variable_scope('embedding'):
            # 细节：glorot_uniform_initializer
            self.src_emb = tf.get_variable("src_embeddings", [src_vocab_size, emb_size], dtype=tf.float32,
                                           initializer=tf.glorot_uniform_initializer(), regularizer=l2)
            self.tgt_emb = tf.get_variable("tgt_embeddings", [tgt_vocab_size, emb_size], dtype=tf.float32,
                                           initializer=tf.glorot_uniform_initializer(), regularizer=l2)
            # additive Gaussian noise (only in training)
            # batch normalization
            def true_fn(src_emb, tgt_emb):
                src_emb = gaussian_noise_layer(src_emb, std=0.1)
                tgt_emb = gaussian_noise_layer(tgt_emb, std=0.1)
                src_emb = tf.layers.batch_normalization(src_emb, gamma_regularizer=l2, beta_regularizer=l2)
                tgt_emb = tf.layers.batch_normalization(tgt_emb, gamma_regularizer=l2, beta_regularizer=l2)
                return src_emb, tgt_emb

            self.src_emb, self.tgt_emb = \
                tf.cond(self.training,
                        true_fn=lambda: true_fn(self.src_emb, self.tgt_emb),
                        false_fn=lambda: (self.src_emb, self.tgt_emb))

        with tf.variable_scope('src_rnn', initializer=tf.orthogonal_initializer()):
            self.src_rnn_cell = {
                'f': tf.nn.rnn_cell.GRUCell(enc_hidden_size, kernel_initializer=tf.glorot_uniform_initializer()),
                'w': tf.nn.rnn_cell.GRUCell(enc_hidden_size, kernel_initializer=tf.glorot_uniform_initializer())}
        with tf.variable_scope('tgt_rnn', initializer=tf.orthogonal_initializer()):
            self.tgt_rnn_cell = {
                'f': tf.nn.rnn_cell.GRUCell(enc_hidden_size, kernel_initializer=tf.glorot_uniform_initializer()),
                'w': tf.nn.rnn_cell.GRUCell(enc_hidden_size, kernel_initializer=tf.glorot_uniform_initializer())}
        with tf.variable_scope('attention'):
            self.weight_a = tf.get_variable('W_a', shape=[2 * self.dec_hidden_size, 1], dtype=tf.float32,
                                            initializer=tf.initializers.random_normal(0.1))
            self.dense = tf.layers.Dense(units=1)
        with tf.variable_scope('predict'):
            self.pred = self.predict(src, tgt, src_len, tgt_len)
            self.all_pred, self.pred_update = tf.contrib.metrics.streaming_concat(self.pred)
            self.loss = tf.losses.mean_squared_error(
                labels=tf.expand_dims(hter, 1),
                predictions=self.pred)
            self.bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate) \
                .minimize(self.loss)
            self.train_op = tf.group([self.bn_update_ops, self.train_op])
            self.mse, self.mse_update, self.mse_reset = create_reset_metric(
                tf.metrics.mean_squared_error,
                'mse',
                labels=tf.expand_dims(hter, 1),
                predictions=self.pred,
                name="mse")
            self.pearson, self.pearson_update, self.pearson_reset = create_reset_metric(
                tf.contrib.metrics.streaming_pearson_correlation,
                'pearson',
                labels=tf.expand_dims(hter, 1),
                predictions=self.pred,
                name="pearson")

    def predict(self, src, tgt, src_len, tgt_len):
        embedded_src = tf.nn.embedding_lookup(self.src_emb, src)
        embedded_tgt = tf.nn.embedding_lookup(self.tgt_emb, tgt)
        with tf.variable_scope('src_birnn'):
            src_h = tf.nn.bidirectional_dynamic_rnn(
                self.src_rnn_cell['f'],
                self.src_rnn_cell['w'],
                embedded_src,
                dtype=tf.float32,  # 如果不给定RNN initial state，则必须给定dtype（是状态的dtype！）
                sequence_length=src_len)
            src_h = src_h[0]  # 原来是(outputs, output_states)
            src_h = tf.concat(src_h, 2)
        with tf.variable_scope('tgt_birnn'):
            tgt_h = tf.nn.bidirectional_dynamic_rnn(
                self.tgt_rnn_cell['f'],
                self.tgt_rnn_cell['w'],
                embedded_tgt,
                dtype=tf.float32,
                sequence_length=tgt_len)
            tgt_h = tgt_h[0]
            tgt_h = tf.concat(tgt_h, 2)
        h = tf.concat([src_h, tgt_h], 1)
        # 对h进行attention
        # [66, 1000]
        def unbatch_h(h):
            # [1000]
            def unpack_h(h):
                # 然而这个broadcast比我想象得难用
                return tf.reduce_sum(tf.multiply(h, self.weight_a))
            a = tf.map_fn(unpack_h, h)
            a = tf.nn.softmax(a)
            v = tf.reduce_sum(tf.multiply(tf.expand_dims(a, 1), h), 0)
            return v
        v = tf.map_fn(unbatch_h, h)
        v = self.dense(v)
        pred = tf.nn.sigmoid(v)
        return pred