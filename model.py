import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from tensorflow.python.ops import rnn_cell
from utils import get_logger, load_word2vec, calculate_accuracy
from tensorflow.python.ops import init_ops


class Model(object):
    def __init__(self, name, word_to_id, id_to_tag, parameters):

        self.logger = get_logger(name)
        self.params = parameters
        self.num_words = len(word_to_id)
        self.learning_rate = self.params.lr
        self.global_step = tf.Variable(0, trainable=False)
        self.initializer = tf.contrib.layers.xavier_initializer
        self.tags = [tag for i, tag in id_to_tag.items()]
        self.tag_num = len(self.tags)

        # add placeholders for the model
        self.inputs = tf.placeholder(dtype=tf.int32,
                                     shape=[None, self.params.word_max_len],
                                     name="Inputs")
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=[None, self.params.word_max_len],
                                     name="Labels")
        self.lengths = tf.placeholder(dtype=tf.int32,
                                      shape=[None],
                                      name="Lengths")
        if self.params.feature_dim:
            self.features = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.params.word_max_len,
                                                  self.params.feature_dim],
                                           name="Features")
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")
        # get embedding of input sequence
        embedding = self.get_embedding(self.inputs, word_to_id)
        # apply dropout on embedding
        rnn_inputs = tf.nn.dropout(embedding, self.dropout)
        # concat extra features with embedding
        if self.params.feature_dim:
            rnn_inputs = tf.concat(2, [rnn_inputs, self.features])
        # extract features
        rnn_features = self.bilstm_layer(rnn_inputs)
        # projection layer
        self.scores = self.project_layer(rnn_features, self.tag_num)
        # calculate loss of crf layer
        self.trans, self.loss = self.loss_layer(self.scores, self.tag_num)
        # optimizer of the model
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        # apply grad clip to avoid gradient explosion
        grads_vars = self.opt.compute_gradients(self.loss)
        capped_grads_vars = [(tf.clip_by_value(g, -self.params.clip, self.params.clip), v)
                             for g, v in grads_vars]  # gradient capping
        self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())

    def get_embedding(self, inputs, id_to_word):
        # embedding layer for input projection
        with tf.variable_scope("Embedding"), tf.device('/cpu:0'):
            if not self.params.pre_emb:
                embedding = tf.get_variable("word_emb",
                                            [self.num_words, self.params.word_dim],
                                            initializer=init_ops.uniform_unit_scaling_initializer())
            else:
                print("load word2vec")
                embedding = tf.get_variable("word_emb",
                                            dtype=tf.float32,
                                            initializer=np.asarray(
                                                load_word2vec(self.params.pre_emb, id_to_word),
                                            dtype=np.float32))

        x = tf.nn.embedding_lookup(embedding, inputs)
        return x

    def bilstm_layer(self, inputs):
        # bidirectional lstm layer for feature extration
        with tf.variable_scope("BiLSTM"):
            fw_cell = rnn_cell.LSTMCell(self.params.word_hidden_dim,
                                        use_peepholes=True,
                                        initializer=self.initializer())
            bw_cell = rnn_cell.LSTMCell(self.params.word_hidden_dim,
                                        use_peepholes=True,
                                        initializer=self.initializer())
            length64 = tf.cast(self.lengths, tf.int64)
            forward_output, _ = tf.nn.dynamic_rnn(
                fw_cell,
                inputs,
                dtype=tf.float32,
                sequence_length=self.lengths,
                scope="fw"
            )
            backward_output, _ = tf.nn.dynamic_rnn(
                bw_cell,
                tf.reverse_sequence(inputs, length64, seq_dim=1),
                dtype=tf.float32,
                sequence_length=self.lengths,
                scope="bw"
            )
            backward_output = tf.reverse_sequence(backward_output, length64, seq_dim=1)
            # concat forward and backward outputs into a 2*hiddenSize vector
            outputs = tf.concat(2, [forward_output, backward_output])
            lstm_features = tf.reshape(outputs, [-1, self.params.word_hidden_dim * 2])
            return lstm_features

    def project_layer(self, lstm_features, tag_num):
        # projection layer
        with tf.variable_scope("Project",
                               initializer=self.initializer()):
            w1 = tf.get_variable(
                'W1',
                [self.params.word_hidden_dim * 2, tag_num],
                regularizer=tf.contrib.layers.l2_regularizer(0.001))
            b1 = tf.get_variable(
                'b1', [tag_num])
            scores = tf.batch_matmul(lstm_features, w1) + b1
            scores = tf.reshape(scores, [-1, self.params.word_max_len, tag_num])
            return scores

    def loss_layer(self, scores, tag_num):
        # crf layer
        with tf.variable_scope("CRF"):
            trans = tf.get_variable('trans',
                                    shape=[tag_num, tag_num],
                                    initializer=self.initializer())
            log_likelihood, _ = crf_log_likelihood(scores,
                                                   self.labels,
                                                   self.lengths,
                                                   trans)
            loss = tf.reduce_mean(-1.0 * log_likelihood)
            return trans, loss

    def create_feed_dict(self, is_train, **kwargs):
        feed_dict = {
            self.inputs: kwargs["words"],
            self.lengths: kwargs["len"],
            self.features: kwargs["features"],
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.labels] = kwargs["tags"]
            feed_dict[self.dropout] = self.params.dropout
        return feed_dict

    def run_step(self, sess, is_train, batch):
        feed_dict = self.create_feed_dict(is_train, **batch)
        if is_train:
            loss, _ = sess.run(
                [self.loss, self.train_op],
                feed_dict)
            return loss
        else:
            scores = sess.run(self.scores, feed_dict)
            return scores

    @staticmethod
    def decode(scores, lengths, matrix):
        # inference final labels usa viterbi Algorithm
        paths = []
        for score, length in zip(scores, lengths):
            score = score[:length]
            path, _ = viterbi_decode(score, matrix)
            paths.append(path)
        return paths

    def valid(self, sess, data):
        trans = self.trans.eval()
        total_correct = 0
        total_labels = 0
        for batch in data.iter_batch():
            lengths = batch["len"]
            tags = batch["tags"]
            scores = self.run_step(sess, None, batch)
            batch_paths = self.decode(scores, lengths, trans)
            batch_correct, batch_total = calculate_accuracy(tags, batch_paths, lengths)
            total_correct += batch_correct
            total_labels += batch_total
        return total_correct / total_labels

    def predict(self, sess, data):
        results = []
        trans = self.trans.eval()
        for batch in data.iter_batch():
            tags = batch["tags"]
            lengths = batch["len"]
            str_lines = batch["str_lines"]
            end_of_doc = batch["end_of_doc"]
            scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(batch)):
                result = []
                for char, gold, pred in zip(str_lines[i][:lengths[i]],
                                            tags[i][:lengths[i]],
                                            batch_paths[i][:lengths[i]]):
                    result.append(" ".join([char, self.tags[int(gold)], self.tags[int(pred)]]))
                results.append([result, end_of_doc[i]])
        return results

