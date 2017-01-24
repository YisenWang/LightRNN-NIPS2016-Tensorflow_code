#!/usr/bin/python -u
import os
import time
import math
import collections

import numpy as np
import tensorflow as tf

import pdb

from reader import *
import mcmf
import rnn

flags = tf.flags
flags.DEFINE_string("data_path", "ptb_data", "Where the training/test data is stored.")

FLAGS = flags.FLAGS

class LSTMLM(object):
    def __init__(self, config, mode, device, reuse=None):
        self.config = config
        self.mode = mode
        if mode == "Train":
            self.is_training = True
            self.batch_size = self.config.train_batch_size
            self.step_size = self.config.train_step_size
        elif mode == "Valid":
            self.is_training = False
            self.batch_size = self.config.valid_batch_size
            self.step_size = self.config.valid_step_size 
        else:
            self.is_training = False
            self.batch_size = self.config.test_batch_size
            self.step_size = self.config.test_step_size 

        vocab_size = int(math.ceil(math.sqrt(config.vocab_size)))
        embed_dim = config.word_embedding_dim     
        lstm_size = config.lstm_size              
        lstm_layers = config.lstm_layers          
        lstm_forget_bias = config.lstm_forget_bias
        batch_size = self.batch_size
        step_size = self.step_size

        with tf.device(device), tf.name_scope(mode), tf.variable_scope("LSTMLM", reuse=reuse):
            # INPUTS and TARGETS
            self.inputs  = tf.placeholder(tf.int32, [batch_size, step_size, 2]) 
            self.targets = tf.placeholder(tf.int32, [batch_size, step_size, 2])

            # Inititial state
            self.initial_state = tf.placeholder(tf.float32, 
                [batch_size, lstm_size * 2 * lstm_layers])

            # WORD EMBEDDING
            stdv = np.sqrt(1. / vocab_size)
            self.word_embedding_r = tf.get_variable("word_embedding_r", [
                vocab_size, embed_dim], initializer=tf.random_uniform_initializer(-stdv, stdv))
            self.word_embedding_c = tf.get_variable("word_embedding_c", [
                vocab_size, embed_dim], initializer=tf.random_uniform_initializer(-stdv, stdv))

            #input_r = tf.reshape(inputs[:,0], [batch_size, step_size])
            #pdb.set_trace()
            input_c = self.inputs[:,:,1]

            #input_r = tf.nn.embedding_lookup(self.word_embedding_r, input_r)
            input_c = tf.nn.embedding_lookup(self.word_embedding_c, input_c)

            #inputs = tf.nn.embedding_lookup(self.word_embedding, self.inputs)

            # INPUT DROPOUT 
            if self.is_training and self.config.dropout_prob > 0:
                input_c = tf.nn.dropout(input_c, keep_prob=1 - config.dropout_prob)

            # LSTM
            softmax_w_c = tf.get_variable("softmax_w_c", [lstm_size, vocab_size], dtype=tf.float32)
            softmax_w_r = tf.get_variable("softmax_w_r", [lstm_size, vocab_size], dtype=tf.float32)
            softmax_b_c = tf.get_variable("softmax_b_c", [vocab_size], dtype=tf.float32)
            softmax_b_r = tf.get_variable("softmax_b_r", [vocab_size], dtype=tf.float32)

            self.lstm = rnn.MultiLSTM(embed_dim, lstm_size, lstm_layers, lstm_forget_bias, scope="LSTM")
            lstm_dropout = self.config.dropout_prob if self.is_training and self.config.dropout_prob > 0 else None
            state_r = self.initial_state

            input_c = tf.unstack(input_c, axis=1)

            logits_r = []
            logits_c = []
            self.probs_r = []
            self.probs_c = []

            for t, _input_c in enumerate(input_c):
                if t>0:
                    tf.get_variable_scope().reuse_variables()
                output_r, state_c = self.lstm(_input_c, state_r, output_dropout=lstm_dropout)
                logit_r = tf.matmul(output_r, softmax_w_r) + softmax_b_r
                row = tf.argmax(logit_r, axis=1)
                input_r = tf.nn.embedding_lookup(self.word_embedding_r, row)

                if self.is_training and self.config.dropout_prob > 0:
                    input_r = tf.nn.dropout(input_r, keep_prob=1 - config.dropout_prob)

                tf.get_variable_scope().reuse_variables()

                output_c, state_r = self.lstm(input_r, state_c, output_dropout=lstm_dropout)
                logit_c = tf.matmul(output_c, softmax_w_c) + softmax_b_c

                logits_r.append(logit_r)
                logits_c.append(logit_c)

                self.probs_r.append(-tf.nn.log_softmax(logit_r))
                self.probs_c.append(-tf.nn.log_softmax(logit_c))

            self.final_state = state_r

            logits_r = tf.reshape(tf.concat(1, logits_r), [-1, vocab_size])
            logits_c = tf.reshape(tf.concat(1, logits_c), [-1, vocab_size])

            self.probs_r = tf.reshape(tf.concat(1, self.probs_r), [-1, vocab_size])
            self.probs_c = tf.reshape(tf.concat(1, self.probs_c), [-1, vocab_size])


            # Loss
            labels = tf.reshape(self.targets, [-1,2])
            label_r = labels[:,0]
            label_c = labels[:,1]

            self.loss_r = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_r, label_r)
            self.loss_c = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_c, label_c)

            #self.loss = [self.loss_r, self.loss_c]
            self.loss = [(self.loss_r[i] + self.loss_c[i]) for i in range(batch_size * step_size)]


            training_losses = self.loss

            self.cost = tf.reduce_sum(self.loss_r + self.loss_c)

            if self.is_training:
                self.lr = tf.Variable(0.0, trainable=False)
                optimizer = tf.train.AdagradOptimizer(self.lr, config.adagrad_eps)
                tvars = tf.trainable_variables()
                grads = tf.gradients([tf.reduce_sum(loss) / batch_size for loss in training_losses],
                    tvars)
                grads = [tf.clip_by_norm(grad, config.max_grad_norm) if grad is not None else grad for grad in grads]
                self.eval_op = optimizer.apply_gradients(zip(grads, tvars))
            else:
                self.eval_op = tf.no_op()

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.lr, learning_rate))

    def get_initial_state(self):
        return np.zeros([self.batch_size, self.config.lstm_size * 2 * self.config.lstm_layers], dtype=np.float32)

class Config(object):
    epoch_num = 100
    train_batch_size = 128
    train_step_size = 20
    valid_batch_size = 128
    valid_step_size = 20
    test_batch_size = 20
    test_step_size = 1
    word_embedding_dim = 512
    lstm_layers = 1
    lstm_size = 512
    lstm_forget_bias = 0.0
    max_grad_norm = 0.25
    init_scale = 0.05
    learning_rate = 0.2
    decay = 0.5
    decay_when = 1.0
    dropout_prob = 0.5
    adagrad_eps = 1e-5
    vocab_size = 10000

class LearningRateUpdater(object):
    def __init__(self, init_lr, decay_rate, decay_when):
        self._init_lr = init_lr
        self._decay_rate = decay_rate
        self._decay_when = decay_when
        self._current_lr = init_lr
        self._last_ppl = -1

    def get_lr(self):
        return self._current_lr

    def update(self, cur_ppl):
        if self._last_ppl > 0 and self._last_ppl - cur_ppl < self._decay_when:
            current_lr = self._current_lr * self._decay_rate
            INFO_LOG("learning rate: {} ==> {}".format(self._current_lr, current_lr))
            self._current_lr = current_lr
        self._last_ppl = cur_ppl

def run(session, model, reader, word_dict, verbose=True):
    state = model.get_initial_state()
    total_cost = 0
    total_word_cnt = 0
    start_time = time.time()
    prob_r = []
    prob_c = []
    #word_loss_dict={}
    loss_dict_r = collections.defaultdict(list)
    loss_dict_c = collections.defaultdict(list)

    for batch in reader.yieldSpliceBatch(model.mode, model.batch_size, model.step_size, word_dict):
        batch_id, batch_num, x, y, word_cnt = batch
        feed = {model.inputs: x, model.targets:y, model.initial_state: state}
        cost, state, _, prob1r, prob1c = session.run([model.cost, model.final_state, model.eval_op, model.probs_r, model.probs_c], feed)
        total_cost += cost
        total_word_cnt += word_cnt
        if model.mode == "Train":
            word = np.reshape(x, (-1,2))
            for index, i in enumerate(word):
                loss_dict_r[tuple(i)].append(prob1r[index])
                loss_dict_c[tuple(i)].append(prob1c[index])

        if verbose and (batch_id % max(10, batch_num//10)) == 0:
            ppl = np.exp(total_cost / total_word_cnt)
            wps = total_word_cnt / (time.time() - start_time)
            print "  [%5d/%d]ppl: %.3f speed: %.0f wps costs %.3f words %d" % (
                batch_id, batch_num, ppl, wps, total_cost, total_word_cnt)

    return total_cost, total_word_cnt, np.exp(total_cost / total_word_cnt), loss_dict_r, loss_dict_c

def main(_):
    reader = Reader(FLAGS.data_path)
    config = Config()

    gpuid = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpuid)
    device = '/gpu:0'
    
    lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_when)

    graph = tf.Graph()
    with graph.as_default():
        trainm = LSTMLM(config, device=device, mode="Train", reuse=False)
        validm = LSTMLM(config, device=device, mode="Valid", reuse=True)
        testm  = LSTMLM(config, device=device, mode="Test", reuse=True)

    word_dict = reader.get_word_dict()
    
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    last_ppl = 1000.0
    with tf.Session(graph=graph, config=session_config) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(config.epoch_num):
            trainm.update_lr(session, lr_updater.get_lr())
            INFO_LOG("Epoch {}, learning rate: {}".format(epoch + 1, lr_updater.get_lr()))
            cost, word_cnt, ppl, loss_dict_r, loss_dict_c = run(session, trainm, reader, word_dict)
            INFO_LOG("Epoch %d Train perplexity %.3f words %d" % (epoch + 1, ppl, word_cnt))

            #pdb.set_trace()

            cost, word_cnt, ppl, _, _ = run(session, validm, reader, word_dict)
            INFO_LOG("Epoch %d Valid perplexity %.3f words %d" % (epoch + 1, ppl, word_cnt))

            lr_updater.update(ppl)
            cost, word_cnt, ppl, _, _ = run(session, testm, reader, word_dict)
            INFO_LOG("Epoch %d Test perplexity %.3f words %d" % (epoch + 1, ppl, word_cnt))

            if (last_ppl - ppl) < 5:
                
                word_dict = mcmf.MCMF(word_dict, loss_dict_r, loss_dict_c)

                lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_when)

                INFO_LOG("Epoch %d Allocation Success" % (epoch + 1))

            last_ppl = ppl


if __name__ == '__main__':
    tf.app.run()

