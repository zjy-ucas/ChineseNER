import os
import tensorflow as tf
from model import Model
from loader import load_data
from utils import BatchManager, test_ner

FLAGS = tf.app.flags.FLAGS
# path for log, model and result
tf.app.flags.DEFINE_string("log_path", './log', "path for log files")
tf.app.flags.DEFINE_string("model_path", './weights', "path to save model")
tf.app.flags.DEFINE_string("result_path", './results', "path to save result")
tf.app.flags.DEFINE_string("train_file", './data/SIGHAN.NER.train', "path for train data")
tf.app.flags.DEFINE_string("dev_file", './data/SIGHAN.NER.dev', "path for valid data")
tf.app.flags.DEFINE_string("test_file", './data/SIGHAN.NER.test', "path for test data")
# config for model
tf.app.flags.DEFINE_boolean("lower", True, "True for lowercase all characters")
tf.app.flags.DEFINE_string("pre_emb", "./embedding/wiki_word2vec.pkl",
                           "path for pre-trained embedding, False for randomly initialize")
tf.app.flags.DEFINE_integer("min_freq", 2, "")
tf.app.flags.DEFINE_integer("word_max_len", 100, "maximum words in a sentence")
tf.app.flags.DEFINE_integer("word_dim", 100, "dimension of char embedding")
tf.app.flags.DEFINE_integer("word_hidden_dim", 150, "dimension of word LSTM hidden units")
tf.app.flags.DEFINE_string("feature_dim", 4, "dimension of extra features, 0 for not used")
# config for training process

tf.app.flags.DEFINE_float("dropout", 0.5, "dropout rate")
tf.app.flags.DEFINE_float("clip", 5, "gradient to clip")
tf.app.flags.DEFINE_float("lr", 0.001, "initial learning rate")
tf.app.flags.DEFINE_integer("max_epoch", 150, "maximum training epochs")
tf.app.flags.DEFINE_integer("batch_size", 20, "num of sentences per batch")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "steps per checkpoint")
tf.app.flags.DEFINE_integer("valid_batch_size", 100, "num of sentences per batch")


def create_model(session, word_to_id, id_to_tag):
    # create model, reuse parameters if exists
    model = Model("tagger", word_to_id, id_to_tag, FLAGS)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        model.logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def main(_):
    if not os.path.isdir(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    if not os.path.isdir(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)
    if not os.path.isdir(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)
    tag_to_id = {"O": 0, "B-LOC": 1, "I-LOC": 2,
                 "B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6}
    # load data
    id_to_word, id_to_tag, train_data, dev_data, test_data = load_data(FLAGS, tag_to_id)
    train_manager = BatchManager(train_data, len(id_to_tag), FLAGS.word_max_len, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, len(id_to_tag), FLAGS.word_max_len, FLAGS.valid_batch_size)
    test_manager = BatchManager(test_data, len(id_to_tag), FLAGS.word_max_len, FLAGS.valid_batch_size)
    with tf.Session() as sess:
        model = create_model(sess, id_to_word, id_to_tag)
        loss = 0
        best_test_f1 = 0
        steps_per_epoch = len(train_data) // FLAGS.batch_size + 1
        for _ in range(FLAGS.max_epoch):
            iteration = (model.global_step.eval()) // steps_per_epoch + 1
            train_manager.shuffle()
            for batch in train_manager.iter_batch():
                global_step = model.global_step.eval()
                step = global_step % steps_per_epoch
                batch_loss = model.run_step(sess, True, batch)
                loss += batch_loss / FLAGS.steps_per_checkpoint
                if global_step % FLAGS.steps_per_checkpoint == 0:
                    model.logger.info("iteration:{} step:{}/{}, NER loss:{:>9.6f}"
                                      .format(iteration,
                                              step,
                                              steps_per_epoch,
                                              loss))
                    loss = 0

            model.logger.info("validating ner")
            ner_results = model.predict(sess, dev_manager)
            eval_lines = test_ner(ner_results, FLAGS.result_path)
            for line in eval_lines:
                model.logger.info(line)
            test_f1 = float(eval_lines[1].strip().split()[-1])
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                model.logger.info("new best f1 score:{:>.3f}".format(test_f1))
                model.logger.info("saving model ...")
                checkpoint_path = os.path.join(FLAGS.model_path, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        # test model
        model.logger.info("testing ner")
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
        model.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        ner_results = model.predict(sess, test_manager)
        eval_lines = test_ner(ner_results, FLAGS.result_path)
        for line in eval_lines:
            model.logger.info(line)


if __name__ == "__main__":
    tf.app.run(main)
