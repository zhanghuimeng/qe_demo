import tensorflow as tf
import argparse

from qe.qe_dataset import read_vocab, one_dataset_loader, load_dataset_from_lists
from qe.qe_model import Model

BATCH_SIZE = 50
MAX_EPOCH = 500
PATIENCE = 5
EMB_SIZE = 300
ENC_HIDDEN_SIZE = 50
DEC_HIDDEN_SIZE = 500
LR = 1.0


def test(vocab, test, model_addr):
    with tf.device('/cpu:0'):
        print("Loading vocabulary from %s and %s ..." % (vocab[0], vocab[1]))
        vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt = read_vocab(vocab[0], vocab[1])
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            src_vocab_size = sess.run(vocab_idx_src.size())
            tgt_vocab_size = sess.run(vocab_idx_tgt.size())
        print('Loaded src vocabulary size %d' % src_vocab_size)
        print('Loaded tgt vocabulary size %d' % tgt_vocab_size)
        if __name__ == "__main__":
            test_dataset = one_dataset_loader(
                src=test[0],
                tgt=test[1],
                hter=test[2],
                vocab_idx_src=vocab_idx_src,
                vocab_idx_tgt=vocab_idx_tgt,
                batch_size=BATCH_SIZE)
        else:
            test_dataset = load_dataset_from_lists(
                src=test[0],
                tgt=test[1],
                hter=test[2],
                vocab_idx_src=vocab_idx_src,
                vocab_idx_tgt=vocab_idx_tgt,
                batch_size=1)

    test_iter = test_dataset.make_initializable_iterator()
    test_ele = test_iter.get_next()
    model = Model(enc_hidden_size=ENC_HIDDEN_SIZE,
                  dec_hidden_size=DEC_HIDDEN_SIZE,
                  emb_size=EMB_SIZE,
                  src_vocab_size=src_vocab_size,
                  tgt_vocab_size=tgt_vocab_size,
                  learning_rate=LR,
                  src=test_ele['src'],
                  tgt=test_ele['tgt'],
                  src_len=test_ele['src_len'],
                  tgt_len=test_ele['tgt_len'],
                  hter=test_ele['hter'])
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_addr)
        print("model restored.")

        # tvars = tf.trainable_variables()
        # tvars_vals = sess.run(tvars)
        # for var, val in zip(tvars, tvars_vals):
        #     print(var.name, val)

        sess.run(tf.local_variables_initializer())  # 这个需要吗？
        sess.run(tf.tables_initializer())  # 看起来table需要单独initialize
        sess.run(test_iter.initializer)
        sess.run(model.mse_reset)
        sess.run(model.pearson_reset)
        try:
            while True:
                sess.run([model.mse_update, model.pearson_update, model.pred_update],
                         feed_dict={model.training: False})
        except tf.errors.OutOfRangeError:  # Thrown at the end of the epoch.
            mse, pearson, pred = sess.run([model.mse, model.pearson, model.all_pred],
                                          feed_dict={model.training: False})
            print('test: mse=%f, pearson=%f' % (mse, pearson))
            return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', type=str, nargs=2, help='Parallel vocab files (with ctrl)')
    parser.add_argument('--test', type=str, nargs=3, help='Parallel test files and HTER score')
    parser.add_argument('--model', type=str, help='Model directory.')
    parser.add_argument('--output', type=str, help='Output file.')
    args = parser.parse_args()
    pred = test(args.vocab, args.test, args.model)
    with open(args.output, 'w') as f:
        for p in pred:
            f.write("%f\n" % p)
