import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, help='The corpus file')
parser.add_argument('--name', type=str, help='The name of the vocab file generated')
parser.add_argument('--limit', type=int, default=0,
                    help='The limit to the size of the vocab file, default to no limit (0)')
parser.add_argument('--ctrl', type=str, default=['<pad>', '<eos>', '<go>', '<unk>'],
                    help='The control chars to be added')
args = parser.parse_args()

vocab_cnt = {}
total_cnt = 0
cov_cnt = 0
print('Reading %s ...' % args.corpus)
with open(args.corpus, 'r') as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            vocab_cnt[word] = vocab_cnt.get(word, 0) + 1
            total_cnt += 1
print('Total number of unique words: %d' % len(vocab_cnt))
limited_vocab = sorted(vocab_cnt.keys(), key=vocab_cnt.get, reverse=True)
# 加入控制字符（方便之后直接使用tf.contrib.lookup.index_table_from_file）
limited_vocab = args.ctrl + limited_vocab
if not args.limit == 0 and len(limited_vocab) > args.limit:
    limited_vocab = limited_vocab[:args.limit]
print('Writing to %s ...' % args.name)
with open(args.name, 'w') as f:
    for word in limited_vocab:
        f.write('%s\n' % word)
        cov_cnt += vocab_cnt.get(word, 0)
print('Finished writing')
print('Vocab size: %d' % len(limited_vocab))
print('Coverage: %f' % (cov_cnt / total_cnt))
