from scipy.stats.stats import pearsonr
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

data_dir = '../data/qe-2017/'
with open(data_dir + 'test.src', 'r', encoding='UTF-8') as f:
    src = f.read().splitlines()
with open(data_dir + 'test.mt', 'r', encoding='UTF-8') as f:
    mt = f.read().splitlines()
with open(data_dir + 'test.pe', 'r', encoding='UTF-8') as f:
    pe = f.read().splitlines()
with open(data_dir + 'test.hter', 'r') as f:
    gold = []
    for line in f.read().splitlines():
        gold.append(float(line))
# output_dir1 = '../kiwi_out/sentence_scores'
output_dir1 = '../model/birnn/test_epoch_12_output_0.pred'
with open(output_dir1, 'r') as f:
    hter1 = []
    for line in f.read().splitlines():
        hter1.append(float(line))

rmse = sqrt(mean_squared_error(gold, hter1))
pearson = pearsonr(hter1, gold)
mae = sqrt(mean_absolute_error(gold, hter1))
print("rmse:", rmse)
print("pearson:", pearson)
print("mae:", mae)

# n = len(src)
# l = []
# for i in range(n):
#     diff = abs(hter1[i] - gold[i])
#     l.append((diff, src[i], mt[i], pe[i], gold[i], hter1[i]))
# l = sorted(l, key=lambda tup: tup[0])
# # for i in itertools.chain(range(30), range(1970, 2000)):
# for i in range(n):
#     print(i)
#     print("diff=%f" % l[i][0])
#     print("src=%s" % l[i][1])
#     print('mt=%s' % l[i][2])
#     print('pe=%s' % l[i][3])
#     print('gold=%f' % l[i][4])
#     print('hter1=%f' % l[i][5])
