# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
from BestExporter import *
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("early_stopping", 10,
                     "")

flags.DEFINE_string("loss_type", "mse", "The type of loss function to use.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("eval_steps", 1000,
                     "How often to evaluate the model.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, score=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      score: (Optional) double. The score of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.score = score


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               score,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.score = score
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_lines(cls, input_file, convert2f=False):
    """Reads lines of file."""
    with tf.gfile.Open(input_file, "r") as f:
      lines = []
      for line in f.read().splitlines():
        if convert2f:
          line = float(line)
        lines.append(line)
      return lines

class QESentProcessor(DataProcessor):
  """Processor for the QE sentence data set."""

  def __init__(self):
    self.language = "en"

  def get_train_examples(self, data_dir):
    """See base class."""
    src_lines = self._read_lines(os.path.join(data_dir, 'train.src'))
    tgt_lines = self._read_lines(os.path.join(data_dir, 'train.mt'))
    scores = self._read_lines(os.path.join(data_dir, 'train.hter'), True)
    examples = []
    for (i, (src, tgt, score)) in enumerate(zip(src_lines, tgt_lines, scores)):
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(src)
      text_b = tokenization.convert_to_unicode(tgt)
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, score=score))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    src_lines = self._read_lines(os.path.join(data_dir, 'dev.src'))
    tgt_lines = self._read_lines(os.path.join(data_dir, 'dev.mt'))
    scores = self._read_lines(os.path.join(data_dir, 'dev.hter'), True)
    examples = []
    for (i, (src, tgt, score)) in enumerate(zip(src_lines, tgt_lines, scores)):
      guid = "dev-%d" % (i)
      text_a = tokenization.convert_to_unicode(src)
      text_b = tokenization.convert_to_unicode(tgt)
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, score=score))
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    src_lines = self._read_lines(os.path.join(data_dir, 'test.2017.src'))
    tgt_lines = self._read_lines(os.path.join(data_dir, 'test.2017.mt'))
    scores = self._read_lines(os.path.join(data_dir, 'test.2017.hter'), True)
    examples = []
    for (i, (src, tgt, score)) in enumerate(zip(src_lines, tgt_lines, scores)):
      guid = "test-%d" % (i)
      text_a = tokenization.convert_to_unicode(src)
      text_b = tokenization.convert_to_unicode(tgt)
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, score=score))
    return examples

def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        score=0.0,
        is_real_example=False)

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For regression tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("score: %f" % (example.score))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      score=example.score,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["scores"] = create_float_feature([feature.score])  # 这个应该是单数还是复数呢
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "scores": tf.FixedLenFeature([], tf.float32),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, use_one_hot_embeddings):
  """Creates a regression model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [1], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    predictions = tf.matmul(output_layer, output_weights, transpose_b=True)
    predictions = tf.nn.bias_add(predictions, output_bias)
    predictions = tf.reshape(predictions, [-1])  # 注意形状的差别：prediction是[12,1]，label是[12]
    # 这里的loss是自定义的。
    if FLAGS.loss_type == "mse":
      loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
    elif FLAGS.loss_type == "xent":
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions)
      loss = tf.reduce_mean(loss)
      predictions = tf.sigmoid(predictions)
    else:
      tf.logging.error("Unknown loss type %s" % FLAGS.loss_type)
      exit(-1)

    # return loss, predictions
    # 那么应该返回sigmoid
    return loss, predictions


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""
  # 为了summary，多加了一点东西

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    scores = features["scores"]
    is_real_example = None
    if "is_real_example" in features:  # 这一段暂时不太确定……
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(scores), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # 这里的loss和model已经加上自定义的部分了
    loss, predictions = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, scores, use_one_hot_embeddings)

    # 这段是读checkpoint，虽然不知道scaffold_fn是啥
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # 有时我也许应该控制一部分不进行训练
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    # 所以这个是叫model_fn，实际上可以进行训练/验证/测试……
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

      # 会自动保存，不需要
      # tf.summary.scalar("loss", loss)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(labels, predictions, is_real_example):
        # 所以is_real_example大概只是说是不是用来padding的example吧
        rmse = tf.metrics.root_mean_squared_error(
          labels=labels,
          predictions=predictions,
          weights=is_real_example
        )
        mae = tf.metrics.mean_absolute_error(
          labels=labels,
          predictions=predictions,
          weights=is_real_example
        )
        # Values of eval_metric_ops must be (metric_value, update_op) tuples
        # 这可很有趣……
        pearson = tf.contrib.metrics.streaming_pearson_correlation(
          labels=labels,
          predictions=predictions,
          weights=is_real_example
        )
        return {
            "eval_rmse": rmse,
            "eval_mae": mae,
            "eval_pearson": pearson
        }

      # 函数指针和函数参数……（真是搞不懂）
      eval_metrics = (metric_fn,
                      [scores, predictions, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"predictions": predictions},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "qe-sent": QESentProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  # 当然也得GPU放得下才行
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  # loss
  if FLAGS.loss_type not in ["mse", "xent"]:
    raise ValueError("Loss type not found: %s" % FLAGS.loss_type)

  processor = processors[task_name]()

  # 这个我不管，让BERT自己搞去吧
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # 这都是啥（除了checkpoint以外）
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  # 这又是啥
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  # 这里可以设置600s的间隔，但代价是不能设steps（所以还是算了）
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  # 重点是整一个estimator能用的model_fn
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  # 我看到了eval/predict_batch_size，那么请问怎么让model_fn eval呢？
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train and FLAGS.do_eval and FLAGS.do_predict:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)

    dev_examples = processor.get_dev_examples(FLAGS.data_dir)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(dev_examples) % FLAGS.eval_batch_size != 0:
        dev_examples.append(PaddingInputExample())

    dev_file = os.path.join(FLAGS.output_dir, "dev.tf_record")
    file_based_convert_examples_to_features(
      dev_examples, FLAGS.max_seq_length, tokenizer, dev_file)

    dev_drop_remainder = True if FLAGS.use_tpu else False
    dev_input_fn = file_based_input_fn_builder(
        input_file=dev_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=dev_drop_remainder)

    # 在此处创建predict
    test_examples = processor.get_test_examples(FLAGS.data_dir)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(test_examples) % FLAGS.eval_batch_size != 0:
        test_examples.append(PaddingInputExample())

    test_file = os.path.join(FLAGS.output_dir, "test.tf_record")
    file_based_convert_examples_to_features(test_examples,
                                            FLAGS.max_seq_length, tokenizer,
                                            test_file)

    test_drop_remainder = True if FLAGS.use_tpu else False
    test_input_fn = file_based_input_fn_builder(
      input_file=test_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=test_drop_remainder)

    # 做不到了，干脆直接用测试集算了……
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
    eval_spec = tf.estimator.EvalSpec(
      input_fn=test_input_fn,  # 变成test了
      steps=FLAGS.eval_steps,
      throttle_secs=120,
    )
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    '''
    step = 0
    max_pearson = -1
    cnt = 0
    while step < num_train_steps:
      estimator.train(input_fn=train_input_fn, steps=FLAGS.eval_steps)
      step += FLAGS.eval_steps
      tf.logging.info("Begin evaluating")

      dev_result = estimator.evaluate(input_fn=dev_input_fn, steps=None, name="dev")
      dev_metric_file = os.path.join(FLAGS.output_dir, "dev_metric-%d.txt" % step)
      with tf.gfile.GFile(dev_metric_file, "w") as writer:
        tf.logging.info("***** Dev results *****")
        # tf.summary.scalar("dev/rmse", dev_result["eval_rmse"])
        # tf.summary.scalar("dev/mae", dev_result["eval_mae"])
        # tf.summary.scalar("dev/pearson", dev_result["eval_pearson"])
        pearson = dev_result["eval_pearson"]
        for key in sorted(dev_result.keys()):
          tf.logging.info("  %s = %s", key, str(dev_result[key]))
          writer.write("%s = %s\n" % (key, str(dev_result[key])))
      # predict没有steps
      dev_result = estimator.predict(input_fn=dev_input_fn)
      dev_pred_file = os.path.join(FLAGS.output_dir, "dev_pred-%d.tsv" % step)
      with tf.gfile.GFile(dev_pred_file, "w") as writer:
        tf.logging.info("***** Dev Predict results *****")
        for (i, prediction) in enumerate(dev_result):
          predictions = prediction["predictions"]
          output_line = str(predictions) + "\n"
          writer.write(output_line)

      test_result = estimator.evaluate(input_fn=test_input_fn, steps=None, name="test")
      test_metric_file = os.path.join(FLAGS.output_dir, "test_metric-%d.txt" % step)
      with tf.gfile.GFile(test_metric_file, "w") as writer:
        tf.logging.info("***** Test results *****")
        # tf.summary.scalar("test/rmse", test_result["eval_rmse"])
        # tf.summary.scalar("test/mae", test_result["eval_mae"])
        # tf.summary.scalar("test/pearson", test_result["eval_pearson"])
        for key in sorted(test_result.keys()):
          tf.logging.info("  %s = %s", key, str(test_result[key]))
          writer.write("%s = %s\n" % (key, str(test_result[key])))
      # predict没有steps
      test_result = estimator.predict(input_fn=test_input_fn)
      test_pred_file = os.path.join(FLAGS.output_dir, "test_pred-%d.tsv" % step)
      with tf.gfile.GFile(test_pred_file, "w") as writer:
        tf.logging.info("***** Test Predict results *****")
        for (i, prediction) in enumerate(test_result):
          predictions = prediction["predictions"]
          output_line = str(predictions) + "\n"
          writer.write(output_line)

      if pearson < max_pearson:
        cnt += 1
      else:
        max_pearson = pearson
        cnt = 0
      tf.logging.info("***** Early Stopping *****")
      tf.logging.info("%d checkpoints lower than best result" % cnt)
      if cnt >= FLAGS.early_stopping:
        tf.logging.info("Performing early stopping...")
        break
      '''


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
