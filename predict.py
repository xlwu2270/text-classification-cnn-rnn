# coding: utf-8

from __future__ import print_function

import os
import datetime
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.proj4_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data\\Corpus_15Cats\\corpus.vocab'
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        with tf.device("/gpu:0"):
            # 原始的概率分布情况
            y_pred = self.session.run(tf.nn.softmax(self.model.logits), feed_dict=feed_dict)
            # 经过 argmax 得出的序列号
            y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)

        return y_pred, self.categories[y_pred_cls[0]]

    def predictBatch(self, messages):
        dataList = []
        for i in range(len(messages)):
            dataList.append([self.word_to_id[x] for x in messages[i] if x in self.word_to_id])

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences(dataList, self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        with tf.device("/gpu:0"):
            # 原始的概率分布情况
            y_pred = self.session.run(tf.nn.softmax(self.model.logits), feed_dict=feed_dict)
            # 经过 argmax 得出的序列号
            y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)

            y_cats = [self.categories[cat] for cat in y_pred_cls]
        return y_pred, y_cats


if __name__ == '__main__':
    cnn_model = CnnModel()
    start = datetime.datetime.now()
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00',
                 '#家属称闷死老人保姆作案前认真热情# 5月12日，江苏溧阳警方通报保姆闷死老人事件。目前，保姆已被刑拘，案件正进一步侦办中。']
    # with tf.device("/gpu:0"):
    print(cnn_model.predictBatch(test_demo))
        # for cont in test_demo:
            # print(cnn_model.predict(cont))
    end = datetime.datetime.now()
    print(start)
    print("Start Time: %s\nEnd Time: %s\nPrediction During Time: %sms" % (start, end, (end - start).total_seconds()))
    print("Prediction During Time: %.3ss" % (end - start).total_seconds())

