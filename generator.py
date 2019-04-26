import keras.backend as K
import numpy as np
from config import cfg
from dataset import TextDataset


class DataGenerator(object):
    def __init__(self, dataset, batchsize, eos=False):
        self.dataset = dataset
        self.imsize = dataset.imsize
        self.batchsize = batchsize
        self.n_words = dataset.n_words
        self.count = 0
        self.length = len(dataset)
        self.maxcount = int(self.length / self.batchsize)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        img64_ar = np.empty((0, self.imsize[0], self.imsize[0], 3))
        if cfg.TREE.BRANCH_NUM > 1:
            img128_ar = np.empty((0, self.imsize[1], self.imsize[1], 3))
        if cfg.TREE.BRANCH_NUM > 2:
            img256_ar = np.empty((0, self.imsize[2], self.imsize[2], 3))

        captions_ar = np.empty((0, cfg.TEXT.WORDS_NUM, 1))
        captions_ar_prezeropad = np.zeros((
            self.batchsize,
            cfg.TEXT.WORDS_NUM,
        ))
        keys_list = []
        for i in range(self.batchsize):
            imgs, captions, cap_lens, class_ids, keys = self.dataset[
                self.count]
            self.count += 1
            if self.count == self.maxcount:
                self.count = 0

            img64_ar = np.vstack((img64_ar, imgs[0][np.newaxis, :]))
            if cfg.TREE.BRANCH_NUM > 1:
                img128_ar = np.vstack((img128_ar, imgs[1][np.newaxis, :]))
            if cfg.TREE.BRANCH_NUM > 2:
                img256_ar = np.vstack((img256_ar, imgs[2][np.newaxis, :]))
            captions_ar = np.vstack((captions_ar, captions[np.newaxis, :]))
            zero_count = int(sum(captions == 0))
            captions_ar_prezeropad[i, zero_count:] = captions.reshape(
                -1)[:cfg.TEXT.WORDS_NUM - zero_count]
            keys_list += [keys]

        y_captions_ar = np.concatenate([
            captions_ar,
            np.zeros((captions_ar.shape[0], 1, captions_ar.shape[2]))
        ],
                                       axis=1)

        captions_ar = np.squeeze(captions_ar, axis=2).astype("f")
        captions_ar_prezeropad = captions_ar_prezeropad.astype("f")
        mask = (captions_ar == 0)
        #ノイズ
        z_code = np.random.normal(0, 1,
                                  (self.batchsize, cfg.GAN.Z_DIM)).astype("f")

        #gan_label
        real_label = np.ones((self.batchsize, 1), dtype="i4")
        fake_label = np.zeros((self.batchsize, 1), dtype="i4")
        #ワンホット
        captions_label = np.zeros(
            (y_captions_ar.shape[0], y_captions_ar.shape[1],
             self.n_words)).astype("i4")
        for j in range(y_captions_ar.shape[0]):
            for i in range(y_captions_ar.shape[1]):
                ind = y_captions_ar[j, i, 0].astype("int")
                captions_label[j, i, ind] = 1
        image_list = []
        if cfg.TREE.BRANCH_NUM > 0:
            image_list += [img64_ar]
        if cfg.TREE.BRANCH_NUM > 1:
            image_list += [img128_ar]
        if cfg.TREE.BRANCH_NUM > 2:
            image_list += [img256_ar]
        
        return image_list, captions_ar, captions_ar_prezeropad, \
                    z_code, mask, keys_list, captions_label, real_label, fake_label


class DataGenerator_encode(object):
    def __init__(self, dataset, batchsize=20):
        self.dataset = dataset
        self.imsize = dataset.imsize
        self.batchsize = batchsize
        self.n_words = dataset.n_words
        self.count = 0
        self.length = len(dataset)
        self.maxcount = int(self.length / self.batchsize)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        img256_ar = np.empty((0, self.imsize[2], self.imsize[2], 3))
        captions_ar = np.empty((0, cfg.TEXT.WORDS_NUM, 1))
        for _ in range(self.batchsize):
            imgs, captions, cap_lens, class_ids, keys = self.dataset[
                self.count]
            self.count += 1
            if self.count == self.maxcount:
                self.count = 0
            img256_ar = np.vstack((img256_ar, imgs[2][np.newaxis, :]))
            captions_ar = np.vstack((captions_ar, captions[np.newaxis, :]))

        img256_ar = img256_ar.astype("f")
        y_captions_ar = np.concatenate([
            captions_ar,
            np.zeros((captions_ar.shape[0], 1, captions_ar.shape[2]))
        ],
                                       axis=1)
        captions_ar = np.squeeze(captions_ar, axis=2).astype("f")
        #ワンホット
        y = np.zeros((y_captions_ar.shape[0], y_captions_ar.shape[1],
                      self.n_words)).astype("i4")
        for j in range(y_captions_ar.shape[0]):
            for i in range(y_captions_ar.shape[1]):
                ind = y_captions_ar[j, i, 0].astype("int")
                y[j, i, ind] = 1
        return [img256_ar, captions_ar], y

