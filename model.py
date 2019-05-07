from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Input, Concatenate, Lambda
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU, Dropout
from keras.layers import Reshape, LeakyReLU, ZeroPadding2D
from keras.layers import Conv1D, Add, Conv2D, UpSampling2D
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
import keras
from keras.optimizers import Adam
from keras.backend import tf as ktf
from config import cfg
from keras.losses import categorical_crossentropy, binary_crossentropy


class GLU(Layer):
    def __init__(self):
        super(GLU, self).__init__()

    def build(
            self,
            input_shape,
    ):
        assert input_shape[-1] % 2 == 0, 'channels dont divide 2!'
        self.output_dim = input_shape[1:-1] + (int(input_shape[-1] / 2), )
        super(GLU, self).build(input_shape)

    def call(self, x):
        nc = int(self.output_dim[-1])
        if K.ndim(x) == 4:
            return x[:, :, :, :nc] * K.sigmoid(x[:, :, :, nc:])
        if K.ndim(x) == 3:
            return x[:, :, :nc] * K.sigmoid(x[:, :, nc:])
        if K.ndim(x) == 2:
            return x[:, :nc] * K.sigmoid(x[:, nc:])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.output_dim)


def conv1x1(out_planes, bias=False):
    #1x1 convolution with padding
    return Conv2D(
        out_planes, kernel_size=1, strides=1, padding='valid', use_bias=bias)


def conv3x3(out_planes):
    #3x3 convolution with padding
    return Conv2D(
        out_planes, kernel_size=3, strides=1, padding="same", use_bias=False)


def upBlock(out_planes):
    #x2 upsampling
    block = Sequential([
        UpSampling2D(size=2),
        conv3x3(out_planes * 2),
        BatchNormalization(epsilon=1e-05),
        GLU()
    ])
    return block


def Block3x3_relu(out_planes):
    block = Sequential(
        [conv3x3(out_planes * 2),
         BatchNormalization(epsilon=1e-05),
         GLU()])
    return block


def ResBlock(channel_num, input_tensor):
    z = conv3x3(channel_num * 2)(input_tensor)
    z = BatchNormalization(epsilon=1e-05)(z)
    z = GLU()(z)
    z = conv3x3(channel_num)(z)
    z = BatchNormalization(epsilon=1e-05)(z)
    out = Add()([z, input_tensor])
    return out


# ############## Text2Image Encoder ##############
def RNN_ENCODER(emb_size,
                hidden_size,
                vocab_size,
                drop_prob=0.5,
                rec_unit='gru'):
    """
    :param embed_size: size of word embeddings
    :param hidden_size: size of hidden state of the recurrent unit
    :param vocab_size: size of the vocabulary (output of the network)
    :param rec_unit: type of recurrent unit (default=gru)
    """
    __rec_units = {'gru': GRU, 'lstm': LSTM}
    assert rec_unit in __rec_units, 'Specified recurrent unit is not available'
    c_dim = cfg.GAN.CONDITION_DIM

    cap_input = Input(shape=(cfg.TEXT.WORDS_NUM, ), name="caption_input")
    embeddings = Embedding(vocab_size, emb_size)(cap_input)
    drop = Dropout(drop_prob)(embeddings)
    if rec_unit == 'gru':
        words_emb, forward_h, backward_h = Bidirectional(__rec_units[rec_unit](
            hidden_size, return_sequences=True, return_state=True))(drop)
    else:  #LSTM
        words_emb, forward_h, forward_c, backward_h, backward_c = Bidirectional(
            __rec_units[rec_unit](
                hidden_size, return_sequences=True, return_state=True))(drop)
    sent_emb = Concatenate()([forward_h, backward_h])
    c_code, eps_input = CA_NET(sent_emb, c_dim)
    RNN_model = Model(
        [cap_input, eps_input], [words_emb, sent_emb, c_code],name="RNN_ENCODER")
    return RNN_model, words_emb, sent_emb, c_code


# ############## Image2text Encoder-Decoder #######
def CNN_ENCODER_RNN_DECODER(emb_size, hidden_size, vocab_size, rec_unit='gru'):
    """
    :param embed_size: size of word embeddings
    :param hidden_size: size of hidden state of the recurrent unit
    :param vocab_size: size of the vocabulary (output of the network)
    :param rec_unit: type of recurrent unit (default=gru)
    """
    __rec_units = {
        'gru': GRU,
        'lstm': LSTM,
        'cugru': CuDNNGRU,
        'culstm': CuDNNLSTM
    }
    assert rec_unit in __rec_units, 'Specified recurrent unit is not available'

    pic_input = Input(shape=(None,None,3,), name="cr_pic_input")
    upsamp = Lambda(lambda image: ktf.image.resize_images(image, (299, 299)))(
        pic_input)
    inc = InceptionV3(include_top=False, pooling='avg')
    inc.trainable = False  #凍結
    cnn_code = inc(upsamp)
    #cnn_code = keras.layers.GlobalMaxPooling2D()(upsamp)
    cnn_code = Dense(emb_size)(cnn_code)
    cnn_code = Reshape((1, emb_size))(cnn_code)

    cap_input = Input(shape=(cfg.TEXT.WORDS_NUM, ), name="cr_cap_input")
    #cap_input = Input(shape=(None,))
    embeddings = Embedding(vocab_size, emb_size)(cap_input)
    concat = Concatenate(axis=1)([cnn_code, embeddings])
    #rnn = GRU(hidden_size, return_sequences=True)(concat)
    rnn = __rec_units[rec_unit](
        hidden_size, return_sequences=True, recurrent_dropout=0.3)(concat)
    out = Dense(vocab_size, activation='softmax', name='CR_out_layre')(rnn)
    model = Model([pic_input, cap_input], out, name="CNN_RNN_DEC")
    return model


################ Attntion ################
def ATT_NET(target, context, mask, use_mask):
    idf = int(target.shape[-1])
    source = Conv1D(idf, kernel_size=1)(context)
    weightedContext, attn = GlobalAttentionGeneral(use_mask)(
        [target, source, mask])
    return weightedContext, attn


class GlobalAttentionGeneral(Layer):
    def __init__(self, use_mask):
        self.use_mask = ktf.constant(use_mask, dtype=ktf.bool)  #True or False
        super(GlobalAttentionGeneral, self).__init__()

    def build(self, input_shape):
        self.output_dim = input_shape[0][-1]
        self.ih = input_shape[0][1]
        self.iw = input_shape[0][2]
        self.queryL = self.ih * self.iw
        self.sourceL = input_shape[1][1]
        super(GlobalAttentionGeneral, self).build(input_shape)

    def call(self, input_tensor):
        """
        target: batch x  ih x iw (queryL=ihxiw) x idf
        source: batch x sourceL(seq_len) x idf
        mask: batch x sourceL 
            -inf or 0
        """
        target, source, mask = input_tensor
        idf = self.output_dim
        ih = self.ih
        iw = self.iw
        queryL = self.queryL
        sourceL = self.sourceL

        # --> batch x queryL x idf
        target = K.reshape(target, (-1, queryL, idf))
        # --> batch x idf x sourceL
        sourceT = ktf.transpose(source, perm=[0, 2, 1])
        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = ktf.matmul(target, sourceT)
        addmask = K.switch(
            self.use_mask,
            lambda: K.repeat_elements(
                K.reshape(mask, (-1, 1, sourceL)), rep=queryL, axis=1),
            lambda: 0.0)
        attn = attn + addmask
        attn = K.softmax(attn)
        # (batch x queryL x sourceL)(batch x sourceL x idf)
        # --> batch x queryL x idf
        weightedContext = ktf.matmul(attn, source)
        weightedContext = K.reshape(weightedContext, (-1, ih, iw, idf))
        attn = K.reshape(attn, (-1, ih, iw, sourceL))
        return [weightedContext, attn]

    def compute_output_shape(self, input_shape):
        return [input_shape[0], (None, self.ih, self.iw, self.sourceL)]


# ############## G networks ###################
def CA_NET(sent_emb, c_dim):
    eps_input = Input(shape=(c_dim,), name='eps_input')
    z = Dense(c_dim * 4)(sent_emb)
    z = GLU()(z)
    mu = Lambda(lambda x: x[:, :c_dim], output_shape=(c_dim,))(z)
    logvar = Lambda(lambda x: x[:, c_dim:], output_shape=(c_dim, ))(z)
    c_code = Lambda(
        canet_function, name="canet_function",
        output_shape=(c_dim, ))([mu, logvar, eps_input])
    return c_code, eps_input


def canet_function(input_tensor):
    mu, logvar, eps = input_tensor
    std = K.exp(logvar * 0.5)
    c_code = eps * std + mu
    return c_code


def INIT_STAGE_G(c_code, ngf):
    """
    :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
    """
    z_dim = cfg.GAN.Z_DIM

    z_code_input = Input(shape=(z_dim,), name="z_code_input_initG")

    c_z_code = Concatenate()([z_code_input, c_code])
    # state size 4 x 4 x ngf
    out_code = Dense(ngf * 4 * 4 * 2, use_bias=False)(c_z_code)
    out_code = BatchNormalization(epsilon=1e-05)(out_code)
    out_code = GLU()(out_code)
    out_code = Reshape((4, 4, ngf))(out_code)
    # state size 8 x 8 x ngf/3
    out_code = upBlock(ngf // 2)(out_code)
    # state size 16 x 16 x ngf/4
    out_code = upBlock(ngf // 4)(out_code)
    # state size 32 x 32 x ngf/8
    out_code32 = upBlock(ngf // 8)(out_code)
    # state size  64 x 64 x ngf/16
    out_code64 = upBlock(ngf // 16)(out_code32)
    return out_code64, z_code_input


def NEXT_STAGE_G(h_code, word_emb, c_code, mask_input, ngf):
    """
    h_code(query):  batch x ih x iw (queryL=ihxiw) x idf
    word_embs(context): batch x sourceL (sourceL=seq_len) x cdf
    """
    c_code_unsq = Reshape((1, -1))(c_code)
    cm_code, attm = ATT_NET(h_code, word_emb, mask_input, use_mask=True)
    cs_code, atts = ATT_NET(h_code, c_code_unsq, mask_input, use_mask=False)
    h_c_code = Concatenate()([h_code, cm_code, cs_code])

    layers = []
    for i in range(cfg.GAN.R_NUM):  #==2
        h_c_code = ResBlock(ngf * 3, h_c_code)
    out_code = upBlock(ngf)(h_c_code)
    return out_code, attm, atts


def G_DCGAN(sent_emb, word_emb, c_code):
    ngf = cfg.GAN.GF_DIM  #128
    c_dim = cfg.GAN.CONDITION_DIM
    attm, atts = None, None

    mask_input = Input(shape=(cfg.TEXT.WORDS_NUM, ), name='mask_input')
    netGsout_code = []
    if cfg.TREE.BRANCH_NUM > 0:
        out_code, z_code_input = INIT_STAGE_G(c_code, ngf * 16)
        netGsout_code += [out_code]
    if cfg.TREE.BRANCH_NUM > 1:
        out_code, attm, atts = NEXT_STAGE_G(out_code, word_emb, c_code,
                                            mask_input, ngf)
        netGsout_code += [out_code]
    if cfg.TREE.BRANCH_NUM > 2:
        out_code, attm, atts = NEXT_STAGE_G(out_code, word_emb, c_code,
                                            mask_input, ngf)
        netGsout_code += [out_code]
    out_image_block = GET_IMAGE_G(out_code)
    return netGsout_code, out_image_block, attm, atts, z_code_input, mask_input


def GET_IMAGE_G(out_code):
    out_image_block = Sequential([conv3x3(3), Activation('tanh')],
                                 name='GET_IMAGE_G{}'.format(
                                     int(out_code.shape[2])))
    return out_image_block


# ############## D networks ###################
def Block3x3_leakRelu(out_planes):
    block = Sequential([
        conv3x3(out_planes),
        BatchNormalization(epsilon=1e-05),
        LeakyReLU(alpha=0.2)
    ])
    return block


def downBlock(out_planes):
    # Downsale the spatial size by a factor of 2
    block = Sequential([
        ZeroPadding2D(padding=1),
        Conv2D(out_planes, kernel_size=4, strides=2, use_bias=False),
        BatchNormalization(epsilon=1e-05),
        LeakyReLU(alpha=0.2)
    ])
    return block


def encode_image_by_16times(ndf):
    # Downsale the spatial size by a factor of 16
    encode_img = Sequential([
        # --> state size. ndf x in_size/2 x in_size/2
        ZeroPadding2D(padding=1),
        Conv2D(ndf, kernel_size=4, strides=2, use_bias=False),
        LeakyReLU(alpha=0.2),
        # --> state size 2ndf x x in_size/4 x in_size/4
        ZeroPadding2D(padding=1),
        Conv2D(ndf * 2, kernel_size=4, strides=2, use_bias=False),
        LeakyReLU(alpha=0.2),
        # --> state size 4ndf x in_size/8 x in_size/8
        ZeroPadding2D(padding=1),
        Conv2D(ndf * 4, kernel_size=4, strides=2, use_bias=False),
        BatchNormalization(epsilon=1e-05),
        LeakyReLU(alpha=0.2),
        # --> state size 8ndf x in_size/16 x in_size/16
        ZeroPadding2D(padding=1),
        Conv2D(ndf * 8, kernel_size=4, strides=2, use_bias=False),
        BatchNormalization(epsilon=1e-05),
        LeakyReLU(alpha=0.2)
    ])
    return encode_img


def D_GET_LOGITS(ndf, nef, h_code, sent_emb):
    # conditioning output
    s_code = Reshape((1, 1, -1))(sent_emb)
    #s_code = Lambda(lambda x: x.repeat(1, 4, 4, 1))(s_code)
    s_code = Lambda(lambda x: K.repeat_elements(x, 4, axis=2))(s_code)
    s_code = Lambda(lambda x: K.repeat_elements(x, 4, axis=1))(s_code)
    #state size 4 x 4 x (ngf+egf)
    h_c_code = Concatenate()([h_code, s_code])
    #state size in_size x in_size x ngf
    h_c_code = Block3x3_leakRelu(ndf * 8)(h_c_code)
    conv2d_lg = Conv2D(1, kernel_size=4, strides=4)
    h_logits = conv2d_lg(h_code)
    h_c_logits = conv2d_lg(h_c_code)
    h_logits = Activation('sigmoid')(h_logits)
    h_c_logits = Activation('sigmoid')(h_c_logits)
    h_logits = Reshape((-1, ), name="D_h_out_layre")(h_logits)
    h_c_logits = Reshape((-1, ), name="D_hc_out_layre")(h_c_logits)
    return h_logits, h_c_logits


# For 64 x 64 images
def D_NET64(sent_emb):
    ndf = cfg.GAN.DF_DIM
    nef = cfg.TEXT.EMBEDDING_DIM
    D_pic_input = Input((64, 64, 3), name="D_pic_input64")
    x_code4 = encode_image_by_16times(ndf)(D_pic_input)  # 4 x 4 x 8df
    h_logits, h_c_logits = D_GET_LOGITS(ndf, nef, x_code4, sent_emb)
    return h_logits, h_c_logits, D_pic_input


# For 128 x 128 images
def D_NET128(sent_emb):
    ndf = cfg.GAN.DF_DIM
    nef = cfg.TEXT.EMBEDDING_DIM
    D_pic_input = Input((128, 128, 3), name="D_pic_input128")
    x_code8 = encode_image_by_16times(ndf)(D_pic_input)  # 8 x 8 x 8df
    x_code4 = downBlock(ndf * 16)(x_code8)  # 4 x 4 x 16df
    x_code4 = Block3x3_leakRelu(ndf * 8)(x_code4)  # 4 x 4 x 8df
    h_logits, h_c_logits = D_GET_LOGITS(ndf, nef, x_code4, sent_emb)
    return h_logits, h_c_logits, D_pic_input


# For 256 x 256 images
def D_NET256(sent_emb):
    ndf = cfg.GAN.DF_DIM
    nef = cfg.TEXT.EMBEDDING_DIM
    D_pic_input = Input((256, 256, 3), name="D_pic_input256")
    x_code16 = encode_image_by_16times(ndf)(D_pic_input)  # 16 x 16 x 8df
    x_code8 = downBlock(ndf * 16)(x_code16)  # 8 x 8 x 16df
    x_code4 = downBlock(ndf * 32)(x_code8)  # 4 x 4 x 32df
    x_code4 = Block3x3_leakRelu(ndf * 16)(x_code4)  # 4 x 4 x 16df
    x_code4 = Block3x3_leakRelu(ndf * 8)(x_code4)  # 4 x 4 x 8df
    h_logits, h_c_logits = D_GET_LOGITS(ndf, nef, x_code4, sent_emb)
    return h_logits, h_c_logits, D_pic_input