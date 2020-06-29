import tensorflow as tf

class Context_Encoder(object):

    def __init__(self, height, width, channel, z_dim, learning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.k_size, self.z_dim = 3, z_dim
        self.learning_rate = learning_rate

        self.x = tf.compat.v1.placeholder(tf.float32, \
            shape=[None, self.height, self.width, self.channel], name="x")
        self.m = tf.compat.v1.placeholder(tf.float32, \
            shape=[None, self.height, self.width, self.channel], name="z")
        self.batch_size = tf.placeholder(tf.int32, \
            shape=[], name="batch_size")

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []
        self.fc_shapes, self.conv_shapes = [], []

        self.drop = (tf.ones_like(self.m) - self.m) * self.x
        self.x_hat, self.d_real, self.d_fake = \
            self.build_model(input=self.x, mask=self.drop, ksize=self.k_size)

        self.loss_rec_b = tf.compat.v1.reduce_sum((self.m * (self.x - self.x_hat))**2, axis=(1, 2, 3))
        self.loss_adv_b = -tf.compat.v1.reduce_sum(tf.math.log(self.d_real + 1e-12) + tf.math.log(tf.ones_like(self.d_fake) - self.d_fake + 1e-12), axis=1)

        self.loss_tot = tf.compat.v1.reduce_mean(self.loss_rec_b + self.loss_adv_b)
        self.loss_rec = tf.compat.v1.reduce_mean(self.loss_rec_b)
        self.loss_adv = tf.compat.v1.reduce_mean(self.loss_adv_b)

        #default: beta1=0.9, beta2=0.999
        self.optimizer = tf.compat.v1.train.AdamOptimizer( \
            self.learning_rate, beta1=0.9, beta2=0.999).minimize(self.loss_tot)

        self.mse_r = self.mean_square_error(x1=self.x, x2=self.x_hat)

        tf.compat.v1.summary.scalar('loss_rec', self.loss_rec)
        tf.compat.v1.summary.scalar('loss_adv', self.loss_adv)
        tf.compat.v1.summary.scalar('total loss', self.loss_tot)
        self.summaries = tf.compat.v1.summary.merge_all()

    def mean_square_error(self, x1, x2):

        data_dim = len(x1.shape)
        if(data_dim == 4):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2, 3))
        elif(data_dim == 3):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1, 2))
        elif(data_dim == 2):
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2), axis=(1))
        else:
            return tf.compat.v1.reduce_sum(tf.square(x1 - x2))

    def build_model(self, input, mask, ksize=3):

        with tf.name_scope('encoder') as scope_enc:
            latent = self.encoder(input=mask, ksize=ksize)

        with tf.name_scope('chan_wise_fc') as scope_cw:
            cw_fc = self.channel_wise_fully_connected(input=latent)

        with tf.name_scope('decoder') as scope_dec:
            x_hat = self.decoder(input=cw_fc, ksize=ksize)

        with tf.name_scope('discriminator') as scope_dec:
            d_real = self.discriminator(input=input, ksize=ksize)
            d_fake = self.discriminator(input=x_hat, ksize=ksize)

        return x_hat, d_real, d_fake

    def encoder(self, input, ksize=3):

        print("Encode-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, self.channel, 16], activation="elu", name="conv1_1")
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="conv1_2")
        maxp1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="max_pool1")
        self.conv_shapes.append(conv1_2.shape)

        print("Encode-2")
        conv2_1 = self.conv2d(input=maxp1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 32], activation="elu", name="conv2_1")
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="conv2_2")
        maxp2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="max_pool2")
        self.conv_shapes.append(conv2_2.shape)

        print("Encode-3")
        conv3_1 = self.conv2d(input=maxp2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 64], activation="elu", name="conv3_1")
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="conv3_2")
        maxp3 = self.maxpool(input=conv3_2, ksize=2, strides=2, padding='SAME', name="max_pool3")
        self.conv_shapes.append(conv3_2.shape)

        print("Encode-4")
        conv4_1 = self.conv2d(input=maxp3, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 128], activation="elu", name="conv4_1")
        conv4_2 = self.conv2d(input=conv4_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 128], activation="elu", name="conv4_2")
        maxp4 = self.maxpool(input=conv4_2, ksize=2, strides=2, padding='SAME', name="max_pool4")
        self.conv_shapes.append(conv4_2.shape)

        print("Encode-5")
        conv5_1 = self.conv2d(input=maxp4, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 128], activation="elu", name="conv5_1")
        conv5_2 = self.conv2d(input=conv5_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 128], activation="elu", name="conv5_2")
        maxp5 = self.maxpool(input=conv5_2, ksize=2, strides=2, padding='SAME', name="max_pool5")
        self.conv_shapes.append(conv5_2.shape)

        print("Encode-6")
        conv6_1 = self.conv2d(input=maxp5, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 256], activation="elu", name="conv6_1")
        conv6_2 = self.conv2d(input=conv6_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 256, 256], activation="elu", name="conv6_2")
        self.conv_shapes.append(conv6_2.shape)

        self.fc_shapes.append(conv6_2.shape)
        latent = conv6_2
        return latent

    def channel_wise_fully_connected(self, input):

        output = []
        for channel in range(256):
            [n, h, w, c] = self.fc_shapes[0]
            tmp_feat = input[:, :, :, channel]
            fulcon_in = tf.compat.v1.reshape(tmp_feat, shape=[self.batch_size, h*w], name="cwfc_flat_%d" %(channel))
            fulcon_out = self.fully_connected(input=fulcon_in, num_inputs=h*w, \
                num_outputs=h*w, activation="elu", name="cwfc_%d" %(channel))
            output.append(tf.compat.v1.reshape(fulcon_out, shape=[self.batch_size, h, w], name="cwfc_out_%d" %(channel)))

        output = tf.transpose(output, perm=(1, 2, 3, 0))
        return output

    def decoder(self, input, ksize=3):

        print("Decode-1")
        convt1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 256, 256], activation="elu", name="convt1_1")
        convt1_2 = self.conv2d(input=convt1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 256, 256], activation="elu", name="convt1_2")

        print("Decode-2")
        [n, h, w, c] = self.conv_shapes[-2]
        convt2_1 = self.conv2d_transpose(input=convt1_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 128, 256], \
            dilations=[1, 1, 1, 1], activation="elu", name="convt2_1")
        convt2_2 = self.conv2d(input=convt2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 128], activation="elu", name="convt2_2")

        print("Decode-2")
        [n, h, w, c] = self.conv_shapes[-3]
        convt3_1 = self.conv2d_transpose(input=convt2_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 128, 128], \
            dilations=[1, 1, 1, 1], activation="elu", name="convt3_1")
        convt3_2 = self.conv2d(input=convt3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 128], activation="elu", name="convt3_2")

        print("Decode-4")
        [n, h, w, c] = self.conv_shapes[-4]
        convt4_1 = self.conv2d_transpose(input=convt3_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 64, 128], \
            dilations=[1, 1, 1, 1], activation="elu", name="convt4_1")
        convt4_2 = self.conv2d(input=convt4_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="convt4_2")

        print("Decode-5")
        [n, h, w, c] = self.conv_shapes[-5]
        convt5_1 = self.conv2d_transpose(input=convt4_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 32, 64], \
            dilations=[1, 1, 1, 1], activation="elu", name="convt5_1")
        convt5_2 = self.conv2d(input=convt5_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="convt5_2")

        print("Decode-6")
        [n, h, w, c] = self.conv_shapes[-6]
        convt6_1 = self.conv2d_transpose(input=convt5_2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, 16, 32], \
            dilations=[1, 1, 1, 1], activation="elu", name="convt6_1")
        convt6_2 = self.conv2d(input=convt6_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="convt6_2")
        convt6_3 = self.conv2d(input=convt6_2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, self.channel], activation="sigmoid", name="convt6_3")

        return convt6_3

    def discriminator(self, input, ksize=3):

        print("Discriminator-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, self.channel, 16], activation="elu", name="dis_conv1_1")
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="dis_conv1_2")
        maxp1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="dis_max_pool1")

        print("Discriminator-2")
        conv2_1 = self.conv2d(input=maxp1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 32], activation="elu", name="dis_conv2_1")
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="dis_conv2_2")
        maxp2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="dis_max_pool2")

        print("Discriminator-3")
        conv3_1 = self.conv2d(input=maxp2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 64], activation="elu", name="dis_conv3_1")
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="dis_conv3_2")
        maxp3 = self.maxpool(input=conv3_2, ksize=2, strides=2, padding='SAME', name="dis_max_pool3")

        print("Discriminator-4")
        conv4_1 = self.conv2d(input=maxp3, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 128], activation="elu", name="dis_conv4_1")
        conv4_2 = self.conv2d(input=conv4_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 128], activation="elu", name="dis_conv4_2")
        maxp4 = self.maxpool(input=conv4_2, ksize=2, strides=2, padding='SAME', name="dis_max_pool4")

        print("Discriminator-5")
        conv5_1 = self.conv2d(input=maxp4, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 128], activation="elu", name="dis_conv5_1")
        conv5_2 = self.conv2d(input=conv5_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 128], activation="elu", name="dis_conv5_2")
        maxp5 = self.maxpool(input=conv5_2, ksize=2, strides=2, padding='SAME', name="dis_max_pool5")

        print("Discriminator-6")
        conv6_1 = self.conv2d(input=maxp5, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 256], activation="elu", name="dis_conv6_1")
        conv6_2 = self.conv2d(input=conv6_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 256, 256], activation="elu", name="dis_conv6_2")

        [n, h, w, c] = conv6_2.shape
        fulcon_in = tf.compat.v1.reshape(conv6_2, shape=[self.batch_size, h*w*c], name="dis_cwfc_flat")
        fulcon_out = self.fully_connected(input=fulcon_in, num_inputs=h*w*c, \
            num_outputs=1, activation="sigmoid", name="disfc")

        return fulcon_out

    def initializer(self):
        return tf.compat.v1.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def maxpool(self, input, ksize, strides, padding, name=""):

        out_maxp = tf.compat.v1.nn.max_pool(value=input, \
            ksize=ksize, strides=strides, padding=padding, name=name)
        print("Max-Pool", input.shape, "->", out_maxp.shape)

        return out_maxp

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" == activation):
            out = tf.compat.v1.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            out = tf.compat.v1.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" == activation):
            out = tf.compat.v1.nn.relu(input, name='%s_relu' %(name))
        elif("lrelu" == activation):
            out = tf.compat.v1.nn.leaky_relu(input, name='%s_lrelu' %(name))
        elif("elu" == activation):
            out = tf.compat.v1.nn.elu(input, name='%s_elu' %(name))
        else: out = input

        return out

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        try:
            var_idx = name_bank.index(name)
        except:
            variable = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def conv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_inputs, num_outputs]
        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-1]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def conv2d_transpose(self, input, stride, padding, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_outputs, num_inputs]
        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-2]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d_transpose(
            value=input,
            filter=weight,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv-Tr", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=[num_inputs, num_outputs], name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[num_outputs], name='%s_b' %(name))

        out_mul = tf.compat.v1.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))

        print("Full-Con", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)
