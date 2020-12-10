import tensorflow as tf
import os
from opts import *
import Util
import numpy as np
import scipy.sparse
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from sklearn.metrics import homogeneity_score as hs
from sklearn.metrics.cluster import completeness_score as cs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy import *
import datetime
from tqdm import tqdm

np.random.seed(0)
tf.set_random_seed(0)


class Test_scRAE(object):
    def __init__(self, sess, epoch=200, ae_lr=0.001, gan_lr=0.0001, batch_size=128, X_dim=720, z_dim=10,
                 dataset_name=None, checkpoint_dir='checkpoint', sample_dir='samples', result_dir='result',
                 num_layers=2, g_h_dim=None, lg_h_dim=None, d_h_dim=None, gen_activation='sig', leak=0.2,
                 rate=0.1, trans='sparse', is_bn=False, g_iter=2, lam=10.0, sampler='uniform', perplexity=50):

        self.sess = sess
        self.epoch = epoch
        self.ae_lr = ae_lr
        self.gan_lr = gan_lr
        self.batch_size = batch_size
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.result_dir = result_dir
        self.num_layers = num_layers
        self.g_h_dim = g_h_dim  # Fully connected layers for Generator
        self.lg_h_dim = lg_h_dim  # Fully connected layers for Latent Generator
        self.d_h_dim = d_h_dim  # Fully connected layers for Discriminator
        self.gen_activation = gen_activation
        self.leak = leak
        self.rate = rate
        self.trans = trans
        self.is_bn = is_bn
        self.g_iter = g_iter
        self.lam = lam
        self.sampler = sampler
        self.perplexity = perplexity
        self._is_train = False
        self.lr_decay_val = np.asarray([1]).astype(np.float32)

        if self.dataset_name == '10x_73k' or self.dataset_name == '10x_68k' or self.dataset_name == 'Zeisel' \
                or self.dataset_name == 'Macosko' or self.dataset_name == 'Rosenberg':

            if self.trans == 'sparse':
                self.data_train, self.data_val, self.data_test, self.scale = \
                    Util.load_gene_mtx(self.dataset_name, transform=False, count=False, actv=self.gen_activation)
            else:
                self.data_train, self.data_val, self.data_test = Util.load_gene_mtx(self.dataset_name,
                                                                                    transform=True)
                self.scale = 1.0

        self.labels_train, self.labels_val, self.labels_test = Util.load_labels(self.dataset_name)

        self.train_size = self.data_train.shape[0]
        self.test_size = self.data_test.shape[0]
        self.total_size = self.train_size + self.test_size

        self.data = np.concatenate([self.data_train, self.data_test])

        print("Shape self.data_train:", self.data_train.shape)
        print("Shape self.data_test:", self.data_test.shape)

        self.build_model()

    def build_model(self):

        self.x_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Input')
        self.x_target = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.X_dim], name='Target')
        self.dropout_rate = tf.compat.v1.placeholder(dtype=tf.float32, name='dropout_rate')
        self.noise = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name='Noise')
        self.lr_decay = tf.compat.v1.placeholder(tf.float32, [1], name='lr-decay')

        self.training_phase = True
        self.n_layers = self.num_layers
        self.n_latent = self.z_dim

        self.z, self.l_post_m, self.l_post_v = self.encoder(self.x_input)

        log_library_size = np.log(np.sum(self.data_train, axis=1))
        mean, variance = np.mean(log_library_size), np.var(log_library_size)
        library_size_mean = mean
        library_size_variance = variance
        self.library_size_mean = tf.to_float(tf.constant(library_size_mean))
        self.library_size_variance = tf.to_float(tf.constant(library_size_variance))
        self.library = self.sample_gaussian(self.l_post_m, self.l_post_v)

        self.expression = self.x_input

        self.decoder_output = self.decoder(self.z)
        self.n_input = self.expression.get_shape().as_list()[1]

        self.x_post_scale = tf.nn.softmax(dense(self.decoder_output, self.g_h_dim[0], self.n_input,
                                                name='dec_x_post_scale'))
        self.x_post_r = tf.Variable(tf.random.normal([self.n_input]), name="dec_x_post_r")
        self.x_post_rate = tf.exp(self.library) * self.x_post_scale
        self.x_post_dropout = dense(self.decoder_output, self.g_h_dim[0], self.n_input, name='dec_x_post_dropout')

        local_dispersion = tf.exp(self.x_post_r)
        local_l_mean = self.library_size_mean
        local_l_variance = self.library_size_variance

        self.fake_distribution = self.latent_generator(self.noise, self.z_dim)
        self.dis_real_logit = self.discriminator(self.z, self.z_dim)
        self.dis_fake_logit = self.discriminator(self.fake_distribution, self.z_dim, reuse=True)

        # Reconstruction loss 
        recon_loss = self.zinb_model(self.expression, self.x_post_rate, local_dispersion, self.x_post_dropout)
        kl_gauss_l = 0.05 * tf.reduce_sum(- tf.log(self.l_post_v + 1e-8) \
                                          + self.l_post_v / local_l_variance \
                                          + tf.square(self.l_post_m - local_l_mean) / local_l_variance \
                                          + tf.log(local_l_variance + 1e-8) - 1, 1)

        # AE loss
        self.autoencoder_loss = - tf.reduce_mean(recon_loss) + tf.reduce_mean(kl_gauss_l)

        alpha = tf.random.uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
        )
        differences = self.z - self.fake_distribution
        interpolates = self.fake_distribution + (alpha * differences)
        gradients = tf.gradients(self.discriminator(interpolates, self.z_dim, reuse=True), [interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.math.reduce_mean((self.slopes - 1.) ** 2)
        self.dis_loss = tf.math.reduce_mean(self.dis_fake_logit) - tf.math.reduce_mean(self.dis_real_logit)
        self.dis_loss += self.lam * gradient_penalty

        self.generator_loss = tf.math.reduce_mean(self.dis_real_logit)
        self.latent_generator_loss = -tf.math.reduce_mean(self.dis_fake_logit)

        t_vars = tf.compat.v1.trainable_variables()
        self.ae_vars = [var for var in t_vars if ('enc_' in var.name or 'dec_' in var.name)]
        self.dis_vars = [var for var in t_vars if 'dis_' in var.name]
        self.gen_vars = [var for var in t_vars if 'enc_' in var.name]
        self.lat_gen_vars = [var for var in t_vars if 'lat_gen_' in var.name]

        self.saver = tf.compat.v1.train.Saver()

    def train_cluster(self):

        print('Cluster scRAE on DataSet {} ... '.format(self.dataset_name))

        autoencoder_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.ae_lr).minimize(self.autoencoder_loss * self.lr_decay, var_list=self.ae_vars)

        discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.gan_lr, beta1=0.0, beta2=0.9).minimize(self.dis_loss * self.lr_decay,
                                                                      var_list=self.dis_vars)

        generator_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.ae_lr * 1e-3, beta1=0.0, beta2=0.9).minimize(self.generator_loss * self.lr_decay,
                                                                            var_list=self.gen_vars)

        latent_generator_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.gan_lr, beta1=0.0, beta2=0.9).minimize(self.latent_generator_loss * self.lr_decay,
                                                                      var_list=self.lat_gen_vars)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        a_loss_epoch = []
        d_loss_epoch = []
        g_loss_epoch = []
        lg_loss_epoch = []
        nmi_epoch = []
        ami_epoch = []
        hs_epoch = []
        cs_epoch = []

        control = 5  # Discriminator D1 is updated five times for each Generator and Latent Generator update

        num_batch_iter = self.total_size // self.batch_size
        for ep in tqdm(range(self.epoch)):
            d_loss_curr = g_loss_curr = lg_loss_curr = a_loss_curr = np.inf
            self._is_train = True
            for it in range(num_batch_iter):

                batch_x = self.next_batch(self.data_train, self.train_size)
                batch_z_real_dist = self.sample_Z(self.batch_size, self.z_dim)

                _, a_loss_curr = self.sess.run([autoencoder_optimizer, self.autoencoder_loss],
                                               feed_dict={self.x_input: batch_x, self.x_target: batch_x,
                                                          self.dropout_rate: self.rate,
                                                          self.lr_decay: self.lr_decay_val})

                if it % control == 0:
                    _, g_loss_curr = self.sess.run([generator_optimizer, self.generator_loss],
                                                   feed_dict={self.x_input: batch_x, self.dropout_rate: self.rate,
                                                              self.lr_decay: self.lr_decay_val})

                    _, lg_loss_curr = self.sess.run([latent_generator_optimizer, self.latent_generator_loss],
                                                    feed_dict={
                                                        self.x_input: batch_x, self.noise: batch_z_real_dist,
                                                        self.dropout_rate: self.rate,
                                                        self.lr_decay: self.lr_decay_val})
                else:
                    _, d_loss_curr = self.sess.run([discriminator_optimizer, self.dis_loss],
                                                   feed_dict={
                                                       self.x_input: batch_x, self.noise: batch_z_real_dist,
                                                       self.dropout_rate: self.rate,
                                                       self.lr_decay: self.lr_decay_val})

            if ep % 10 == 0 and ep > 0 and ep <= 30:
                self.lr_decay_val = self.lr_decay_val / 2

            self._is_train = False
            a_loss_epoch.append(a_loss_curr)
            d_loss_epoch.append(d_loss_curr)
            g_loss_epoch.append(g_loss_curr)
            lg_loss_epoch.append(lg_loss_curr)
            nmi, ami, hs, cs, _ = self.compute_metrices()
            nmi_epoch.append(nmi)
            ami_epoch.append(ami)
            hs_epoch.append(hs)
            cs_epoch.append(cs)
            if ep == 0:
                best_nmi = nmi
            else:
                if nmi > best_nmi:
                    best_nmi = nmi

        print(f'NMI: {max(nmi_epoch)}, AMI: {max(ami_epoch)}, HS: {max(hs_epoch)}, '
              f'CS: {(max(cs_epoch))}')

        if not os.path.exists('./Res_scRAE'):
            os.makedirs('./Res_scRAE')
        out_file_name = './Res_scRAE/Metrics_{}.txt'.format(self.dataset_name)
        f = open(out_file_name, 'a')
        f.write('\n{}, NMI = {}, AMI = {}, Homogeneity = {}, Completeness = {}'.
                format(self.model_dir, max(nmi_epoch), max(ami_epoch), max(hs_epoch), max(cs_epoch)))
        f.close()

    # The autoencoder network
    def encoder(self, x, reuse=False):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """

        with tf.compat.v1.variable_scope('Encoder') as scope:
            if reuse:
                scope.reuse_variables()

            if self.is_bn:
                h = tf.layers.batch_normalization(
                    lrelu(dense(x, self.X_dim, self.g_h_dim[0], name='enc_h0_lin'), alpha=self.leak),
                    training=self._is_train, name='enc_bn0')

                for i in range(1, self.num_layers):
                    h = tf.layers.batch_normalization(
                        lrelu(dense(h, self.g_h_dim[i - 1], self.g_h_dim[i], name='enc_h' + str(i) + '_lin'),
                              alpha=self.leak),
                        training=self._is_train, name='enc_bn' + str(i))

                z = dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim,
                          name='enc_z' + str(self.num_layers) + '_lin')

                h = tf.nn.relu(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim,
                                     name='enc_h' + str(self.num_layers) + '_lin'))

                l_post_m = dense(h, self.z_dim, 1, name='enc_l_post_m' + str(self.num_layers) + '_lin')
                l_post_v = tf.exp(dense(h, self.z_dim, 1, name='enc_l_post_v' + str(self.num_layers) + '_lin'))

            else:

                h = tf.nn.dropout(lrelu(dense(x, self.X_dim, self.g_h_dim[0], name='enc_h0_lin'), alpha=self.leak),
                                  rate=self.dropout_rate)

                for i in range(1, self.num_layers):
                    h = tf.nn.dropout(lrelu(dense(h, self.g_h_dim[i - 1], self.g_h_dim[i],
                                                  name='enc_h' + str(i) + '_lin'),
                                            alpha=self.leak), rate=self.dropout_rate)

                z = dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim,
                          name='enc_z' + str(self.num_layers) + '_lin')

                h = tf.nn.relu(dense(h, self.g_h_dim[self.num_layers - 1], self.z_dim,
                                     name='enc_h' + str(self.num_layers) + '_lin'))

                l_post_m = dense(h, self.z_dim, 1, name='enc_l_post_m' + str(self.num_layers) + '_lin')
                l_post_v = tf.exp(dense(h, self.z_dim, 1, name='enc_l_post_v' + str(self.num_layers) + '_lin'))

        return z, l_post_m, l_post_v

    def decoder(self, z, reuse=False):
        """
        Decoder part of the autoencoder.
        :param z: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """

        with tf.compat.v1.variable_scope('Decoder') as scope:
            if reuse:
                scope.reuse_variables()

            if self.is_bn:
                h = tf.layers.batch_normalization(
                    lrelu(dense(z, self.z_dim, self.g_h_dim[self.num_layers - 1],
                                name='dec_h' + str(self.num_layers - 1) + '_lin'),
                          alpha=self.leak),
                    training=self._is_train, name='dec_bn' + str(self.num_layers - 1))

                for i in range(self.num_layers - 2, -1, -1):
                    h = tf.layers.batch_normalization(
                        lrelu(dense(h, self.g_h_dim[i + 1], self.g_h_dim[i], name='dec_h' + str(i) + '_lin'),
                              alpha=self.leak),
                        training=self._is_train, name='dec_bn' + str(i))
            else:
                h = tf.nn.dropout(lrelu(dense(z, self.z_dim, self.g_h_dim[self.num_layers - 1],
                                              name='dec_h' + str(self.num_layers - 1) + '_lin'),
                                        alpha=self.leak),
                                  rate=self.dropout_rate)
                for i in range(self.num_layers - 2, -1, -1):
                    h = tf.nn.dropout(
                        lrelu(dense(h, self.g_h_dim[i + 1], self.g_h_dim[i], name='dec_h' + str(i) + '_lin'),
                              alpha=self.leak), rate=self.dropout_rate)

            return h

    def latent_generator(self, noise, z_dim, reuse=False):
        """
        Latent Generator that is used to generate fake latent codes.
        :param noise: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, z_dim]
        """
        with tf.compat.v1.variable_scope('LatentGenerator') as scope:
            if reuse:
                scope.reuse_variables()
            if self.is_bn:
                h = tf.layers.batch_normalization(
                    lrelu(dense(noise, z_dim, self.lg_h_dim[0], name='lat_gen_h0_lin'),
                          alpha=self.leak),
                    training=self._is_train, name='lat_gen_bn0')
                for i in range(1, self.num_layers):
                    h = tf.layers.batch_normalization(
                        lrelu(dense(h, self.lg_h_dim[i - 1], self.lg_h_dim[i], name='lat_gen_h' + str(i) + '_lin'),
                              alpha=self.leak),
                        training=self._is_train, name='lat_gen_bn' + str(i))

            else:
                h = tf.nn.dropout(
                    lrelu(dense(noise, z_dim, self.lg_h_dim[0], name='lat_gen_h0_lin'),
                          alpha=self.leak),
                    rate=self.dropout_rate)
                for i in range(1, self.num_layers):
                    h = tf.nn.dropout(
                        lrelu(dense(h, self.lg_h_dim[i - 1], self.lg_h_dim[i], name='lat_gen_h' + str(i) + '_lin'),
                              alpha=self.leak), rate=self.dropout_rate)

            fake_lat = dense(h, self.lg_h_dim[self.num_layers - 1], z_dim, name='lat_gen_output')

            return fake_lat

    def discriminator(self, z, z_dim, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given prior distribution.
        :param z: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """
        with tf.compat.v1.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            if self.is_bn:

                h = tf.layers.batch_normalization(
                    lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1],
                                name='dis_h' + str(self.num_layers - 1) + '_lin'),
                          alpha=self.leak),
                    training=self._is_train, name='dis_bn' + str(self.num_layers - 1))
                for i in range(self.num_layers - 2, -1, -1):
                    h = tf.layers.batch_normalization(
                        lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='dis_h' + str(i) + '_lin'),
                              alpha=self.leak),
                        training=self._is_train, name='dis_bn' + str(i))

            else:

                h = tf.nn.dropout(
                    lrelu(dense(z, z_dim, self.d_h_dim[self.num_layers - 1],
                                name='dis_h' + str(self.num_layers - 1) + '_lin'),
                          alpha=self.leak),
                    rate=self.dropout_rate)
                for i in range(self.num_layers - 2, -1, -1):
                    h = tf.nn.dropout(
                        lrelu(dense(h, self.d_h_dim[i + 1], self.d_h_dim[i], name='dis_h' + str(i) + '_lin'),
                              alpha=self.leak), rate=self.dropout_rate)

            output = dense(h, self.d_h_dim[0], 1, name='dis_output')
            return output

    @property
    def model_dir(self):
        s = "scRAE_{}_{}_b_{}_g{}_lg{}_d{}_{}_{}_aelr_{}_ganlr_{}_leak_{}_dor_{}_z_{}_{}_bn_{}_lam_{}_giter_{}_epoch_{}".format(
            datetime.datetime.now(), self.dataset_name,
            self.batch_size, self.g_h_dim, self.lg_h_dim, self.d_h_dim, self.gen_activation, self.trans, self.ae_lr,
            self.gan_lr, self.leak, self.rate, self.z_dim, self.sampler, self.is_bn,
            self.lam, self.g_iter, self.epoch)
        s = s.replace('[', '_')
        s = s.replace(']', '_')
        s = s.replace(' ', '')
        return s

    def sample_Z(self, m, n, sampler='uniform'):
        if self.sampler == 'uniform':
            return np.random.uniform(-1., 1., size=[m, n])
        elif self.sampler == 'normal':
            return np.random.randn(m, n)

    def next_batch(self, data, max_size):
        indx = np.random.randint(max_size - self.batch_size)
        return data[indx:(indx + self.batch_size), :]

    def sample_gaussian(self, mean, variance, scope=None):

        with tf.compat.v1.variable_scope(scope, 'sample_gaussian'):
            sample = tf.random.normal(tf.shape(mean), mean, tf.sqrt(variance))
            sample.set_shape(mean.get_shape())
            return sample

    # Zero-inflated negative binomial (ZINB) model is for modeling count variables with excessive zeros and it is
    # usually for overdispersed count outcome variables.
    def zinb_model(self, x, mean, inverse_dispersion, logit, eps=1e-8):

        expr_non_zero = - tf.nn.softplus(- logit) \
                        + tf.math.log(inverse_dispersion + eps) * inverse_dispersion \
                        - tf.math.log(inverse_dispersion + mean + eps) * inverse_dispersion \
                        - x * tf.math.log(inverse_dispersion + mean + eps) \
                        + x * tf.math.log(mean + eps) \
                        - tf.math.lgamma(x + 1) \
                        + tf.math.lgamma(x + inverse_dispersion) \
                        - tf.math.lgamma(inverse_dispersion) \
                        - logit

        expr_zero = - tf.nn.softplus(- logit) \
                    + tf.nn.softplus(- logit + tf.math.log(inverse_dispersion + eps) * inverse_dispersion \
                                     - tf.math.log(inverse_dispersion + mean + eps) * inverse_dispersion)

        template = tf.cast(tf.less(x, eps), tf.float32)
        expr = tf.multiply(template, expr_zero) + tf.multiply(1 - template, expr_non_zero)
        return tf.reduce_sum(expr, axis=-1)

    def compute_metrices(self):
        try:
            # Embedding points in the test data to the latent space
            inp_encoder = self.data_test
            latent_matrix = self.sess.run(self.z, feed_dict={self.x_input: inp_encoder, self.dropout_rate: 0.0})

            labels = self.labels_test
            K = np.size(np.unique(labels))
            kmeans = KMeans(n_clusters=K, random_state=0).fit(latent_matrix)
            y_pred = kmeans.labels_
            NMI = nmi(labels.flatten(), y_pred.flatten(), average_method='geometric')
            AMI = ami(labels.flatten(), y_pred.flatten())
            HS = hs(labels.flatten(), y_pred.flatten())
            CS = cs(labels.flatten(), y_pred.flatten())
        except:
            latent_matrix = None
            NMI = 0
            AMI = 0
            HS = 0
            CS = 0
        return NMI, AMI, HS, CS, latent_matrix


if __name__ == '__main__':

    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
    flags.DEFINE_float("ae_lr", 0.001, "Learning rate of for adam [0.001]")
    flags.DEFINE_float("gan_lr", 0.0001, "Learning rate of for adam [0.001]")
    flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
    flags.DEFINE_integer("z_dim", 10, "Latent space dimension")
    flags.DEFINE_integer("n_l", 2, "# Hidden Layers")
    flags.DEFINE_integer("g_h_l1", 256, "#Generator Hidden Units in Layer 1")
    flags.DEFINE_integer("g_h_l2", 256, "#Generator Hidden Units in Layer 2")
    flags.DEFINE_integer("g_h_l3", 0, "#Generator Hidden Units in Layer 3")
    flags.DEFINE_integer("g_h_l4", 0, "#Generator Hidden Units in Layer 4")
    flags.DEFINE_integer("lg_h_l1", 256, "#Latent Generator Hidden Units in Layer 1")
    flags.DEFINE_integer("lg_h_l2", 256, "#Latent Generator Hidden Units in Layer 2")
    flags.DEFINE_integer("lg_h_l3", 0, "#Latent Generator Hidden Units in Layer 3")
    flags.DEFINE_integer("lg_h_l4", 0, "#Latent Generator Hidden Units in Layer 4")
    flags.DEFINE_integer("d_h_l1", 256, "#Discriminator Hidden Units in Layer 1")
    flags.DEFINE_integer("d_h_l2", 256, "#Discriminator Hidden Units in Layer 2")
    flags.DEFINE_integer("d_h_l3", 0, "#Discriminator Hidden Units in Layer 3")
    flags.DEFINE_integer("d_h_l4", 0, "#Discriminator Hidden Units in Layer 4")
    flags.DEFINE_string("actv", "sig", "Decoder Activation [sig, tanh, lin]")
    flags.DEFINE_float("leak", 0.3, "Leak factor")
    flags.DEFINE_float("dropout_rate", 0.1, "Dropout rate")
    flags.DEFINE_string("trans", "sparse", "Data Transformation [dense, sparse]")
    flags.DEFINE_string("dataset", "10x_73k", "The name of dataset [10x_73k, 10x_68k, Zeisel, Macosko]")
    flags.DEFINE_string("checkpoint_dir", "/data/eugene/AAE-20180306-Hemberg/test_checkpoint",
                        "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("sample_dir", "test_samples", "Directory name to save the image samples [samples]")
    flags.DEFINE_string("result_dir", "test_result", "Directory name to results of gene imputation [result]")
    flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
    flags.DEFINE_integer("g_iter", 2, "# Generator Iterations [2]")
    flags.DEFINE_boolean("bn", False, "True for batch Norm [False]")
    flags.DEFINE_float("lam", 10.0, "Lambda for regularization")
    flags.DEFINE_string("sampler", "normal", "The sampling distribution of z [uniform, normal, mix_gauss]")
    flags.DEFINE_string("model", "aae", "Model to train [aae, van_ae] [aae]")
    flags.DEFINE_integer("X_dim", 720, "Input dimension")
    flags.DEFINE_integer("perplexity", 50,
                         "Related to the number of nearest neighbors used in manifold learning algorithms")
    FLAGS = flags.FLAGS

    print("dataset: {}".format(FLAGS.dataset))
    print("checkpoint_dir: {}".format(FLAGS.checkpoint_dir))
    print("n_l: {}".format(FLAGS.n_l))
    print("g_h_l1: {}".format(FLAGS.g_h_l1))
    print("g_h_l2: {}".format(FLAGS.g_h_l2))
    print("g_h_l3: {}".format(FLAGS.g_h_l3))
    print("g_h_l4: {}".format(FLAGS.g_h_l4))
    print("lg_h_l1: {}".format(FLAGS.lg_h_l1))
    print("lg_h_l2: {}".format(FLAGS.lg_h_l2))
    print("lg_h_l3: {}".format(FLAGS.lg_h_l3))
    print("lg_h_l4: {}".format(FLAGS.lg_h_l4))
    print("d_h_l1: {}".format(FLAGS.d_h_l1))
    print("d_h_l2: {}".format(FLAGS.d_h_l2))
    print("d_h_l3: {}".format(FLAGS.d_h_l3))
    print("d_h_l4: {}".format(FLAGS.d_h_l4))
    print("batch_size: {}".format(FLAGS.batch_size))
    print("ae_learning_rate: {}".format(FLAGS.ae_lr))
    print("gan_learning_rate: {}".format(FLAGS.gan_lr))
    print("z_dim: {}".format(FLAGS.z_dim))
    print("epoch: {}".format(FLAGS.epoch))
    print("leak: {}".format(FLAGS.leak))
    print("dropout rate: {}".format(FLAGS.dropout_rate))
    print("model: {}".format(FLAGS.model))
    print("trans: {}".format(FLAGS.trans))
    print("actv: {}".format(FLAGS.actv))
    print("X_dim: {}".format(FLAGS.X_dim))
    print("bn: {}".format(FLAGS.bn))
    print("g_iter: {}".format(FLAGS.g_iter))
    print("lam: {}".format(FLAGS.lam))
    print("sampler: {}".format(FLAGS.sampler))
    print("perplexity: {}".format(FLAGS.perplexity))


    def main(_):

        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 0.333
        run_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=run_config) as sess:

            g_h_dim = [FLAGS.g_h_l1, FLAGS.g_h_l2, FLAGS.g_h_l3, FLAGS.g_h_l4]
            lg_h_dim = [FLAGS.lg_h_l1, FLAGS.lg_h_l2, FLAGS.lg_h_l3, FLAGS.lg_h_l4]
            d_h_dim = [FLAGS.d_h_l1, FLAGS.d_h_l2, FLAGS.d_h_l3, FLAGS.d_h_l4]

            if FLAGS.model == 'scRAE':
                test_scRAE = Test_scRAE(
                    sess,
                    epoch=FLAGS.epoch,
                    ae_lr=FLAGS.ae_lr,
                    gan_lr=FLAGS.gan_lr,
                    batch_size=FLAGS.batch_size,
                    X_dim=FLAGS.X_dim,
                    z_dim=FLAGS.z_dim,
                    dataset_name=FLAGS.dataset,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    sample_dir=FLAGS.sample_dir,
                    result_dir=FLAGS.result_dir,
                    num_layers=FLAGS.n_l,
                    g_h_dim=g_h_dim[:FLAGS.n_l],
                    lg_h_dim=lg_h_dim[:FLAGS.n_l],
                    d_h_dim=d_h_dim[:FLAGS.n_l],
                    gen_activation=FLAGS.actv,
                    leak=FLAGS.leak,
                    rate=FLAGS.dropout_rate,
                    trans=FLAGS.trans,
                    is_bn=FLAGS.bn,
                    g_iter=FLAGS.g_iter,
                    lam=FLAGS.lam,
                    sampler=FLAGS.sampler,
                    perplexity=FLAGS.perplexity)

            # show_all_variables()
            if FLAGS.train:
                if FLAGS.model == 'scRAE':
                    test_scRAE.train_cluster()


    tf.compat.v1.app.run()
