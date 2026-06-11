import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a music sequence."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class KLDivergenceLayer(layers.Layer):
    """Calculates and adds KL divergence loss for continuous latent variables."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        self.add_loss(tf.reduce_mean(kl_loss) * 0.1)
        return z_mean, z_log_var

class GumbelSoftmaxLayer(layers.Layer):
    """Applies Gumbel-Softmax sampling to discrete token probabilities."""
    def __init__(self, temperature=1.0, **kwargs):
        super(GumbelSoftmaxLayer, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, logits):
        U = tf.random.uniform(tf.shape(logits), minval=0.0, maxval=1.0)
        gumbel_noise = -tf.math.log(-tf.math.log(U + 1e-20) + 1e-20)
        y = (logits + gumbel_noise) / self.temperature
        return tf.nn.softmax(y)

class GumbelKLDivergenceLayer(layers.Layer):
    """Calculates categorical KL divergence for the discrete choices."""
    def call(self, logits):
        q_prop = tf.nn.softmax(logits)
        log_q_prop = tf.math.log(q_prop + 1e-20)
        n_classes = tf.cast(tf.shape(logits)[-1], tf.float32)
        kl_loss = tf.reduce_sum(q_prop * (log_q_prop - tf.math.log(1.0 / n_classes)), axis=-1)
        self.add_loss(tf.reduce_mean(kl_loss) * 0.1)
        return logits