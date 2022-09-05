from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence, Reduction
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfpd = tfp.distributions
tfpl = tfp.layers

mu, sigma = 0.0, 4.0

def priorModel(kernel_size, bias_size, dtype=None):
    size_sum = kernel_size + bias_size
    prior_model = Sequential()
    prior_model.add(tfpl.DistributionLambda(
        lambda t: tfpd.MultivariateNormalDiag(loc=mu*tf.ones(size_sum), scale_diag=sigma*tf.ones(size_sum))))
    return prior_model

def postModel(kernel_size, bias_size, dtype=None):
    size_sum = kernel_size + bias_size
    post_model = Sequential()
    post_model.add(tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(size_sum), dtype=dtype))
    post_model.add(tfpl.MultivariateNormalTriL(size_sum))
    return post_model

def fullModel(nbsteps):
    full_model = Sequential()
    for _ in np.arange(0,1):
        full_model.add(tfpl.DenseVariational(64, input_dim=1,
                                        make_prior_fn=priorModel, make_posterior_fn=postModel,
                                        kl_weight=1/nbsteps, kl_use_exact=False, activation="sigmoid"))
    for _ in np.arange(0,1):
        full_model.add(tfpl.DenseVariational(tfpl.IndependentNormal.params_size(1),
                                        make_prior_fn=priorModel, make_posterior_fn=postModel,
                                        kl_weight=1/nbsteps, kl_use_exact=False))
    full_model.add(tfpl.IndependentNormal(1))
    kl_divergence = KLDivergence(reduction="auto", name="kl_divergence")
    opt = Adam(learning_rate=3e-4)
    full_model.compile(loss=kl_divergence, optimizer=opt)
    # full_model.summary()
    return full_model
