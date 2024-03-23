import jax
import jax.numpy as jnp
from jax.scipy.special import entr
from jaxtyping import Array, Float, Scalar
precision = jax.lax.Precision.HIGHEST

def deep_proj_P(cov: Float[Array, "d d"], dCov: Float[Array, "d d"], eps: Scalar = 1e-6) -> Scalar:
    estimator = jnp.matmul(spd_inv(cov, eps = eps), dCov, precision=precision)
    return jnp.trace(estimator)

def metric_reg(cov: Float[Array, "d d"], reg: Scalar = 1, eps: Scalar = 1e-6) -> Scalar:
    vals = jnp.linalg.eigvalsh(cov)
    vals = jnp.where(vals > eps, vals, 0)
    return reg*jnp.mean(1 - vals - entr(vals))

def fro_metric_reg(cov: Float[Array, "d d"], reg: Scalar = 1) -> Scalar:
    return reg*jnp.linalg.norm(cov - jnp.eye(cov.shape[0]), ord='fro')**2

def spd_inv(A: Float[Array, "d d"], eps: Scalar = 1e-6, strategy: str = 'trunc') -> Float[Array, "d d"]:
    """
    Truncated eigenvalue decomposition of A
    """
    w, v = jnp.linalg.eigh(A)
    if strategy == 'trunc':
        inv_w = jnp.where(w > eps, w**-1, 0.0)
        v = jnp.where(w > eps, v, 0.0)
    elif strategy == 'tikhonov':
        inv_w = (w + eps)**-1
    else:
        raise NotImplementedError(f"Strategy {strategy} not implemented")
    return jnp.linalg.multi_dot([v, jnp.diag(inv_w), v.T], precision=precision)
