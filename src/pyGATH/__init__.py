"""pyGATH: GPU-accelerated ray tracing for high-power laser systems.

pyGATH uses double precision throughout. This setting is applied before any
package submodules create JAX arrays.
"""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

del _jax_config
