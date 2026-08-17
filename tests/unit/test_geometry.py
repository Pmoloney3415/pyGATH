import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyGATH.grid import Geometry, convert_positions, convert_vectors


@pytest.mark.parametrize("target", [Geometry.CYLINDRICAL, Geometry.SPHERICAL])
def test_batched_position_round_trip(target):
    cartesian = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [-2.0, 1.0, -0.5],
            [0.5, -1.5, 2.0],
        ]
    )
    transformed = convert_positions(cartesian, Geometry.CARTESIAN, target)
    recovered = convert_positions(transformed, target, Geometry.CARTESIAN)
    np.testing.assert_allclose(recovered, cartesian, rtol=1.0e-6, atol=1.0e-6)


def test_cylindrical_physical_vector_components():
    positions = jnp.array([[2.0, jnp.pi / 2.0, -1.0]])
    radial = jnp.array([[1.0, 0.0, 0.0]])
    azimuthal = jnp.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(
        convert_vectors(radial, positions, "cylindrical", "cartesian"),
        [[0.0, 1.0, 0.0]],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        convert_vectors(azimuthal, positions, "cylindrical", "cartesian"),
        [[-1.0, 0.0, 0.0]],
        atol=1.0e-6,
    )


def test_spherical_physical_vector_components():
    positions = jnp.array([[2.0, 0.0, jnp.pi / 2.0]])
    vectors = jnp.eye(3)
    positions = jnp.broadcast_to(positions, (3, 3))
    converted = convert_vectors(vectors, positions, "spherical", "cartesian")
    np.testing.assert_allclose(
        converted,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        atol=1.0e-6,
    )


def test_geometry_conversions_are_jittable():
    convert = jax.jit(
        lambda values: convert_positions(values, "cartesian", "spherical")
    )
    result = convert(jnp.array([[0.0, 1.0, 0.0]]))
    np.testing.assert_allclose(result, [[1.0, jnp.pi / 2.0, jnp.pi / 2.0]])


def test_coordinate_inputs_require_component_axis():
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        convert_positions(jnp.ones((2, 2)), "cartesian", "spherical")
