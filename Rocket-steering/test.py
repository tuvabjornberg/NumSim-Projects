import main
import numpy as np


def test_mass():
    assert main.mass(30) == 4
    assert main.mass(5) == 6.0
    assert main.mass(0) == 8


def test_mass_der():
    assert main.mass_der(5) == -0.4
    assert main.mass_der(15) == 0


def test_engine_dir():
    assert main.engine_dir(5, 0, 10) == np.pi / 2
    assert main.engine_dir(25, 0, 30) == 25  # TODO: Stub in engine dir height >= 20


def test_fuel_velocity():
    np.testing.assert_allclose(
        main.fuel_velocity(30, 10), (4.2862637970157365e-14, 700.0)
    )
    np.testing.assert_allclose(
        main.fuel_velocity(0, 0), (4.2862637970157365e-14, 700.0)
    )
    np.testing.assert_allclose(
        main.fuel_velocity(40, 20), (-466.85664315658335, 521.5792123355442)
    )  # TODO: Stub in engine dir


def test_external_forces():
    v = np.array([0, 0])
    t = 5
    expected_forces = (0, -58.92)
    ext_forces = main.external_forces(t, v)
    np.testing.assert_allclose(ext_forces, expected_forces, atol=1e-2)


def test_ode_rhs():
    t = 5
    v = np.array([0, 0])
    height = 10
    rhs_value = main.ode_rhs(t, v, height)
    assert rhs_value is not None


test_mass()
test_mass_der()
test_engine_dir()
test_fuel_velocity()
test_external_forces()
test_ode_rhs()

print("All tests passed")
