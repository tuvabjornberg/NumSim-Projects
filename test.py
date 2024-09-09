import main

def test_mass():
    assert main.mass(30) == 4
    assert main.mass(5) == 6.0
    assert main.mass(0) == 8
    
def test_velocity():
    assert main.velocity(30) == (107.97601492130883, -691.6221368650033)
    assert main.velocity(0) == (700, 0)
    
#def test_external_forces():
#    assert main.external_forces(30) == 4

test_mass()
test_velocity()
#test_external_forces()
print("All tests passed")