
import subprocess
import os
import pytest
import netCDF4
import numpy as np

@pytest.fixture(scope="module")
def runoptics_output():
    """Fixture to run the runoptics.py script and provide the output file path."""
    run_dir = "src/geosmie"
    input_json = "../config/geosparticles/bc_light.json"
    output_dir_from_root = "test_output"
    output_dir_from_rundir = f"../../{output_dir_from_root}"
    output_filename = "optics_bc_light.nomom.nc4"
    output_filepath = os.path.join(output_dir_from_root, output_filename)

    # Ensure the output directory exists
    os.makedirs(output_dir_from_root, exist_ok=True)

    command = [
        "python",
        "./runoptics.py",
        "--name",
        input_json,
        "--dest",
        output_dir_from_rundir,
    ]

    # Run the script from the correct directory
    # Add project root to PYTHONPATH so that absolute imports work
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
    subprocess.run(command, check=True, cwd=run_dir, capture_output=True, text=True, env=env)

    # Yield the path to the output file
    yield output_filepath

    # Teardown: remove the generated file and directory
    if os.path.exists(output_filepath):
        os.remove(output_filepath)
    if os.path.exists(output_dir_from_root):
        os.rmdir(output_dir_from_root)

def test_output_file_creation(runoptics_output):
    """Tests if the output NetCDF file is created."""
    assert os.path.exists(runoptics_output), "Output file was not created."

def test_output_file_variables_and_dimensions(runoptics_output):
    """Tests the variables and their dimensions in the output NetCDF file."""
    with netCDF4.Dataset(runoptics_output, 'r') as ncfile:
        # Expected dimensions based on bc_light.json and the script's logic
        expected_dims = {
            'rh': 4,
            'wavelength': 61,
            'bin': 2,  # Updated to 2 to account for the hydrophobic bin
            'ang': 371,
            'p': 6
        }

        # Check dimensions
        for dim_name, expected_size in expected_dims.items():
            assert dim_name in ncfile.dimensions, f"Dimension '{dim_name}' not found."
            assert len(ncfile.dimensions[dim_name]) == expected_size, \
                f"Dimension '{dim_name}' has incorrect size."

        # Expected variables and their dimension shapes
        expected_vars = {
            'p11': ('bin', 'wavelength', 'rh', 'ang'),
            'qext': ('bin', 'wavelength', 'rh'),
            'g': ('bin', 'wavelength', 'rh'),
            'ssa': ('bin', 'wavelength', 'rh'),
            'mass': ('bin', 'rh'),
            'refreal': ('bin', 'wavelength', 'rh'),
        }

        # Check variables and their shapes
        for var_name, expected_shape_names in expected_vars.items():
            assert var_name in ncfile.variables, f"Variable '{var_name}' not found."

            var = ncfile.variables[var_name]
            var_shape = var.shape
            expected_shape = tuple(expected_dims[dim] for dim in expected_shape_names)

            assert var_shape == expected_shape, \
                f"Variable '{var_name}' has shape {var_shape}, expected {expected_shape}."

def test_numerical_identity(runoptics_output):
    """
    Tests that the numerical output of the refactored code is identical to a
    golden reference file.
    """
    golden_filepath = "golden_files/optics_bc_light.nomom.nc4"
    assert os.path.exists(golden_filepath), "Golden reference file not found."

    with netCDF4.Dataset(runoptics_output, 'r') as generated_file, \
         netCDF4.Dataset(golden_filepath, 'r') as golden_file:

        # Compare dimensions
        for dim_name in golden_file.dimensions:
            assert dim_name in generated_file.dimensions
            assert len(generated_file.dimensions[dim_name]) == len(golden_file.dimensions[dim_name])

        # Compare variables
        for var_name in golden_file.variables:
            assert var_name in generated_file.variables
            generated_var = generated_file.variables[var_name]
            golden_var = golden_file.variables[var_name]
            np.testing.assert_allclose(generated_var[:], golden_var[:], rtol=1e-7, atol=0)
