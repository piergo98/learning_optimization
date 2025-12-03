import numpy as np
import os, subprocess, sys
import scipy.io

def load_data(filename):
    """
    Load data from file <filename>, which has to be in the data folder.
    The function loads both csv or mat files
    """
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    file_path = os.path.join(data_folder, filename)

    if filename.endswith('.csv'):
        loaded_data = np.loadtxt(file_path, delimiter=',')
    elif filename.endswith('.mat'):
        loaded_data = scipy.io.loadmat(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .mat files.")
    
    # Handle data after loading
    keys_to_keep = ['n_phases', 'controls', 'phases_duration']

    data = {k: loaded_data[k] for k in keys_to_keep}
    # Normalize and reshape controls into shape (n_inputs, n_phases)
    controls = np.asarray(data['controls']).ravel()
    n_phases = int(np.squeeze(np.asarray(data['n_phases'])))
    if controls.size % n_phases != 0:
        raise ValueError(f"Controls length ({controls.size}) is not divisible by n_phases ({n_phases}).")
    n_inputs = controls.size // n_phases
    controls = controls.reshape((n_inputs, n_phases))
    data['n_inputs'] = n_inputs

    # Ensure phases_duration is a 1D array
    data['phases_duration'] = np.asarray(data['phases_duration']).ravel()

    return data