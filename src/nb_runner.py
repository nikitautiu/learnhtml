# standard libraries
import os

# third-party libraries
import nbformat
import nbparameterise
from nbconvert.preprocessors import ExecutePreprocessor


def read_in_notebook(notebook_fp):
    with open(notebook_fp) as f:
        nb = nbformat.read(f, as_version=4)
    return nb


def set_parameters(nb, params_dict):
    orig_parameters = nbparameterise.extract_parameters(nb)
    params = nbparameterise.parameter_values(orig_parameters, **params_dict)
    new_nb = nbparameterise.replace_definitions(nb, params, execute=False)
    return new_nb


# modified from https://nbconvert.readthedocs.io/en/latest/execute_api.html
def execute_notebook(notebook_filename, notebook_filename_out=None, params_dict={}, run_path="", timeout=6000000):
    """Executes a notebook with a given filename and uoutputs to a give filename, or the same.
    Also receives a dict of parameters to override in the notebook. This gives the ability to run a
    notebook and override desired globals.

    The only globals that can be updated are those in cells that only contain simple assignments.
    Such a cell should be situated a t the top of the notebook for visibility."""
    notebook_fp = os.path.join(run_path, notebook_filename)
    nb = read_in_notebook(notebook_fp)
    new_nb = set_parameters(nb, params_dict)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')

    try:
        ep.preprocess(new_nb, {'metadata': {'path': run_path}})
    except:
        msg = 'Error executing the notebook "{0}".\n\n'.format(notebook_filename)
        msg = '{0}See notebook "{1}" for the traceback.'.format(msg, notebook_filename_out)
        print(msg)
        raise
    finally:
        with open(notebook_filename_out, mode='wt') as f:
            nbformat.write(new_nb, f)
