from os.path import dirname, abspath, join

"""------------------------- Can be modified. ------------------------"""

# The project's plotting mode. It can be 'save', 'show', 'both' or 'none'.
PLOTTING_MODE = 'show'

# Whether predictions should be saved as an excel file or not.
SAVE_PRED_RESULTS = False

# The output directory's name.
OUT_DIR_NAME = 'out'
# The datasets folder name.
DATASETS = 'datasets'

# The seizure detection dataset's filename.
DATASET_SEIZURE = 'eeg.csv'

# The mnist train dataset's filename.
DATASET_MNIST = 'mnist.csv'

""" ----------------------------------------------------------------- """

""" ------------------------- Do not modify. -------------------------"""

# The root folder.
__ROOT = dirname(abspath(__file__))
# The output folder's path.
__OUT_PATH = join(__ROOT, OUT_DIR_NAME)

# The path to the seizure detection dataset.
__SEIZURE_PATH = join(__ROOT, DATASETS, DATASET_SEIZURE)

# The path to the seizure detection train dataset.
__MNIST_PATH = join(__ROOT, DATASETS, DATASET_MNIST)

# Check plotting mode.
if PLOTTING_MODE != 'show' and PLOTTING_MODE != 'save' and PLOTTING_MODE != 'both':
    raise ValueError('Plotter\'s mode can be \'none\', \'save\', \'show\' or \'both\'.\nGot {} instead.'
                     .format(PLOTTING_MODE))
