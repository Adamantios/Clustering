from os.path import dirname, abspath, join

"""------------------------- Can be modified. ------------------------"""

# The project's plotting mode. It can be 'save', 'show', 'both' or 'none'.
# TODO change to show.
PLOTTING_MODE = 'both'

# Whether predictions should be saved as an excel file or not.
SAVE_PRED_RESULTS = False

# The output directory's name.
OUT_DIR_NAME = 'out'
# The datasets folder name.
DATASETS = 'datasets'

# The voice dataset's filename.
DATASET_VOICE = 'voice.csv'
# The voice train dataset's filename.
DATASET_VOICE_TRAIN = 'voice_train.csv'
# The voice test dataset's filename.
DATASET_VOICE_TEST = 'voice_test.csv'

# The genes data filename.
DATA_GENES = 'genes.csv'
# The genes labels filename.
LABELS_GENES = 'genes_labels.csv'
# The genes train dataset's filename.
DATASET_GENES_TRAIN = 'genes_train.csv'
# The genes test dataset's filename.
DATASET_GENES_TEST = 'genes_test.csv'

""" ----------------------------------------------------------------- """

""" ------------------------- Do not modify. -------------------------"""

# The root folder.
__ROOT = dirname(abspath(__file__))
# The output folder's path.
__OUT_PATH = join(__ROOT, OUT_DIR_NAME)

# The path to the voice dataset.
__VOICE_PATH = join(__ROOT, DATASETS, DATASET_VOICE)
# The path to the voice train dataset.
__VOICE_TRAIN_PATH = join(__ROOT, DATASETS, DATASET_VOICE_TRAIN)
# The path to the voice test dataset.
__VOICE_TEST_PATH = join(__ROOT, DATASETS, DATASET_VOICE_TEST)

# The path to the genes data.
__GENES_DATA_PATH = join(__ROOT, DATASETS, DATA_GENES)
# The path to the genes labels.
__GENES_LABELS_PATH = join(__ROOT, DATASETS, LABELS_GENES)
# The path to the genes train dataset.
__GENES_TRAIN_PATH = join(__ROOT, DATASETS, DATASET_GENES_TRAIN)
# The path to the genes test dataset.
__GENES_TEST_PATH = join(__ROOT, DATASETS, DATASET_GENES_TEST)

# Check plotting mode.
if PLOTTING_MODE != 'show' and PLOTTING_MODE != 'save' and PLOTTING_MODE != 'both':
    raise ValueError('Plotter\'s mode can be \'none\', \'save\', \'show\' or \'both\'.\nGot {} instead.'
                     .format(PLOTTING_MODE))
