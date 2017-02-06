
IMAGE_LIST_TRAIN = './data/train.list'
IMAGE_LIST_VAL = './data/val.list'
IMAGE_LIST_TEST = './data/test.list'
N_CLASSES = 40

INIT_LEARNING_RATE = 0.001
BATCH_SIZE = 32
VAL_SAMPLE_SIZE = 256

BATCH_NORM = False
# BATCH_NORM = False
BN_AFTER_ACTV = True  # conv -> relu -> bn
# BN_AFTER_ACTV = False  # conv -> bn -> relu

print {k: v for k,v in locals().iteritems() if '__' not in k}
