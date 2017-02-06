
IMAGE_LIST_TRAIN = './data/image/train.txt'
IMAGE_LIST_VAL = './data/image/val.txt'
IMAGE_LIST_TEST = './data/image/test.txt'
N_CLASSES = 40

INIT_LEARNING_RATE = 0.01
BATCH_SIZE = 32
VAL_SAMPLE_SIZE = 256

BATCH_NORM = False
# BATCH_NORM = False
BN_AFTER_ACTV = True  # conv -> relu -> bn
# BN_AFTER_ACTV = False  # conv -> bn -> relu

print {k: v for k,v in locals().iteritems() if '__' not in k}
