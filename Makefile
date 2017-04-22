DATA_DIR = data

TRAIN_CSV = $(DATA_DIR)/train.csv
TEST_CSV = $(DATA_DIR)/test.csv

TRAIN_LABELS_NPY = $(DATA_DIR)/train_labels.npy
TRAIN_PIXELS_NPY = $(DATA_DIR)/train_pixels.npy
TEST_PIXELS_NPY  = $(DATA_DIR)/test_pixels.npy

.PHONY : data_npy
data_npy: $(TRAIN_LABELS_NPY) $(TRAIN_PIXELS_NPY) $(TEST_PIXELS_NPY)

$(TRAIN_LABELS_NPY) $(TRAIN_PIXELS_NPY) $(TEST_PIXELS_NPY): $(TRAIN_CSV) $(TEST_CSV)
	./activate
	python ./data_make_ndarray.py

$(TRAIN_CSV) $(TEST_CSV):
	@echo "Download `basename $@`"
	exit 1
