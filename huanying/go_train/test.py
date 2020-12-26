import h5py
f = h5py.File('./model_data/yolo.h5', 'r')
print(f.attrs.get('keras_version'))