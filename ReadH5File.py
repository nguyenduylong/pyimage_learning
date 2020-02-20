import h5py

class ReadH5Data:
    def readData(input_file):
        with h5py.File(input_file, 'r') as f:
            return f['images'].value, f['labels'].value; 
        f.close()