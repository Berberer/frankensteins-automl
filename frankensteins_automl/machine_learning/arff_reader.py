import arff
import numpy


def read_arff(path, target_column_index):
    data = numpy.array(arff.load(open(path, "r"))["data"])
    data = data.astype(numpy.float64)
    data_x = data[:, :target_column_index]
    data_y = data[:, target_column_index]
    return data_x, data_y
