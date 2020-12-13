import numpy

#reduces mnist data set to only 0 to digits
def reduce_mnist(x, y, digits):
    delrows = numpy.zeros(y.size,numpy.int8)
    for i in range(0, y.size):
        if y[i] > digits:
            delrows[i] = 1
    delrows = numpy.where(delrows)
    x = numpy.delete(x,delrows,0)
    y = numpy.delete(y,delrows,0)
    delrows = numpy.zeros(60000,numpy.int8)
    return (x, y)

    