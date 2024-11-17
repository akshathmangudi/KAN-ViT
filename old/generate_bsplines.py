import numpy
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

"""
generate_bsplines.py:

Generates the B-spline using a given set of points. Uses scipy's BSpline.

Arguments:
- X: vector of input points
- y: vector of output/response points
- s: smoothing parameter.
- k: degree of the spline. [When dealing with small values of s, avoid even values of k]
- N: the number of points to visualize a smooth line for the spline interpolation.
"""

def generate_bspline(X: numpy.array, y: numpy.array, s: int, k: int, N: int) -> None:
    assert len(X) == len(y), "X and y must have the same length"
    knots, coeffs, degree = interpolate.splrep(numpy.sort(X), numpy.sort(y), s=s, k=k)
    x_line = numpy.linspace(X.min(), X.max(), N)

    spline = interpolate.BSpline(knots, coeffs, degree, extrapolate=False)

    return x_line, spline

def visualize_spline(X: numpy.array, y: numpy.array, x_line, spline):
    plt.plot(X, y, 'bo', label="Original points")
    plt.plot(x_line, spline(x_line), 'r', label="BSpline")
    plt.grid()
    plt.legend(loc='best')
    plt.show()
