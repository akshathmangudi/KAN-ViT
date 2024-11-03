import torch
import numpy


class NaiveFourierKANLayer(torch.nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True, smootorch_initialization=False):
        """
        Constructor for NaiveFourierKANLayer.

        Parameters
        ----------
        inputdim : int
            The number of input dimensions.
        outdim : int
            The number of output dimensions.
        gridsize : int
            The number of grid points to use for the Fourier transform.
        addbias : bool, optional
            Whether to include a bias term in the layer. Defaults to True.
        smootorch_initialization : bool, optional
            Whether to use the initialization scheme for the Fourier coefficients. Defaults to False.
        """
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        grid_norm_factor = (
            torch.arange(gridsize) + 1)**2 if smootorch_initialization else numpy.sqrt(gridsize)
        self.fouriercoeffs = torch.nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                                (numpy.sqrt(inputdim) * grid_norm_factor))
        if (self.addbias):
            self.bias = torch.nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        k = torch.reshape(torch.arange(1, self.gridsize+1,
                                       device=x.device), (1, 1, 1, self.gridsize))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))

        c = torch.cos(k*xrshp)
        s = torch.sin(k*xrshp)

        y = torch.sum(c*self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(s*self.fouriercoeffs[1:2], (-2, -1))
        if (self.addbias):
            y += self.bias
        y = torch.reshape(y, outshape)
        return y
