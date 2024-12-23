import math
import torch

# Code directly taken from this repository: https://github.com/ereinha/SineKAN


def forward_step(i_n, grid_size, A, K, C):
    """
    Computes the next step in a sequence using the given parameters.

    Args:
        i_n (torch.Tensor): The current input tensor.
        grid_size (int): The size of the grid.
        A (float): A scaling factor for the grid size.
        K (float): An exponent applied to the grid size.
        C (float): A constant added to the scaled grid size.

    Returns:
        torch.Tensor: The computed output tensor for the next step.
    """
    ratio = A * grid_size**(-K) + C
    i_n1 = ratio * i_n
    return i_n1


class SineKANLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu', grid_size=5, is_first=False, add_bias=True, norm_freq=True):
        """
        Initializes the SineKAN layer.

        Args:
            input_dim (int): The input dimensionality.
            output_dim (int): The output dimensionality.
            device (str): The device to put the layer on.
            grid_size (int): The grid size for the sine functions.
            is_first (bool): Whether this is the first layer in the network.
            add_bias (bool): Whether to add a bias term to the layer.
            norm_freq (bool): Whether to normalize the frequency of the sine functions.
        """
        super(SineKANLayer, self).__init__()
        self.grid_size = grid_size
        self.device = device
        self.is_first = is_first
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A, self.K, self.C = 0.9724108095811765, 0.9884401790754128, 0.999449553483052

        self.grid_norm_factor = (torch.arange(grid_size) + 1)
        self.grid_norm_factor = self.grid_norm_factor.reshape(1, 1, grid_size)

        if is_first:
            self.amplitudes = torch.nn.Parameter(torch.empty(
                output_dim, input_dim, 1).normal_(0, .4) / output_dim / self.grid_norm_factor)
        else:
            self.amplitudes = torch.nn.Parameter(torch.empty(
                output_dim, input_dim, 1).uniform_(-1, 1) / output_dim / self.grid_norm_factor)

        grid_phase = torch.arange(
            1, grid_size + 1).reshape(1, 1, 1, grid_size) / (grid_size + 1)
        self.input_phase = torch.linspace(
            0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(device)
        phase = grid_phase.to(device) + self.input_phase

        if norm_freq:
            self.freq = torch.nn.Parameter(torch.arange(
                1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
        else:
            self.freq = torch.nn.Parameter(torch.arange(
                1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        for i in range(1, self.grid_size):
            phase = forward_step(phase, i, self.A, self.K, self.C)
        # self.phase = torch.nn.Parameter(phase)
        self.register_buffer('phase', phase)

        if self.add_bias:
            self.bias = torch.nn.Parameter(
                torch.ones(1, output_dim) / output_dim)

    def forward(self, x):
        x_shape = x.shape
        output_shape = x_shape[0:-1] + (self.output_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        s = torch.sin(x_reshaped * self.freq + self.phase)
        y = torch.einsum('ijkl,jkl->ij', s, self.amplitudes)
        if self.add_bias:
            y += self.bias
        y = torch.reshape(y, output_shape)
        return y

    def forward_step(self, i_n, grid_size, A, K, C):
        """
        Computes the next step in a sequence using the given parameters.

        Args:
            i_n (torch.Tensor): The current input tensor.
            grid_size (int): The size of the grid.
            A (float): A scaling factor for the grid size.
            K (float): An exponent applied to the grid size.
            C (float): A constant added to the scaled grid size.

        Returns:
            torch.Tensor: The computed output tensor for the next step.
        """
        ratio = A * grid_size**(-K) + C
        i_n1 = ratio * i_n
        return i_n1
