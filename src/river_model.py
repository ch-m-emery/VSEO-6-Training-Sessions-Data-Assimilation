"""
river_model.py module : Muskingum-Cunge river routing model over a river network
"""

import numpy as np
import time
from matplotlib import pyplot as plt

class RiverModel:
    """A class to simulate a river routing model based on the Muskingum-Cunge model
    """

    def __init__(self, n_dim=5, mat_n_in=None, vec_k_in=None, vec_x_in=None):
        """Class constructor
        :param n_dim:
        :param mat_n_in:
        :param vec_k_in:
        :param vec_x_in:
        """

        if not isinstance(n_dim, int):
            raise TypeError("Input network size 'n_dim' must be an integer")
        if n_dim<=0:
            raise ValueError("Input network size 'n_dim' must be positive")
        self._n_dim = n_dim

        if mat_n_in is None:
            raise ValueError("Missing input network matrix 'mat_n_in'")
        if mat_n_in.ndim != 2:
            raise ValueError("Input network matrix must be 2D")
        if mat_n_in.shape[0] != mat_n_in.shape[1]:
            raise ValueError("Input network matrix must be square")
        if mat_n_in.shape[0] != n_dim:
            raise ValueError("Input network matrix size does not match input network size")
        if np.all(np.tril(mat_n_in, k=0) == 0):
            raise ValueError("Input network matrix must be strictly lower triangular")
        self._mat_n_in = mat_n_in

        if vec_k_in is None:
            raise ValueError("Missing input parameter vector 'vec_k_in'")
        if vec_k_in.ndim != 1:
            raise ValueError("Input k-parameter vector must be 1D")
        if vec_k_in.shape[0] != n_dim:
            raise ValueError("Input k-parameter vector size does not match input network size")
        self._vec_k_in = vec_k_in

        if vec_x_in is None:
            raise ValueError("Missing input parameter vector 'vec_x_in'")
        if vec_x_in.ndim != 1:
            raise ValueError("Input x-parameter vector must be 1D")
        if vec_x_in.shape[0] != n_dim:
            raise ValueError("Input x-parameter vector size does not match input network size")
        self._vec_x_in = vec_x_in

    def set_c1(self, d_t):
        """Set C1 parameters from river network parameters
        :return:
        """

        vec_c1 = (0.5*d_t - self._vec_k_in*self._vec_x_in) / (self._vec_k_in*(1-self._vec_x_in) + 0.5*d_t)

        return vec_c1

    def set_c2(self, d_t):
        """Set C2 parameters from river network parameters
        :return:
        """

        vec_c2 = (0.5 * d_t + self._vec_k_in * self._vec_x_in) / (self._vec_k_in * (1 - self._vec_x_in) + 0.5 * d_t)

        return vec_c2

    def set_c3(self, d_t):
        """Set C1 parameters from river network parameters
        :return:
        """

        vec_c3 = (self._vec_k_in * (1 - self._vec_x_in) - 0.5 * d_t) / (self._vec_k_in * (1 - self._vec_x_in) + 0.5 * d_t)

        return vec_c3

    def _prepare_run(self, in_longest_path=2, vec_c1=None):
        """Not optimized, can not be used on a big network
        :param in_longest_path: int
            add a routine to automatically copute it
        :param vec_c1:
        :return:
        """

        mat_model = np.eye(self._n_dim)
        mat_propag = np.diag(vec_c1) @ self._mat_n_in
        mat_tmp = mat_propag
        for k in range(1,in_longest_path):
            mat_model += mat_tmp
            mat_tmp @= mat_propag
        mat_model += mat_tmp

        print(mat_model)

        return mat_model

    def _single_iteration(self, mat_model=None, vec_c1=None, vec_c2=None, vec_c3=None, vec_q_in_t=None, vec_q_out_t=None):
        """Run a single model iteration
        :param in_longest_path:
        :param vec_c1:
        :param vec_c2:
        :param vec_c3:
        :param vec_q_in_t:
        :param vec_q_out_t:
        :return:
        """

        vec_q_out_next = vec_c1 * vec_q_in_t
        vec_q_out_next += vec_c3 * vec_q_out_t
        vec_q_out_next += np.diag(vec_c2) @ (self._mat_n_in @ vec_q_out_t + vec_q_in_t)

        vec_q_out_next = mat_model @ vec_q_out_next

        return vec_q_out_next

    def run(self, n_iter=10, in_longest_path=2, vec_c1=None, vec_c2=None, vec_c3=None, vec_q_0=None, mat_q_in_ts=None):
        """
        :param n_iter:
        :param in_longest_path:
        :param vec_c1:
        :param vec_c2:
        :param vec_c3:
        :param vec_q_0:
        :param mat_q_in_ts:
        :return:
        """

        if vec_q_0.ndim != 1:
            raise ValueError("Initial condition should be a 1D vector")
        if vec_q_0.size != self._n_dim:
            raise ValueError("Initial condition does not match network size")

        if mat_q_in_ts.ndim != 2:
            raise ValueError("Inflow boundary condition should be a 2D vector")
        if mat_q_in_ts.shape[0] != self._n_dim:
            raise ValueError("Inflow boundary condition does not match network size")
        if mat_q_in_ts.shape[1] != n_iter:
            raise ValueError("Inflow boundary condition timeseries does not match number of iteration")

        mat_model = self._prepare_run(in_longest_path=in_longest_path, vec_c1=vec_c1)
        mat_q_out_ts = np.zeros((self._n_dim, n_iter+1))
        mat_q_out_ts[:,0] = vec_q_0

        for il_i in range(n_iter):
            mat_q_out_ts[:,il_i+1] = self._single_iteration(mat_model=mat_model,
                                                            vec_c1=vec_c1,
                                                            vec_c2=vec_c2,
                                                            vec_c3=vec_c3,
                                                            vec_q_in_t=mat_q_in_ts[:,il_i],
                                                            vec_q_out_t=mat_q_out_ts[:,il_i])

        return mat_q_out_ts

if __name__ == "__main__":
    """Main run
    """

    # Set model
    n_network = 5
    mat_network = np.array([[0, 0, 0, 0 ,0],[0, 0, 0, 0 ,0],[1, 1, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 1, 1, 0]])
    in_longest_past = 2
    x = 0.25 * np.ones((n_network,))
    k = 2.5 * np.ones((n_network,))
    d_t = 0.5
    n_iter = 250

    # Set inflow
    mat_q_in_ts = np.ones((n_network,n_iter))
    mat_q_in_ts[0,:] = 20. * mat_q_in_ts[0,:]
    mat_q_in_ts[1, :] = 5.*(1. + np.sinc(np.linspace(start=-1, stop=1, num=n_iter)))
    mat_q_in_ts[3, 32:45] = np.array([15., 10., 8., 6., 5., 4., 3., 2.5, 2, 1., 1., 1., 1.])

    fig_in, axis_in = plt.subplots(1,5, sharey=True)
    fig_in.suptitle("Inflow")
    for il_i in range(n_network):
        axis_in[il_i].plot(mat_q_in_ts[il_i,:])
        axis_in[il_i].set_title(f"Inflow {il_i+1}")


    # Set initial condition
    vec_q_t0 = np.array([20., 1., 22., 1., 24.])

    # Run model
    my_model = RiverModel(n_dim=n_network,
                          mat_n_in=mat_network,
                          vec_k_in=k,
                          vec_x_in=x)
    vec_c1 = my_model.set_c1(d_t)
    vec_c2 = my_model.set_c2(d_t)
    vec_c3 = my_model.set_c3(d_t)

    mat_q_out_ts = my_model.run(n_iter=n_iter,
                                in_longest_path=in_longest_past,
                                vec_c1=vec_c1,
                                vec_c2=vec_c2,
                                vec_c3=vec_c3,
                                vec_q_0=vec_q_t0,
                                mat_q_in_ts=mat_q_in_ts)

    fig_out, axis_out = plt.subplots(1,5, sharey=True)
    fig_out.suptitle("Outflow")
    for il_i in range(n_network):
        axis_out[il_i].plot(mat_q_out_ts[il_i,:])
        axis_out[il_i].set_title(f"Outflow {il_i+1}")

    plt.draw()
    plt.show()






