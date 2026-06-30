"""
reservoir_model.py module : a simple linear lumped rainfall-runoff model
"""

class ReservoirModel:
    """A class to simulate a lumped rainfall-runoff 'reservoir-like' model
    """

    def __init__(self, alpha=0.85, beta=0.5, gamma=0.9, s_0=1000., q_0=0.):
        """Class constructor
        """

        # Initial condition
        if s_0 < 0.:
            raise ValueError(f"Input s_0 value must be >= 0, got {s_0}")
        self._s_0 = s_0             # Initial storage in the reservoir [m3]

        if q_0 < 0.:
            raise ValueError(f"Input q_0 value must be >= 0, got {q_0}")
        self._q_0 = q_0             # Initial discharge [m3/s]

        # Model parameters
        if alpha < 0. or alpha > 1.:
            raise ValueError(f"Input alpha value must be in [0,1], got {alpha}")
        self._alpha = alpha       # Storage loss coefficient in [0,1]

        if beta <= 0.:
            raise ValueError(f"Input beta value must be > 0., got {beta}")
        self._beta = beta         # Rainfall-to-storage gain coefficient >0

        if gamma <= 0.:
            raise ValueError(f"Input gamma value must be > 0., got {gamma}")
        self._gamma = gamma       # Storage-to-discharge conversion coefficient >0