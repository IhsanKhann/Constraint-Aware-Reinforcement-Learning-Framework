"""
tilecoding.py
=============
Tile Coding — continuous state → discrete hash index.

Why Tile Coding?
----------------
Tabular Q-learning requires a discrete state space. Our spacecraft state is
continuous (angles, angular rates). Tile coding divides each continuous
dimension into overlapping grids ("tilings") and assigns each grid cell a
unique integer index. Multiple overlapping tilings give smooth generalisation
similar to function approximation, but without neural networks.

How It Works
------------
1. Define N_TILINGS overlapping grids per dimension.
2. Each tiling is offset by a fraction of the tile width.
3. The current state value is mapped to a tile index in each tiling.
4. All tile indices are combined into a single hash-based integer.

This reduces memory compared to a dense lookup table:
    Dense table: dimensions^tiles cells
    Tile coded:  N_TILINGS * TILES_PER_DIM * N_DIM cells  (much smaller)

Example
-------
    coder = TileCoder(
        state_low  = np.array([-π, -π, -π, -5, -5, -5]),
        state_high = np.array([ π,  π,  π,  5,  5,  5]),
        n_tilings  = 8,
        tiles_per_dim = 10
    )
    idx = coder.encode(state)   # → single integer index

References
----------
Sutton & Barto, "Reinforcement Learning: An Introduction", Ch. 9.5
"""

import numpy as np


class TileCoder:
    """
    Hash-based tile coder for continuous state spaces.

    Parameters
    ----------
    state_low    : array-like, shape (n_dims,) — lower bound for each state dim
    state_high   : array-like, shape (n_dims,) — upper bound for each state dim
    n_tilings    : int — number of overlapping tilings (8 is a good default)
    tiles_per_dim: int — number of tiles along each dimension in one tiling
    memory_size  : int — hash table size (controls collision rate)
    """

    def __init__(self, state_low, state_high, n_tilings=8, tiles_per_dim=10,
                 memory_size=4096):
        self.state_low     = np.asarray(state_low,  dtype=float)
        self.state_high    = np.asarray(state_high, dtype=float)
        self.n_tilings     = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.memory_size   = memory_size
        self.n_dims        = len(state_low)

        # Tile width in normalised [0, 1] space for each dimension
        # We divide [0, 1] into tiles_per_dim tiles, so width = 1/tiles_per_dim
        self.tile_width = 1.0 / tiles_per_dim

        # Offsets for each tiling: evenly spaced fraction of tile width
        # Example: 8 tilings → offsets are [0, 1/8, 2/8, ..., 7/8] × tile_width
        self.offsets = np.array([
            [i / n_tilings * self.tile_width for _ in range(self.n_dims)]
            for i in range(n_tilings)
        ])  # shape: (n_tilings, n_dims)

        # Total number of features = n_tilings * memory_size
        # Each tiling contributes one active feature
        self.total_features = n_tilings * memory_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, state):
        """
        Map a continuous state to a single discrete integer index.

        This is the key function: it converts a 6-dimensional continuous
        spacecraft state into one integer that indexes into the Q-table.

        Process:
            1. Clip state to valid range
            2. Normalise to [0, 1]
            3. For each tiling, shift by offset and compute tile index
            4. Hash all tile indices into a single integer

        Parameters
        ----------
        state : array-like, shape (n_dims,)

        Returns
        -------
        idx : int in [0, total_features)
        """
        state = np.clip(np.asarray(state, dtype=float),
                        self.state_low, self.state_high)

        # Normalise to [0, 1]
        norm_state = (state - self.state_low) / (self.state_high - self.state_low + 1e-8)

        # Accumulate hash across all tilings
        hash_val = 0
        for t in range(self.n_tilings):
            # Shift state by this tiling's offset
            shifted = norm_state + self.offsets[t]

            # Compute tile index along each dimension
            tile_indices = (shifted / self.tile_width).astype(int)
            tile_indices = np.clip(tile_indices, 0, self.tiles_per_dim)

            # Hash the tile indices for this tiling to a single integer
            tile_hash = self._hash_tiles(tile_indices, t)

            # Combine into the overall hash (XOR for mixing)
            hash_val ^= tile_hash

        return hash_val % self.total_features

    def encode_multi(self, state):
        """
        Return a list of one active tile index per tiling.

        Alternative encoding used when you want to update all active
        tiles simultaneously (standard tile-coded Q-learning).

        Returns
        -------
        active_tiles : list of ints, length n_tilings
            One index per tiling, each in [0, memory_size).
        """
        state = np.clip(np.asarray(state, dtype=float),
                        self.state_low, self.state_high)
        norm_state = (state - self.state_low) / (self.state_high - self.state_low + 1e-8)

        active = []
        for t in range(self.n_tilings):
            shifted     = norm_state + self.offsets[t]
            tile_indices = (shifted / self.tile_width).astype(int)
            tile_indices = np.clip(tile_indices, 0, self.tiles_per_dim)
            h = self._hash_tiles(tile_indices, t) % self.memory_size
            active.append(t * self.memory_size + h)   # global index for this tiling

        return active

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_tiles(self, tile_indices, tiling_id):
        """
        Combine tile indices and tiling ID into a single hash integer.

        Uses a polynomial rolling hash (similar to Python's hash for tuples)
        to spread indices evenly across the memory space.

        Parameters
        ----------
        tile_indices : np.ndarray, shape (n_dims,)
        tiling_id    : int

        Returns
        -------
        h : int
        """
        h = tiling_id * 2654435761   # Knuth's multiplicative hash seed per tiling
        for i, idx in enumerate(tile_indices):
            h += int(idx) * (1 + i * 12345)   # prime-weighted dimension contribution
        return abs(h)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np

    state_low  = np.array([-np.pi, -np.pi, -np.pi, -5.0, -5.0, -5.0])
    state_high = np.array([ np.pi,  np.pi,  np.pi,  5.0,  5.0,  5.0])

    coder = TileCoder(state_low, state_high, n_tilings=8, tiles_per_dim=10)

    test_state = np.array([0.1, -0.2, 0.3, 0.01, -0.05, 0.02])
    idx  = coder.encode(test_state)
    multi = coder.encode_multi(test_state)

    print(f"State:        {test_state}")
    print(f"Encoded idx:  {idx}  (out of {coder.total_features})")
    print(f"Multi-tiles:  {multi}")

    # Verify locality: nearby states should give same or nearby indices
    nearby = test_state + np.array([0.001, 0, 0, 0, 0, 0])
    idx2   = coder.encode(nearby)
    print(f"Nearby state index: {idx2}  (same={idx == idx2})")
    print("TileCoder OK")