import numpy as np
from scipy.spatial import cKDTree

from flatspin.callback import Callback


class NucleateRandom(Callback):
    """
    Callback to nucleate a random pattern at a random location in the model.
    """

    def __init__(
        self,
        nucleate_size=(4, 4),
        nucleate_spin_prob=0.5,
        nucleate_bounds=None,
        nucleate_every=1,
        **kwargs
    ):
        super().__init__(
            nucleate_size=nucleate_size,
            nucleate_spin_prob=nucleate_spin_prob,
            nucleate_bounds=nucleate_bounds,
            nucleate_every=nucleate_every,
            **kwargs
        )

    def on_sample(self, i, model, result):
        if i % self.nucleate_every != 0:
            return False
        pattern_size = self.nucleate_size
        pattern = (np.random.rand(*pattern_size) < self.nucleate_spin_prob) * 2 - 1
        maxs = model.labels.max(axis=0)
        mins = model.labels.min(axis=0)

        # example: nucleate_bounds = np.array([[0, 0], [10, 10]]) to nucleate in a 10x10 box starting at (0,0)
        # print(f"nucleate_bounds: {self.nucleate_bounds}")
        if self.nucleate_bounds is not None:
            mins = np.maximum(mins, self.nucleate_bounds[0])
            maxs = np.minimum(maxs, self.nucleate_bounds[1])
            # print(f"updated mins: {mins}, maxs: {maxs}")
        loc = np.random.randint(mins, maxs - pattern_size + 1)
        model.spin[
            model.L[
                loc[0] : loc[0] + pattern_size[0], loc[1] : loc[1] + pattern_size[1]
            ]
        ] = pattern.flatten()


class KillBiggestRegion(Callback):
    """
    Callback to kill the biggest connected region of spins equal to `target_value`.
    """

    def __init__(self, kill_target_value=1, min_kill_size=1, **kwargs):
        super().__init__(kill_target_value=kill_target_value, min_kill_size=min_kill_size, **kwargs)
        self._neighbor_list = None

    def on_sample(self, i, model, result):
        if self._neighbor_list is None:
            self._neighbor_list = init_neighbor_list(model)

        regions = find_regions(self._neighbor_list, model.spin, self.kill_target_value)
        if regions:
            biggest_region = max(regions, key=len)
            if len(biggest_region) >= self.min_kill_size:
                model.spin[biggest_region] = -self.kill_target_value

        return False

class CopyRegion(Callback):
    """
    Callback to copy a random connected region of spins equal to `target_value`
    to a random location in the model.
    """

    def __init__(self, copy_target_value=1, copy_distinct=False, **kwargs):
        super().__init__(copy_target_value=copy_target_value, copy_distinct=copy_distinct, **kwargs)
        self._neighbor_list = None

    def on_sample(self, i, model, result):
        if self._neighbor_list is None:
            self._neighbor_list = init_neighbor_list(model)

        regions = find_regions(self._neighbor_list, model.spin, self.copy_target_value)
        if not regions:
            return False

        if self.copy_distinct:
            # Pick a random group, then a random region within it
            groups = group_regions(regions, model.labels)
            group = groups[np.random.choice(list(groups.keys()))]
            region = group[np.random.randint(len(group))]
        else:
            # Pick a random region from all regions
            region = regions[np.random.randint(len(regions))]
        region_spins = model.spin[region]

        # Get lattice coordinates of the region's spins
        region_labels = model.labels[region]
        region_mins = region_labels.min(axis=0)
        region_maxs = region_labels.max(axis=0)
        region_shape = region_maxs - region_mins + 1

        # Find a random location for the bounding box of the region
        maxs = model.labels.max(axis=0)
        mins = model.labels.min(axis=0)
        loc = np.random.randint(mins, maxs - region_shape + 1)

        # Translate each spin's coordinate by the offset, then place it
        offsets = region_labels - region_mins
        new_coords = offsets + loc

        indices = model.L[new_coords[:, 0], new_coords[:, 1]]
        model.spin[indices] = region_spins

        return False

class FloodFill(Callback):
    """
    Callback to flood fill at a random location in the model.
    """

    def __init__(
        self, flood_fill_value=-1, flood_fill_ndist=1, flood_fill_repeats=1, flood_fill_threshold=0, **kwargs
    ):
        super().__init__(
            flood_fill_value=flood_fill_value,
            flood_fill_ndist=flood_fill_ndist,
            flood_fill_repeats=flood_fill_repeats,
            flood_fill_threshold=flood_fill_threshold,
            **kwargs
        )
        self._neighbor_list = None

    def on_sample(self, i, model, result):
        # print(f"Flood fill at sample {i} with value {self.flood_fill_value}")
        if self._neighbor_list is None:
            self._neighbor_list = init_neighbor_list(model, self.flood_fill_ndist)

        for _ in range(self.flood_fill_repeats):
            start = np.random.randint(model.spin_count)
            FloodFill.flood_fill(
                self._neighbor_list, model.spin, start, self.flood_fill_value, self.flood_fill_threshold
            )

        return False


    @staticmethod
    def flood_fill(neighbor_list, spin, start, fill_value, threshold=0):
        """
        Flood fill connected spins equal to spin[start], starting from
        `start`, setting them to `fill_value`. Modifies `spin` in place,
        but only if the connected region has more than `threshold` spins.
        """
        target_value = spin[start]

        if target_value == fill_value:
            return

        visited = np.zeros(len(spin), dtype=bool)
        stack = [start]
        visited[start] = True
        region = [start]

        while stack:
            current = stack.pop()

            for n in neighbor_list[current]:
                if n == -1:
                    break
                if not visited[n] and spin[n] == target_value:
                    visited[n] = True
                    region.append(n)
                    stack.append(n)

        if len(region) > threshold:
            spin[region] = fill_value

def find_regions(neighbor_list, spin, target_value=1):
    """
    Partition spins equal to `target_value` into connected regions.
    """
    n = len(spin)
    visited = np.zeros(n, dtype=bool)
    regions = []

    candidates = np.flatnonzero(spin == target_value)

    for i in candidates:
        if visited[i]:
            continue

        region = [i]
        visited[i] = True
        stack = [i]

        while stack:
            current = stack.pop()
            for nb in neighbor_list[current]:
                if nb == -1:
                    break
                if not visited[nb] and spin[nb] == target_value:
                    visited[nb] = True
                    region.append(nb)
                    stack.append(nb)

        regions.append(region)

    return regions

def group_regions(regions, labels):
    """
    Group regions by shape (translation-invariant).

    Two regions are considered the same shape if their spin coordinates,
    normalized to the origin, are identical (as sets).

    Parameters
    ----------
    regions : list of list of int
        Spin indices for each region (as returned by find_regions)
    labels : ndarray, shape (n, 2)
        Lattice coordinates of each spin (model.labels)

    Returns
    -------
    groups : dict mapping shape_key -> list of regions
    """
    groups = {}
    for region in regions:
        coords = labels[region]
        normalized = coords - coords.min(axis=0)
        key = frozenset(map(tuple, normalized))
        groups.setdefault(key, []).append(region)
    return groups

def init_neighbor_list(model, ndist=1.8):
        neighbors = []
        num_neighbors = 0

        tree = cKDTree(model.pos)

        nd = model.lattice_spacing * ndist
        nd += 1e-5  # pad to avoid rounding errors

        for i in range(model.spin_count):
            p = model.pos[i]
            n = tree.query_ball_point([p], nd)[0]
            n.remove(i)
            neighbors.append(n)
            num_neighbors = max(num_neighbors, len(n))

        neighbor_list = np.full((model.spin_count, num_neighbors), -1, dtype=np.int32)
        for i, neighs in enumerate(neighbors):
            neighbor_list[i, : len(neighs)] = neighs

        return neighbor_list