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

    def __init__(self, copy_target_value=1, copy_distinct=False, copy_in_bounds=None, copy_out_bounds=None, **kwargs):
        super().__init__(copy_target_value=copy_target_value, copy_distinct=copy_distinct, copy_in_bounds=copy_in_bounds, copy_out_bounds=copy_out_bounds, **kwargs)
        self._neighbor_list = None

    def on_sample(self, i, model, result):
        if self._neighbor_list is None:
            self._neighbor_list = init_neighbor_list(model)

        regions = find_regions(self._neighbor_list, model.spin, self.copy_target_value)
        if not regions:
            return False

        mins, maxs = model.labels.min(axis=0), model.labels.max(axis=0)

        # Determine destination bounds
        if self.copy_out_bounds is not None:
            out_bounds = normalize_bounds(self.copy_out_bounds)
        else:
            out_bounds = [(mins, maxs)]

        if self.copy_in_bounds is not None:
            in_bounds = normalize_bounds(self.copy_in_bounds)
        else:
            in_bounds = [(mins, maxs)]

        regions_to_copy = []
        for bound in in_bounds:
            source_regions = find_regions_bounded(self._neighbor_list, model.spin, model.labels, [bound], self.copy_target_value)
            if source_regions:
                regions_to_copy.append(pick_region(source_regions, self.copy_distinct))

        for region in regions_to_copy:
            random_place_region(model, region, out_bounds)


        return False
def normalize_bounds(bounds):
    """
    Normalize bounds to a list of (mins, maxs) pairs.
    Accepts either [mins, maxs] or [[mins1, maxs1], ..., [minsN, maxsN]].
    """
    bounds = np.array(bounds)
    if bounds.ndim == 2:
        # single [mins, maxs]
        return [bounds]
    # list of [mins, maxs]
    return list(bounds)

def random_place_region(model, region, out_bounds):
    """
    Place a region at a random location within a randomly chosen out_bound
    that can fit the region's bounding box. Does nothing if no bound fits.
    """
    region_mins, _, region_shape = region_bounding_box(region, model.labels)

    fitting = [
        (b_mins, b_maxs) for b_mins, b_maxs in out_bounds
        if np.all(b_maxs - b_mins >= region_shape - 1)
    ]
    if not fitting:
        return

    dst_mins, dst_maxs = fitting[np.random.randint(len(fitting))]
    loc = np.random.randint(dst_mins, dst_maxs - region_shape + 1)
    place_region(model, region, region_mins, loc)

def place_region(model, region, region_mins, loc):
    """Copy region spins to a new location in the model."""
    region_labels = model.labels[region]
    offsets = region_labels - region_mins
    new_coords = offsets + loc
    model.spin[model.L[new_coords[:, 0], new_coords[:, 1]]] = model.spin[region]

def pick_region(regions, distinct=False):
    """Pick a random region, optionally from distinct shapes only."""
    if distinct:
        groups = group_regions(regions)
        group = groups[np.random.choice(list(groups.keys()))]
        return group[np.random.randint(len(group))]
    return regions[np.random.randint(len(regions))]

def region_bounding_box(region, labels):
    """Return (mins, maxs, shape) of a region's bounding box in label space."""
    region_labels = labels[region]
    mins = region_labels.min(axis=0)
    maxs = region_labels.max(axis=0)
    return mins, maxs, maxs - mins + 1

def bounds_to_mask(labels, bounds):
    """
    Return a boolean mask of spins within any of the given bounds.

    Parameters
    ----------
    labels : ndarray, shape (n, 2)
    bounds : list of (mins, maxs) pairs
    """
    mask = np.zeros(len(labels), dtype=bool)
    for b_mins, b_maxs in bounds:
        in_bounds = np.all(labels >= b_mins, axis=1) & np.all(labels <= b_maxs, axis=1)
        mask |= in_bounds
    return mask

def find_regions_bounded(neighbor_list, spin, labels, bounds, target_value=1):
    """
    Find connected regions of `target_value` spins, restricted to
    spins within the given bounds.

    Parameters
    ----------
    labels : ndarray, shape (n, 2)
    bounds : list of (mins, maxs) pairs
    """
    mask = bounds_to_mask(labels, bounds)
    return find_regions_masked(neighbor_list, spin, mask, target_value)

def find_regions_masked(neighbor_list, spin, mask, target_value=1):
    """
    Find connected regions of `target_value` spins, restricted to
    spins where mask is True.

    Parameters
    ----------
    mask : ndarray of bool, shape (n,)
    """
    masked_spin = spin.copy()
    masked_spin[~mask] = -target_value
    return find_regions(neighbor_list, masked_spin, target_value)


def group_regions(regions):
    """
    Group regions by their normalized shape in index space.

    """
    groups = {}
    for region in regions:
        region = np.array(region)
        normalized = region - region.min(axis=0)
        key = frozenset(normalized)
        groups.setdefault(key, []).append(region)
    return groups
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