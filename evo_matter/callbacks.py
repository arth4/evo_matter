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
        if i % self.nucleate_every != 0 and self.nucleate_every > 1:
            return False
        pattern_size = self.nucleate_size
        maxs = model.labels.max(axis=0)
        mins = model.labels.min(axis=0)

        repeats = int(np.round(1 / self.nucleate_every)) if self.nucleate_every < 1 else 1

        for _ in range(repeats):
            pattern = (np.random.rand(*pattern_size) < self.nucleate_spin_prob) * 2 - 1
            # example: nucleate_bounds = np.array([[0, 0], [10, 10]]) to nucleate in a 10x10 box starting at (0,0)
            # print(f"nucleate_bounds: {self.nucleate_bounds}")
            if self.nucleate_bounds is not None:
                self.nucleate_bounds = np.array(self.nucleate_bounds)
                #check if any bounds are fractional, and if so, convert to absolute bounds
                if np.any((self.nucleate_bounds > 0) & (self.nucleate_bounds < 1)):
                    self.nucleate_bounds = np.array([
                        mins + (maxs - mins) * self.nucleate_bounds[0],
                        mins + (maxs - mins) * self.nucleate_bounds[1]
                    ]).astype(int)
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

class ShootingRange(Callback):
    """
    Callback to implement a shooting range environment.
    """

    def __init__(self, shooting_n_rows=10, shooting_zone_frac=0.5, shooting_target_frac=0.25, **kwargs):
        super().__init__(shooting_n_rows=shooting_n_rows, shooting_zone_frac=shooting_zone_frac, shooting_target_frac=shooting_target_frac, **kwargs)
        self._row_fitness = None

    def _init_shooting_range(self, model):
        self._row_fitness = np.zeros(self.shooting_n_rows)
        maxs, mins = model.labels.max(axis=0), model.labels.min(axis=0)
        row_ys = np.linspace(mins[0], maxs[0], self.shooting_n_rows + 1).astype(np.int32)

        shoot_x = int(self.shooting_zone_frac * (maxs[1] - mins[1]))
        target_x = int((1 - self.shooting_target_frac) * (maxs[1] - mins[1]))

        self._target_zones = [
            ((row_ys[i], row_ys[i + 1] + 1), (target_x, None)) for i in range(self.shooting_n_rows)
        ]
        self._shoot_zones = [
            ((row_ys[i], row_ys[i + 1] + 1), (None, shoot_x)) for i in range(self.shooting_n_rows)
        ]

        self._middle_zones = [
            ((row_ys[i], row_ys[i + 1] + 1), (shoot_x, target_x)) for i in range(self.shooting_n_rows)
        ]


    def on_sample(self, i, model, result):
        if self._row_fitness is None:
            self._init_shooting_range(model)

        on_zones = self.active_targets(model)
        if not on_zones:
            return False

        source_row = np.random.choice(on_zones)
        self._row_fitness += 1
        self._row_fitness[source_row] = 0

        dest_row = np.argmax(self._row_fitness)
        self._row_fitness[dest_row] = 0

        self.clear_zone(model, self._target_zones[source_row])
        self.clear_zone(model, self._middle_zones[source_row])

        self.copy_row(model, source_row, dest_row)

        return False

    def clear_zone(self, model, zone):
        model.spin[model.L[zone[0][0]:zone[0][1], zone[1][0]:zone[1][1]]] = -1

    def copy_row(self, model, source_row, dest_row):
        source_zone = self._target_zones[source_row]
        dest_zone = self._target_zones[dest_row]

        source_spins = model.spin[model.L[source_zone[0][0]:source_zone[0][1], :]]
        model.spin[model.L[dest_zone[0][0]:dest_zone[0][1], :]] = source_spins

    def active_targets(self, model):
        return [ i for i in range(self.shooting_n_rows)
            if np.any(model.spin[model.L[self._target_zones[i][0][0]:self._target_zones[i][0][1],
                                         self._target_zones[i][1][0]:self._target_zones[i][1][1]]
                                ] == 1)
        ]

class RowActivity(Callback):
    """
    Callback to implement an active row environment.
    """

    def __init__(self, row_activity_n_rows=10, row_activity_timestep=4,**kwargs):
        super().__init__(row_activity_n_rows=row_activity_n_rows, row_activity_timestep=row_activity_timestep,
                        row_activity_ignore_edge=4, **kwargs)
        self._rows = None
        self._x_range = None

    def _init_rows(self, model):
        maxs, mins = model.labels.max(axis=0), model.labels.min(axis=0)
        row_ys = np.linspace(mins[0], maxs[0], self.row_activity_n_rows + 1).astype(np.int32)

        self._rows = [
            ((row_ys[i], row_ys[i + 1] + 1), (None, None)) for i in range(self.row_activity_n_rows)
        ]
        self._x_range = (mins[1] + self.row_activity_ignore_edge, maxs[1] - self.row_activity_ignore_edge)

    def on_sample(self, i, model, result):
        if self._rows is None:
            self._init_rows(model)

        active_rows, frozen_rows = self.active_frozen_rows(model, result)
        if not frozen_rows:
            return False

        for frozen_row in frozen_rows:
            if len(active_rows) < 2:
                self.randomize_row(model, frozen_row)
            else:
                row1, row2 = np.random.choice(active_rows, 2, replace=False)
                self.row_crossover(model, row1, row2, frozen_row)

        return False

    def active_frozen_rows(self, model, result):
        if len(result["spin"]) <= self.row_activity_timestep: # not enough history to determine activity
            return None, None

        this_step = result["spin"][-1]
        last_step = result["spin"][-self.row_activity_timestep - 1]
        active_rows = []
        frozen_rows = []

        for i, row in enumerate(self._rows):
            this_row = this_step[model.L[row[0][0]:row[0][1], self._x_range[0]:self._x_range[1]]]
            last_row = last_step[model.L[row[0][0]:row[0][1], self._x_range[0]:self._x_range[1]]]

            if np.any(this_row != last_row):
                active_rows.append(i)
            else:
                frozen_rows.append(i)

        return active_rows, frozen_rows


    def row_crossover(self, model, row1, row2, dest_row):
        row1_zone = self._rows[row1]
        row2_zone = self._rows[row2]
        dest_zone = self._rows[dest_row]

        # use min height to handle rows of different sizes
        h = min(
            row1_zone[0][1] - row1_zone[0][0],
            row2_zone[0][1] - row2_zone[0][0],
            dest_zone[0][1] - dest_zone[0][0],
        )

        crossover_point = np.random.randint(self._x_range[0], self._x_range[1])

        model.spin[model.L[dest_zone[0][0]:dest_zone[0][0]+h, :]] =                 model.spin[model.L[row1_zone[0][0]:row1_zone[0][0]+h, :]]
        model.spin[model.L[dest_zone[0][0]:dest_zone[0][0]+h, crossover_point:]] =  model.spin[model.L[row2_zone[0][0]:row2_zone[0][0]+h, crossover_point:]]

    def randomize_row(self, model, row):
        indices = model.L[self._rows[row][0][0]:self._rows[row][0][1], :]
        model.spin[indices] = np.random.choice([-1, 1], p=[0.8, 0.2], size=indices.shape)

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
        if np.all(b_maxs - b_mins >= region_shape)
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