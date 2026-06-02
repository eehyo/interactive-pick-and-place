import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from scipy.spatial.distance import cdist


def get_idxs(query_list, domain_list):
    if not isinstance(query_list, list) or not isinstance(domain_list, list):
        raise TypeError("[get_idxs] Both inputs should be lists.")
    return [query_list.index(item) for item in domain_list if item in query_list]


def get_items_by_prefix(items, prefix):
    if not isinstance(items, list):
        raise TypeError("[get_items_by_prefix] items should be a list.")
    if not isinstance(prefix, str):
        raise TypeError("[get_items_by_prefix] prefix should be a string.")
    return sorted([item for item in items if item.startswith(prefix)])


def group_by_prefix(items, prefix_dict, rest_key="rest", verbose=False):
    if not isinstance(items, list):
        raise TypeError("[group_by_prefix] items should be a list.")
    if not isinstance(prefix_dict, dict):
        raise TypeError("[group_by_prefix] prefix_dict should be a dict.")

    grouped = {key: [] for key in prefix_dict.keys()}
    grouped[rest_key] = []
    assigned = set()
    for key, prefixes in prefix_dict.items():
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        for item in items:
            if item in assigned:
                continue
            if any(item.startswith(prefix) for prefix in prefixes):
                grouped[key].append(item)
                assigned.add(item)
    for item in items:
        if item not in assigned:
            grouped[rest_key].append(item)
    for key in grouped:
        grouped[key] = sorted(grouped[key])
    if verbose:
        for key, vals in grouped.items():
            print("[%s] %s" % (key, vals))
    return grouped


def indent_xml(elem):
    """
    Convert an XML Element into a neatly formatted, indented XML string.

    Parameters:
        elem (xml.etree.ElementTree.Element): XML element to format.

    Returns:
        str: Pretty-printed XML string without empty lines.
    """
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    # Remove empty lines
    lines = [line for line in pretty_xml.splitlines() if line.strip()]
    return "\n".join(lines)


def get_monitor_size():
    try:
        import pyautogui
        w, h = pyautogui.size()
        return int(w), int(h)
    except Exception:
        return 1920, 1080


class TicToc(object):
    def __init__(self, name="tictoc"):
        self.name = name
        self.time_start = time.time()
        self.time_elapsed = 0.0
        self.cnt = 0

    def tic(self):
        self.time_start = time.time()

    def toc(self, label=None, cnt=None, print_every=None, verbose=False):
        self.time_elapsed = time.time() - self.time_start
        if print_every is not None and verbose:
            if cnt is not None:
                self.cnt = cnt
            if (self.cnt % print_every) == 0:
                prefix = self.name if label is None else label
                print("%s Elapsed time:[%.3f]s" % (prefix, self.time_elapsed))
        self.cnt += 1
        return self.time_elapsed


def trim_scale(x, th):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    if n > float(th) and n > 0:
        return x / n * float(th)
    return x


def sample_xyzs(
        n_sample=1,
        x_range=(-1, -1),
        y_range=(-1, +1),
        z_range=(0, 0),
        min_dist=0.1,
        xy_margin=0.0,
        seed=None,
        return_ranges=False,
    ):
    rng = np.random.default_rng(seed)
    ranges = {
        "x_range": (x_range[0] + xy_margin, x_range[1] - xy_margin),
        "y_range": (y_range[0] + xy_margin, y_range[1] - xy_margin),
        "z_range": (z_range[0], z_range[1]),
    }
    xyzs = np.zeros((int(n_sample), 3))
    for p_idx in range(int(n_sample)):
        while True:
            xyz = np.array([
                rng.uniform(*ranges["x_range"]),
                rng.uniform(*ranges["y_range"]),
                rng.uniform(*ranges["z_range"]),
            ])
            if p_idx == 0:
                break
            devc = cdist(xyz.reshape((-1, 3)), xyzs[:p_idx, :].reshape((-1, 3)), "euclidean")
            if devc.min() > min_dist:
                break
        xyzs[p_idx, :] = xyz
    if return_ranges:
        return xyzs, ranges
    return xyzs


def xy2xyz(xy, z=0.0):
    xy = np.asarray(xy, dtype=np.float64)
    return np.array([xy[0], xy[1], float(z)])


def farthest_point_sampling(pcd, n_sample):
    pcd = np.asarray(pcd)
    assert pcd.ndim == 2, "Input pcd must be a 2D array."
    N = pcd.shape[0]
    if n_sample >= N:
        additional_indices = np.random.choice(N, n_sample - N, replace=True)
        return np.concatenate([np.arange(N, dtype=np.int32), additional_indices])

    sampled_indices = [np.random.randint(0, N)]
    distances = np.ones(N) * 1e7
    for _ in range(n_sample - 1):
        last_sampled_point = pcd[sampled_indices[-1]].reshape(1, -1)
        current_distances = np.linalg.norm(pcd - last_sampled_point, axis=1)
        distances = np.minimum(distances, current_distances)
        sampled_indices.append(int(np.argmax(distances)))
    return np.array(sampled_indices, dtype=np.int32)


class SimpleTimer(object):
    def __init__(self, name="timer", Hz=10, max_sec=np.inf, verbose=True):
        if Hz <= 0:
            raise ValueError("Hz must be positive.")
        self.name = name
        self.Hz = float(Hz)
        self.dt = 1.0 / self.Hz
        self.max_sec = max_sec
        self.verbose = bool(verbose)
        self.sec_period = 1.0 / self.Hz
        self.sec_next = 0.0
        self.sec_elps = 0.0
        self.sec_elps_prev = 0.0
        self.sec_elps_diff = 0.0
        self.sec_elps_loop = None
        self.sec_elps_loop_prev = None
        self.tick = 0
        self.force_finish = False
        self.delayed_flag = False
        self.time_start = None
        self.time_run_start = None

    def start(self):
        self.time_start = time.perf_counter()
        self.time_run_start = None
        self.sec_next = 0.0
        self.sec_elps = 0.0
        self.sec_elps_prev = 0.0
        self.sec_elps_diff = 0.0
        self.sec_elps_loop = None
        self.sec_elps_loop_prev = None
        self.tick = 0
        self.force_finish = False
        self.delayed_flag = False

    def finish(self):
        self.force_finish = True

    def is_finished(self):
        if self.force_finish:
            return True
        self.sec_elps = time.perf_counter() - self.time_start
        return self.sec_elps > self.max_sec

    def is_notfinished(self):
        return not self.is_finished()

    def do_run(self):
        time.sleep(1e-6)
        self.sec_elps = time.perf_counter() - self.time_start
        if self.sec_elps > self.sec_next:
            self.sec_next += self.sec_period
            self.tick += 1
            self.sec_elps_diff = self.sec_elps - self.sec_elps_prev
            self.sec_elps_prev = self.sec_elps
            self.delayed_flag = self.sec_elps_diff > self.sec_period * 1.1
            self.time_run_start = time.perf_counter()
            return True
        self.delayed_flag = False
        return False

    def end(self):
        if self.time_run_start is None:
            return None
        self.sec_elps_loop_prev = self.sec_elps_loop
        self.sec_elps_loop = time.perf_counter() - self.time_run_start
        return self.sec_elps_loop

    def get_elapsed_run_time(self, unit="s", default=0.0):
        sec_elps_loop = self.sec_elps_loop
        if sec_elps_loop is None:
            sec_elps_loop = self.sec_elps_loop_prev
        if sec_elps_loop is None:
            return default
        if unit == "s":
            return sec_elps_loop
        if unit == "ms":
            return sec_elps_loop * 1000.0
        if unit == "us":
            return sec_elps_loop * 1e6
        raise ValueError("Unsupported unit: %s" % unit)

    def get_ems(self):
        return self.get_elapsed_run_time("ms")
