#!/usr/bin/env python3
"""Browse the slices of an MHD volume and pick an ISO threshold for mesh extraction.

The window shows the current X-Y slice, the same slice with an ISO threshold applied, and the
slice's grey value histogram with the threshold marked. Two sliders navigate: one along Z, one
along the grey value axis. The chosen threshold is printed when the window is closed, so it can be
handed straight to the mesh extractor.

Slices are read one at a time straight from the raw file, so the volume is never loaded as a whole
and a scan far larger than memory can be inspected. Only the handful of slices last looked at are
kept, which is what makes dragging the Z slider feel immediate without giving up the streaming.

The MHD subset understood here is the one MeshExtractor's own reader accepts (VolumeIO/
SliceChunkedMHDIO.cpp): a three-dimensional image of one of the eight supported element types,
whose voxels live in a separate raw file named by ElementDataFile, optionally behind a HeaderSize
many bytes of foreign header.

Requires numpy and matplotlib with a GUI backend, e.g.
    python3 -m venv .venv && .venv/bin/pip install numpy matplotlib PyQt5
"""

import argparse
import collections
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("error: this tool needs numpy: python3 -m pip install numpy matplotlib")


# The MetaIO element types the MeshExtractor volume reader supports, mapped to the numpy dtype
# describing the same voxel. The byte order is filled in once the header has been parsed.
ELEMENT_TYPES = {
    "MET_CHAR":   "i1",
    "MET_UCHAR":  "u1",
    "MET_SHORT":  "i2",
    "MET_USHORT": "u2",
    "MET_INT":    "i4",
    "MET_UINT":   "u4",
    "MET_FLOAT":  "f4",
    "MET_DOUBLE": "f8",
}

# How many slices to keep around. The slider hands out one slice at a time, so this only has to
# cover stepping back and forth over a small neighbourhood; it is deliberately small because a
# slice of a large scan is tens of megabytes.
SLICE_CACHE_SIZE = 8

# How many slices to sample when establishing the display window, the histogram range and the span
# of the threshold slider. Reading every slice for that would defeat the point of streaming.
SAMPLE_SLICES = 8

# Bins in the histogram. Enough to resolve the material peaks, few enough to recompute per slice.
HISTOGRAM_BINS = 256


class MHDError(Exception):
    """An MHD file that cannot be read for display."""


def parse_mhd(mhd_path):
    """Return the meta data of the MHD file at mhd_path as a plain dict.

    Lines are "key = value" pairs, values further split on whitespace -- the same shape the C++
    parser expects, so a file rejected here would be rejected there as well.
    """
    meta = {}
    with open(mhd_path, "r") as mhd_file:
        for line in mhd_file:
            if not line.strip():
                continue
            if "=" not in line:
                raise MHDError("Malformed line without '=': {!r}".format(line.strip()))
            key, value = line.split("=", 1)
            meta[key.strip().strip('"')] = value.strip().strip('"')
    return meta


class MHDVolume:
    """A volume read slice by slice from the raw file an MHD points at."""

    def __init__(self, mhd_path):
        meta = parse_mhd(mhd_path)

        if meta.get("ObjectType", "Image") != "Image":
            raise MHDError("Unrecognized ObjectType {!r}.".format(meta["ObjectType"]))
        if int(meta.get("NDims", 3)) != 3:
            raise MHDError("Only three-dimensional images can be browsed slice by slice.")
        if meta.get("CompressedData", "False").lower() == "true":
            raise MHDError("Compressed MHD data is not supported; the voxels have to be raw.")
        if int(meta.get("ElementNumberOfChannels", 1)) != 1:
            raise MHDError("Only single channel volumes are supported.")

        for required in ("DimSize", "ElementType", "ElementDataFile"):
            if required not in meta:
                raise MHDError("Header is missing the required key {}.".format(required))

        element_type = meta["ElementType"]
        if element_type not in ELEMENT_TYPES:
            raise MHDError("Element type {} is not one MeshExtractor reads.".format(element_type))

        data_file = meta["ElementDataFile"]
        if data_file.upper() == "LOCAL" or "%" in data_file:
            raise MHDError(
                "ElementDataFile {!r} does not name a separate raw file; only an MHD pointing at "
                "one raw volume is supported.".format(data_file)
            )

        # MetaIO states its dimensions as X Y Z, with X running fastest in the file.
        self.dim = tuple(int(value) for value in meta["DimSize"].split()[:3])
        if any(extent <= 0 for extent in self.dim):
            raise MHDError("Header declares a non-positive volume size {}.".format(self.dim))

        self.spacing = tuple(float(v) for v in meta.get("ElementSpacing", "1 1 1").split()[:3])
        self.origin = tuple(float(v) for v in meta.get("Offset", "0 0 0").split()[:3])
        self.element_type = element_type

        # MetaIO can declare big endian voxels; MeshExtractor's reader always reads little endian,
        # so a file that says otherwise is displayed as declared but flagged as one the extractor
        # would interpret differently.
        big_endian = (meta.get("ElementByteOrderMSB", meta.get("BinaryDataByteOrderMSB", "False"))
                      .lower() == "true")
        self.dtype = np.dtype((">" if big_endian else "<") + ELEMENT_TYPES[element_type])
        if big_endian:
            print("warning: header declares big endian voxels, which MeshExtractor's reader does "
                  "not honour; its extraction would differ from this view.", file=sys.stderr)

        # Bytes of foreign header to skip before the voxels start -- a .rek header, typically.
        self.header_size = int(meta.get("HeaderSize", 0))

        # The raw file is named relative to the MHD, not to the working directory, matching how
        # the C++ reader resolves it.
        self.data_path = (data_file if os.path.isabs(data_file)
                          else os.path.join(os.path.dirname(os.path.abspath(mhd_path)), data_file))
        if not os.path.exists(self.data_path):
            raise MHDError("Volume data file {} does not exist.".format(self.data_path))

        self.slice_pixels = self.dim[0] * self.dim[1]
        self.slice_bytes = self.slice_pixels * self.dtype.itemsize
        expected = self.slice_bytes * self.dim[2] + self.header_size
        actual = os.path.getsize(self.data_path)
        if actual != expected:
            raise MHDError(
                "Header describes {} bytes but {} holds {}.".format(expected, self.data_path, actual)
            )

        self.data_file = open(self.data_path, "rb")
        self.cache = collections.OrderedDict()

    def close(self):
        self.data_file.close()

    def read_slice(self, index):
        """Return slice index as a (Y, X) array, reading it from disk unless it is still cached."""
        if index in self.cache:
            # Refresh its position so the slices around the slider survive the eviction below.
            self.cache.move_to_end(index)
            return self.cache[index]

        self.data_file.seek(self.header_size + index * self.slice_bytes)
        raw = self.data_file.read(self.slice_bytes)
        if len(raw) != self.slice_bytes:
            raise MHDError("Short read of slice {}; the file ended early.".format(index))

        # frombuffer views the bytes in the file's own type; the copy into float32 both widens the
        # voxels the way the extractor does and detaches the array from the transient buffer.
        data = np.frombuffer(raw, dtype=self.dtype).astype(np.float32).reshape(self.dim[1], self.dim[0])

        self.cache[index] = data
        while len(self.cache) > SLICE_CACHE_SIZE:
            self.cache.popitem(last=False)
        return data

    def sample_range(self, stride):
        """Return (minimum, maximum, low_percentile, high_percentile) over a few sampled slices.

        The percentiles give a display window that is not ruined by a handful of hot voxels, while
        the true extrema bound the threshold slider so every grey value in the volume stays
        reachable.
        """
        indices = np.unique(np.linspace(0, self.dim[2] - 1, min(SAMPLE_SLICES, self.dim[2])).astype(int))
        samples = [self.read_slice(int(index))[::stride, ::stride] for index in indices]
        stacked = np.concatenate([sample.ravel() for sample in samples])
        finite = stacked[np.isfinite(stacked)]
        if finite.size == 0:
            raise MHDError("The sampled slices hold no finite voxel values.")
        low, high = np.percentile(finite, [0.5, 99.5])
        return float(finite.min()), float(finite.max()), float(low), float(high)


class Viewer:
    """The figure, its widgets, and the state tying them to the volume."""

    def __init__(self, volume, index, iso, stride, vmin, vmax, value_min, value_max):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons, RadioButtons, Slider

        self.volume = volume
        self.stride = stride
        self.index = index
        self.iso = iso
        self.view_mode = "mask"
        self.log_histogram = True

        # The arrow keys step through slices here, so they must not also drive matplotlib's own
        # view history.
        for action in ("keymap.back", "keymap.forward"):
            plt.rcParams[action] = [key for key in plt.rcParams[action] if key not in ("left", "right")]

        self.figure = plt.figure(figsize=(13, 9))
        self.figure.canvas.manager.set_window_title(os.path.basename(volume.data_path))

        data = self.current_slice()

        self.slice_axes = self.figure.add_axes([0.04, 0.42, 0.44, 0.52])
        self.slice_image = self.slice_axes.imshow(data, cmap="gray", vmin=vmin, vmax=vmax,
                                                  interpolation="nearest", origin="upper")
        self.slice_axes.set_title("slice")
        self.slice_axes.set_xticks([])
        self.slice_axes.set_yticks([])

        # Sharing the axes means zooming into a detail on one side shows the same detail on the
        # other, which is the whole point of having them next to each other.
        self.threshold_axes = self.figure.add_axes([0.52, 0.42, 0.44, 0.52],
                                                   sharex=self.slice_axes, sharey=self.slice_axes)
        self.threshold_image = self.threshold_axes.imshow(self.threshold_view(data), cmap="gray",
                                                          vmin=0.0, vmax=1.0,
                                                          interpolation="nearest", origin="upper")
        self.threshold_axes.set_title("above ISO")
        self.threshold_axes.set_xticks([])
        self.threshold_axes.set_yticks([])

        # Fixed bin edges: recomputing them per slice would make the bars jump around and turn the
        # histogram useless for comparing one slice against the next.
        self.bin_edges = np.linspace(value_min, value_max, HISTOGRAM_BINS + 1)
        self.histogram_axes = self.figure.add_axes([0.07, 0.19, 0.72, 0.17])
        counts = self.histogram(data)
        (self.histogram_line,) = self.histogram_axes.plot(self.bin_edges[:-1], counts,
                                                          drawstyle="steps-post", color="0.25")
        self.histogram_fill = None
        self.iso_line = self.histogram_axes.axvline(self.iso, color="tab:red", linewidth=1.2)
        self.histogram_axes.set_xlim(value_min, value_max)
        self.histogram_axes.set_xlabel("grey value")
        self.histogram_axes.set_ylabel("voxels")
        self.histogram_axes.set_yscale("log")
        self.redraw_histogram_fill(counts)

        slice_slider_axes = self.figure.add_axes([0.07, 0.10, 0.80, 0.03])
        self.slice_slider = Slider(slice_slider_axes, "slice (Z)", 0, volume.dim[2] - 1,
                                   valinit=self.index, valstep=1)
        self.slice_slider.on_changed(self.on_slice_changed)

        iso_slider_axes = self.figure.add_axes([0.07, 0.05, 0.80, 0.03])
        self.iso_slider = Slider(iso_slider_axes, "ISO", value_min, value_max, valinit=self.iso)
        self.iso_slider.on_changed(self.on_iso_changed)

        # The buttons sit beside the histogram rather than beside the sliders, because a slider draws
        # its value to its right and that text needs the room.
        mode_axes = self.figure.add_axes([0.83, 0.25, 0.14, 0.11])
        mode_axes.set_title("threshold view", fontsize=8)
        self.mode_buttons = RadioButtons(mode_axes, ("mask", "overlay", "masked grey"), active=0)
        self.mode_buttons.on_clicked(self.on_mode_changed)

        log_axes = self.figure.add_axes([0.83, 0.19, 0.14, 0.05])
        self.log_button = CheckButtons(log_axes, ("log counts",), (self.log_histogram,))
        self.log_button.on_clicked(self.on_log_toggled)

        self.status = self.figure.text(0.07, 0.965, "", fontsize=9, family="monospace")
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_status(data)

    def current_slice(self):
        return self.volume.read_slice(self.index)[::self.stride, ::self.stride]

    def threshold_view(self, data):
        """Return what the right hand image shows for the current mode, as values in [0, 1]."""
        mask = data >= self.iso
        if self.view_mode == "mask":
            return mask.astype(np.float32)

        # Both remaining modes need the grey values scaled into the unit interval the image is
        # drawn with, using the display window so the contrast matches the left hand side.
        vmin, vmax = self.slice_image.get_clim()
        scaled = np.clip((data - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
        if self.view_mode == "masked grey":
            return np.where(mask, scaled, 0.0).astype(np.float32)

        # "overlay": the grey slice tinted where it is above the threshold, which keeps the
        # structure visible and shows what the threshold would cut away at the same time.
        rgb = np.repeat(scaled[:, :, None], 3, axis=2).astype(np.float32)
        rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.array([1.0, 0.25, 0.2], dtype=np.float32)
        return rgb

    def histogram(self, data):
        counts, _ = np.histogram(data, bins=self.bin_edges)
        # A log scale cannot show an empty bin, and clamping to a value below one keeps those bins
        # visibly at the axis floor instead of leaving gaps in the curve.
        return np.maximum(counts, 0.5) if self.log_histogram else counts

    def redraw_histogram_fill(self, counts):
        """Shade the part of the histogram the threshold keeps."""
        if self.histogram_fill is not None:
            self.histogram_fill.remove()
        floor = self.histogram_axes.get_ylim()[0]
        self.histogram_fill = self.histogram_axes.fill_between(
            self.bin_edges[:-1], floor, counts, where=self.bin_edges[:-1] >= self.iso,
            step="post", color="tab:red", alpha=0.25,
        )

    def update_status(self, data):
        above = float(np.count_nonzero(data >= self.iso)) / data.size
        z_position = self.volume.origin[2] + self.index * self.volume.spacing[2]
        self.status.set_text(
            "slice {:>5d} / {:<5d}  z = {:.4f} mm    ISO = {:.6g}    above = {:.2f} %    "
            "slice range [{:.6g}, {:.6g}]".format(
                self.index, self.volume.dim[2] - 1, z_position, self.iso, 100.0 * above,
                float(data.min()), float(data.max()),
            )
        )

    def refresh(self, slice_changed):
        data = self.current_slice()
        if slice_changed:
            self.slice_image.set_data(data)
            counts = self.histogram(data)
            self.histogram_line.set_ydata(counts)
        else:
            counts = self.histogram_line.get_ydata()
        self.threshold_image.set_data(self.threshold_view(data))
        self.iso_line.set_xdata([self.iso, self.iso])
        self.redraw_histogram_fill(counts)
        self.update_status(data)
        self.figure.canvas.draw_idle()

    def on_slice_changed(self, value):
        self.index = int(value)
        self.refresh(slice_changed=True)

    def on_iso_changed(self, value):
        self.iso = float(value)
        self.refresh(slice_changed=False)

    def on_mode_changed(self, label):
        self.view_mode = label
        # A mask is scalar and an overlay is RGB, so the image has to be rebuilt rather than just
        # refilled: its colour mapping differs between the two.
        data = self.current_slice()
        view = self.threshold_view(data)
        # imshow scales the axes to the new image, which would throw away whatever detail the user
        # had zoomed into -- and, because the two images share their axes, on both sides at once.
        limits = (self.threshold_axes.get_xlim(), self.threshold_axes.get_ylim())
        self.threshold_image.remove()
        self.threshold_image = self.threshold_axes.imshow(
            view, cmap=None if view.ndim == 3 else "gray",
            vmin=None if view.ndim == 3 else 0.0, vmax=None if view.ndim == 3 else 1.0,
            interpolation="nearest", origin="upper",
        )
        self.threshold_axes.set_xlim(limits[0])
        self.threshold_axes.set_ylim(limits[1])
        self.threshold_axes.set_xticks([])
        self.threshold_axes.set_yticks([])
        self.figure.canvas.draw_idle()

    def on_log_toggled(self, _label):
        self.log_histogram = not self.log_histogram
        self.histogram_axes.set_yscale("log" if self.log_histogram else "linear")
        self.refresh(slice_changed=True)

    def on_key(self, event):
        steps = {"left": -1, "right": 1, "down": -10, "up": 10,
                 "pageup": 100, "pagedown": -100, "home": -self.volume.dim[2], "end": self.volume.dim[2]}
        if event.key in steps:
            target = min(max(self.index + steps[event.key], 0), self.volume.dim[2] - 1)
            if target != self.index:
                # Driving the slider rather than the index keeps the widget and the view in step.
                self.slice_slider.set_val(target)
        elif event.key == "p":
            print("ISO threshold: {:.6g}".format(self.iso))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Browse the slices of an MHD volume and pick an ISO threshold. Slices are "
                    "streamed from the raw file, so the volume need not fit in memory."
    )
    parser.add_argument("mhd", help="the .mhd file to browse")
    parser.add_argument("-s", "--slice", type=int, default=None,
                        help="slice to start on; defaults to the middle of the volume")
    parser.add_argument("-i", "--iso", type=float, default=None,
                        help="ISO threshold to start on; defaults to the middle of the value range")
    parser.add_argument("--stride", type=int, default=1,
                        help="show and histogram only every n-th pixel of a slice, which keeps the "
                             "sliders responsive on very large scans (default: 1)")
    parser.add_argument("--vmin", type=float, default=None, help="lower end of the display window")
    parser.add_argument("--vmax", type=float, default=None, help="upper end of the display window")
    parser.add_argument("--value-min", type=float, default=None,
                        help="lower end of the histogram and the ISO slider")
    parser.add_argument("--value-max", type=float, default=None,
                        help="upper end of the histogram and the ISO slider")
    args = parser.parse_args(argv)

    if args.stride < 1:
        parser.error("--stride has to be at least 1.")

    # The volume is opened before matplotlib is even imported, so that a header this tool cannot
    # read is reported as such instead of behind a complaint about the plotting backend.
    try:
        volume = MHDVolume(args.mhd)
    except (MHDError, OSError, ValueError) as error:
        return fail("{}: {}".format(args.mhd, error))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        volume.close()
        return fail("this tool needs matplotlib: python3 -m pip install matplotlib PyQt5")

    if plt.get_backend().lower() == "agg":
        volume.close()
        return fail("matplotlib has no interactive backend available; install a GUI backend, "
                    "e.g. python3 -m pip install PyQt5, or the system's python3-tk package.")

    try:
        data_min, data_max, low, high = volume.sample_range(args.stride)

        value_min = args.value_min if args.value_min is not None else data_min
        value_max = args.value_max if args.value_max is not None else data_max
        if value_max <= value_min:
            # A volume of one single value would leave the slider with nothing to slide along.
            value_max = value_min + 1.0

        vmin = args.vmin if args.vmin is not None else low
        vmax = args.vmax if args.vmax is not None else high
        if vmax <= vmin:
            vmin, vmax = value_min, value_max

        index = args.slice if args.slice is not None else volume.dim[2] // 2
        if not 0 <= index < volume.dim[2]:
            return fail("slice {} is outside the volume's {} slices.".format(index, volume.dim[2]))

        iso = args.iso if args.iso is not None else 0.5 * (value_min + value_max)

        print("{}: {} x {} x {} voxels, {}, {} mm/voxel".format(
            args.mhd, volume.dim[0], volume.dim[1], volume.dim[2], volume.element_type,
            " x ".join("{:g}".format(s) for s in volume.spacing)))
        print("  sampled value range [{:.6g}, {:.6g}], display window [{:.6g}, {:.6g}]".format(
            data_min, data_max, vmin, vmax))
        print("  arrow keys step through slices, 'p' prints the current ISO threshold")

        viewer = Viewer(volume, index, iso, args.stride, vmin, vmax, value_min, value_max)
        plt.show()
        print("ISO threshold: {:.6g}".format(viewer.iso))
    except (MHDError, OSError) as error:
        return fail("{}: {}".format(args.mhd, error))
    finally:
        volume.close()

    return 0


def fail(message):
    print("error: {}".format(message), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
