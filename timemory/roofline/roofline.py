#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import re
import sys
import json
import math
import argparse
from math import log, pi

import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import numpy


GIGABYTE = 1.0e9
VERBOSE = int(os.environ.get("TIMEMORY_VERBOSE", "0"))


__all__ = ['smooth',
           'get_peak_flops',
           'get_peak_bandwidth',
           'get_hotspots',
           'get_color',
           'plot_parameters',
           'plot_roofline']

#   labels_type m_labels = {{"label", "working-set", "trials", "total-bytes",
#                            "total-ops", "ops-per-set", "counter", "device", "dtype",
#                            "exec-params"}}
#        ar(serializer:: make_nvp("working_set_min", working_set_min),
#           serializer:: make_nvp("memory_max", memory_max),
#           serializer:: make_nvp("nthreads", nthreads),
#           serializer:: make_nvp("nrank", nrank), serializer: : make_nvp("nproc", nproc),
#           serializer:: make_nvp("nstreams", nstreams),
#           serializer:: make_nvp("grid_size", grid_size),
#           serializer:: make_nvp("block_size", block_size),
#           serializer: : make_nvp("shmem_size", shmem_size))


#==============================================================================#
#
def get_json_entry(inp, key):
    for _key in [key, key.replace("_", "-"), key.replace("-", "_")]:
        if _key in inp:
            return inp[_key]
    return None


#==============================================================================#
#
class ert_params():
    """
    ERT execution parameters
    """

    def __init__(self, inp):
        self.working_set_min = get_json_entry(inp, "working_set_min")
        self.memory_max = get_json_entry(inp, "memory_max")
        self.nthreads = get_json_entry(inp, "nthreads")
        self.nrank = get_json_entry(inp, "nrank")
        self.nproc = get_json_entry(inp, "nproc")
        self.nstreams = get_json_entry(inp, "nstreams")
        self.grid_size = get_json_entry(inp, "grid_size")
        self.block_size = get_json_entry(inp, "block_size")
        self.shmem_size = get_json_entry(inp, "shmem_size")

    def __str__(self):
        ret = []
        for itr in ("working_set_min", "memory_max", "nthreads", "nrank",
                    "nproc", "nstreams", "grid_size", "block_size", "shmem_size"):
            ret += ["=".join([itr, "{}".format(getattr(self, itr))])]
        return ", ".join(ret)


#==============================================================================#
#
class ert_counter():
    """
    ERT counter data
    """

    def __init__(self, inp):
        _data = get_json_entry(inp, "repr_data")
        self.units = get_json_entry(inp, "units")
        self.display_units = get_json_entry(inp, "display_units")
        if not isinstance(_data, list) and not isinstance(_data, tuple):
            self.data = [_data]
        else:
            self.data = _data
        for i in range(len(self.data)):
            self.data[i] /= self.units

    def __str__(self):
        ret = []
        for itr in ("data", "units", "display_units"):
            ret += ["=".join([itr, "{}".format(getattr(self, itr))])]
        return ", ".join(ret)

    def get_peak(self, total, peak=[]):
        _data = self.data
        for i in range(len(_data)):
            value = total / _data[i]
            if i >= len(peak):
                peak.append(value)
            else:
                peak[i] = max([peak[i], value])
        return peak

    def get(self, total):
        _data = self.data
        _list = []
        for i in range(len(_data)):
            _list.append(total / _data[i])
        return _list


#==============================================================================#
#
class ert_data():
    """
    ERT instance data
    """

    def __init__(self, inp):
        self.label = get_json_entry(inp, "label")
        self.working_set = get_json_entry(inp, "working-set")
        self.trials = get_json_entry(inp, "trials")
        self.total_bytes = get_json_entry(inp, "total-bytes")
        self.total_ops = get_json_entry(inp, "total-ops")
        self.ops_per_set = get_json_entry(inp, "ops-per-set")
        self.device = get_json_entry(inp, "device")
        self.dtype = get_json_entry(inp, "dtype")
        self.counter = ert_counter(get_json_entry(inp, "counter"))
        self.exec_params = ert_params(get_json_entry(inp, "exec-params"))

    def __str__(self):
        ret = []
        for itr in ("label", "working_set", "trials", "total_bytes",
                    "total_ops", "ops_per_set", "device", "dtype"):
            ret += ["=".join([itr, "{}".format(getattr(self, itr))])]
        for itr in ("counter", "exec_params"):
            ret += ["".join(["\n    {:>12}".format(itr),
                             ": ({})".format(getattr(self, itr))])]
        return ", ".join(ret)


#==============================================================================#
#
def smooth(x, y):
    """
    Smooth a curve
    """
    xs = x[:]
    ys = y[:]
    d = 0
    for i in range(0, len(ys)):
        num = min(len(ys), i+d+1) - max(0, i-d)
        total = sum(ys[max(0, i-d):min(len(ys), i+d+1)])
        ys[i] = total/float(num)
    return xs, ys


#==============================================================================#
#
def read_ert(inp):
    data = inp

    def _read_ert(_data):
        if not "ert" in _data:
            return None
        else:
            inst = []
            for element in _data["ert"]:
                inst.append(ert_data(element))
            return inst

    inst = _read_ert(data)
    if inst is None:
        if "roofline" in data:
            data = data["roofline"]
        inst = _read_ert(data)

    for entry in inst:
        print("{}\n".format(entry))

    return inst


#==============================================================================#
#
def get_peak_ops(roof_data, flop_info=None):
    """
    Get the peak operations / sec
    """
    peak = []

    for element in roof_data:
        total_ops = element.total_ops / GIGABYTE
        peak = element.counter.get_peak(total_ops, peak)

    info = "GFLOPs/sec"
    if flop_info is not None:
        info_list = re.sub(r'[^\w]', ' ', flop_info).split()
        if len(info_list) > 0:
            info = info_list[0] + " GFLOPs/sec"
    peak_ops = [peak, info]
    return peak_ops


#==============================================================================#
#
def get_peak_bandwidth(roof_data):
    """
    Get multi-level bandwidth peaks - Implementation from ERT:
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit
    """
    ref_intensity = 0
    work_set = []
    bandwidth_data = []

    # Read bandwidth raw data
    for element in roof_data:
        intensity = element.ops_per_set
        if ref_intensity == 0:
            ref_intensity = intensity
        if intensity != ref_intensity:
            continue
        work_set.append(element.working_set)
        total_bytes = element.total_bytes / GIGABYTE
        bandwidth_data.append(element.counter.get(total_bytes))

    fraction = 1.05
    samples = 10000

    max_bandwidth = max(bandwidth_data)
    begin = bandwidth_data.index(max_bandwidth)

    work_set = work_set[begin:]
    bandwidth_data = bandwidth_data[begin:]
    min_bandwidth = min(bandwidth_data)

    dband = max_bandwidth/float(samples - 1)

    counts = samples*[0]
    totals = samples*[0.0]

    work_set, bandwidth_data = smooth(work_set, bandwidth_data)

    for i in range(0, samples):
        cband = i*dband
        for bandwidth in bandwidth_data:
            if bandwidth >= cband/fraction and bandwidth <= cband*fraction:
                totals[i] += bandwidth
                counts[i] += 1

    band_list = [[1000*max_bandwidth, 1000]]

    maxc = -1
    maxi = -1

    for i in range(samples-3, 1, -1):
        if counts[i] > 10:
            if counts[i] > maxc:
                maxc = counts[i]
                maxi = i
        else:
            if maxc > 1:
                value = float(totals[maxi])/max(1, counts[maxi])
                if 1.20*value < float(band_list[-1][0])/band_list[-1][1]:
                    band_list.append([totals[maxi], counts[maxi]])
                else:
                    band_list[-1][0] += totals[maxi]
                    band_list[-1][1] += counts[maxi]
            maxc = -1
            maxi = -1

    band_info_list = ["DRAM"]
    cache_num = len(band_list)-1

    for cache in range(1, cache_num+1):
        band_info_list = ["L%d" % (cache_num+1 - cache)] + band_info_list

    peak_bandwidths = []
    for (band, band_info) in zip(band_list, band_info_list):
        band_info = band_info + " GB/s"
        peak_bandwidths.append([float(band[0]/band[1]), band_info])
    return peak_bandwidths


#==============================================================================#
#
def get_hotspots(op_data, ai_data):
    """
    Get the hotspots information
    """
    op_data_type = op_data["type"]
    ai_data_type = ai_data["type"]

    op_type = None
    ai_type = None
    if "cpu_roofline" in op_data_type:
        op_type = "cpu"
    if "cpu_roofline" in ai_data_type:
        ai_type = "cpu"
    if "gpu_roofline" in op_data_type:
        op_type = "gpu"
    if "gpu_roofline" in ai_data_type:
        ai_type = "gpu"

    op_graph_data = op_data["graph"]
    ai_graph_data = ai_data["graph"]
    hotspots = []

    avg_runtime = 0.0
    max_runtime = 0.0
    max_length = min([len(op_graph_data), len(ai_graph_data)])
    all_runtime = []

    def get_runtime(_data, extra=[]):
        opts = ["runtime", "elapsed"] + extra
        for opt in opts:
            if opt in _data:
                return float(_data[opt])
        return None

    def get_flops(_data, extra=[]):
        opts = ["flops", "counted_ops", "flop_count", "flop_count_sp", "flop_count_dp",
                "flop_count_hp", "DP operations", "SP operations"] + extra
        for opt in opts:
            if opt in _data:
                return float(_data[opt])
        return None

    def get_bandwidth(_data, extra=[]):
        opts = ["bandwidth", "counted_ins",
                "ldst_executed", "L/S completed"] + extra
        for opt in opts:
            if opt in _data:
                return float(_data[opt])
        return None

    for i in range(0, max_length):
        ai_repr = ai_graph_data[i]["entry"]["repr_data"]
        op_repr = op_graph_data[i]["entry"]["repr_data"]
        all_runtime += filter(None,
                              [get_runtime(ai_repr), get_runtime(op_repr)])

    for rt in all_runtime:
        avg_runtime += rt

    if len(all_runtime) > 1:
        max_runtime = max(all_runtime)
        avg_runtime -= max_runtime
        avg_runtime /= len(all_runtime) - 1.0

    for i in range(0, max_length):
        runtimes = []
        flop = None
        bandwidth = None

        ai_repr = ai_graph_data[i]["entry"]["repr_data"]
        op_repr = op_graph_data[i]["entry"]["repr_data"]

        label = op_graph_data[i]["prefix"]
        runtimes += filter(None, [get_runtime(ai_repr), get_runtime(op_repr)])

        if op_type == "gpu":
            flop = get_flops(op_repr)
            bandwidth = get_bandwidth(op_repr)
        elif op_type == "cpu":
            flop = get_flops(op_repr)
            if flop is None:
                flop = get_flops(op_repr, ["counted"])

        if ai_type == "cpu":
            bandwidth = get_bandwidth(ai_repr)
            if bandwidth is None:
                bandwidth = get_bandwidth(ai_repr, ["counted"])

        runtime = 0.0
        for rt in runtimes:
            runtime += rt
        runtime /= len(runtimes)

        intensity = flop / bandwidth
        flop = flop / GIGABYTE / runtime
        proportion = runtime / avg_runtime
        label = re.sub(r'^[|0-9]+', ' ', label).replace("> [cxx] ", "").replace(
            "> [_c_] ", "").replace("> [pyc] ", "")

        if VERBOSE > 1:
            print("intensity: {}, flop: {}, proportion: {}, label: {}".format(
                intensity, flop, proportion, label))
        # this can arise from overflow
        if flop <= 1.0e-3 or bandwidth <= 0.0:
            continue
        elif VERBOSE > 0:
            print("{} : runtime = {}, avg = {}, proportion = {}".format(
                label, runtime, avg_runtime, proportion))

        hotspots.append([intensity, flop, proportion, label])
    return hotspots


#==============================================================================#
#
def get_color(proportion):
    if proportion < 0.01:
        color = "lawngreen"
    elif proportion < 0.1:
        color = "yellow"
    else:
        color = "red"
    return color


#==============================================================================#
#
class plot_parameters():
    def __init__(self, peak_flops, hotspots):
        y_digits = int(math.log10(peak_flops[0]))+1
        self.xmin = 0.01
        self.xmax = 100
        self.ymin = 1
        self.ymax = 10**y_digits
        self.xlabel = ('Arithmetic Intensity [FLOPs/Byte]')
        self.ylabel = ('Performance [GFLOPs/sec]')

        for element in (hotspots):
            intensity = element[0]
            flop = element[1]
            if flop > self.ymax:
                self.ymax = 10**int(log(flop)/log(10)+1)
            if flop < self.ymin:
                self.ymin = 10**int(log(flop)/log(10)-1)
            if intensity > self.xmax:
                self.xmax = 10**int(log(intensity)/log(10)+1)
            if intensity < self.xmin:
                self.xmin = 10**int(log(intensity)/log(10)-1)
        print("X (min, max) = {}, {}, Y (min, max) = {}, {}".format(
            self.xmin, self.xmax, self.ymin, self.ymax))


#==============================================================================#
#
def plot_roofline(ai, op, display=False, fname="roofline",
                  image_type="png", output_dir=os.getcwd(), title="Roofline Plot",
                  width=1600, height=1200, dpi=75):
    """
    Plot the roofline
    """
    op_data = op["rank"]["data"]
    ai_data = ai["rank"]["data"]

    band_data = read_ert(ai_data)
    peak_data = read_ert(op_data)

    info = op_data["unit_repr"] if "unit_repr" in op_data else None

    peak_flop = get_peak_ops(peak_data, info)
    peak_band = get_peak_bandwidth(band_data)
    hotspots = get_hotspots(op_data, ai_data)

    plot_params = plot_parameters(peak_flop, hotspots)

    f = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = f.add_subplot(111)

    plt.title(title)
    plt.grid(True, which="major", ls="--", lw=1)
    plt.grid(True, which="minor", ls="--", lw=0.5)
    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel(plot_params.xlabel)
    plt.ylabel(plot_params.ylabel)

    axes = plt.gca()
    axes.set_xlim([plot_params.xmin, plot_params.xmax])
    axes.set_ylim([plot_params.ymin, plot_params.ymax])

    # plot bandwidth roof
    x0 = plot_params.xmax
    for band in (peak_band):
        x1 = plot_params.xmin
        y1 = band[0] * plot_params.xmin
        if y1 < plot_params.ymin:
            x1 = plot_params.ymin / band[0]
            y1 = plot_params.ymin
        x2 = peak_flop[0] / band[0]
        y2 = peak_flop[0]
        if x2 < x0:
            x0 = x2

        x1log = log(x1)/log(10)
        x2log = log(x2)/log(10)
        y1log = log(y1)/log(10)
        y2log = log(y2)/log(10)
        x_text = 10**((x1log + x2log)/2)
        y_text = 10**((y1log + y2log)/2)

        fig = plt.gcf()
        size = fig.get_size_inches()*fig.dpi
        fig_x, fig_y = size

        dx = log(x2) - log(x1)
        dy = log(y2) - log(y1)
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        Dx = dx * fig_x / (log(x_max) - log(x_min))
        Dy = dy * fig_y / (log(y_max) - log(y_min))
        angle = (180/pi)*numpy.arctan(Dy / Dx)

        text(x_text, y_text, "%.2f %s" %
             (band[0], band[1]), rotation=angle, rotation_mode='anchor')
        plt.plot([x1, x2], [y1, y2], color='magenta')

    # plot computing roof
    text(plot_params.xmax, peak_flop[0] + 2, "%.2f %s" % (peak_flop[0],
                                                          peak_flop[1]), horizontalalignment='right')
    plt.plot([x0, plot_params.xmax], [peak_flop[0], peak_flop[0]], color='b')

    # plot hotspots
    for element in (hotspots):
        if VERBOSE > 0:
            print(element[0], element[1])
        c = get_color(element[2])
        plt.scatter(element[0], element[1], color=c)
        text(element[0], element[1], "%s" %
             element[3], rotation=0, rotation_mode='anchor')

    if display:
        print('Displaying plot...')
        plt.show()
    else:
        imgfname = os.path.join(output_dir, os.path.basename(fname))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not ".{}".format(image_type) in imgfname:
            imgfname += ".{}".format(image_type)
        print('Saving plot: "{}"...'.format(imgfname))
        plt.savefig(imgfname)
        plt.close()
