#!/usr/bin/env python3

from __future__ import division
import json
import sys
import argparse
import traceback
import collections
import numpy as np

_matplotlib_backend = 'agg'
import matplotlib
matplotlib.use(_matplotlib_backend)
import matplotlib.pyplot as plt

timing_types = ('wall', 'sys', 'user', 'cpu', 'perc')
memory_types = ('total_peak_rss', 'total_current_rss', 'self_peak_rss',
    'self_current_rss')

concurrency = 1
mpi_size = 1
min_time = 0.01
min_memory = 5
img_dpi = 75
img_size = {'w': 1200, 'h': 800}


#==============================================================================#
def nested_dict():
    return collections.defaultdict(nested_dict)


#==============================================================================#
class memory_data():

    def __init__(self):
        self.data = nested_dict()
        for key in memory_types:
            self.data[key] = []

    def append(self, _data):
        n = 0
        for key in memory_types:
            self.data[key].append(_data[n])
            n += 1

    def __add__(self, rhs):
        for key in memory_types:
            self.data[key].extend(rhs.data[key])

    def reset(self):
        self.data = nested_dict()
        for key in memory_types:
            self.data[key] = []

    def __getitem__(self, key):
        return self.data[key]


#==============================================================================#
class timing_data():

    def __init__(self):
        self.data = nested_dict()
        for key in timing_types:
            self.data[key] = []

    def append(self, _data):
        n = 0
        for key in timing_types:
            self.data[key].append(_data[n])
            n += 1

    def __add__(self, rhs):
        for key in timing_types:
            self.data[key].extend(rhs.data[key])

    def reset(self):
        self.data = nested_dict()
        for key in timing_types:
            self.data[key] = []

    def __getitem__(self, key):
        return self.data[key]


#==============================================================================#
class timing_function():

    def __init__(self):
        self.data = timing_data()
        self.memory = memory_data()
        self.laps = 0

    def process(self, denom, obj, nlap):
        _wall = obj['wall_elapsed'] / denom
        _user = obj['user_elapsed'] / denom
        _sys = obj['system_elapsed'] / denom
        _cpu = obj['cpu_elapsed'] / denom
        _tpeak = obj['rss_max']['peak'] / (1024.0 * 1.0e6)
        _tcurr = obj['rss_max']['current'] / (1024.0 * 1.0e6)
        _speak = obj['rss_self']['peak'] / (1024.0 * 1.0e6)
        _scurr = obj['rss_self']['current'] / (1024.0 * 1.0e6)
        _perc = (_cpu / _wall) * 100.0 if _wall > 0.0 else 100.0
        if _wall > min_time or abs(_speak) > min_memory or abs(_scurr) > min_memory:
            self.data.append([_wall, _sys, _user, _cpu, _perc])
            self.memory.append([_tpeak, _tcurr, _speak, _scurr])
        self.laps += nlap

    def __getitem__(self, key):
        return self.data[key]

    def length(self):
        return max(len(self.data['cpu']), len(self.memory['self_peak_rss']))


#==============================================================================#
def read(filename):

    print('Opening {}...'.format(filename))
    f = open(filename, "r")
    data_0 = json.load(f)
    global concurrency
    global mpi_size

    max_level = 0
    concurrency_sum = 0
    mpi_size = len(data_0['ranks'])
    for i in range(0, len(data_0['ranks'])):
        data_1 = data_0['ranks'][i]
        concurrency_sum += int(data_1['timing_manager']['omp_concurrency'])
        for j in range(0, len(data_1['timing_manager']['timers'])):
            data_2 = data_1['timing_manager']['timers'][j]
            nlaps = int(data_2['timer.ref']['laps'])
            indent = ""
            nlevel = int(data_2['timer.level'])
            max_level = max([max_level, nlevel])

    concurrency = concurrency_sum / mpi_size
    timing_functions = nested_dict()
    for i in range(0, len(data_0['ranks'])):
        data_1 = data_0['ranks'][i]
        for j in range(0, len(data_1['timing_manager']['timers'])):
            data_2 = data_1['timing_manager']['timers'][j]
            nlaps = int(data_2['timer.ref']['laps'])
            indent = ""
            nlevel = int(data_2['timer.level'])
            for n in range(0, nlevel):
                indent = '|{}'.format(indent)
            tag = '{} {}'.format(indent, data_2['timer.tag'])

            if not tag in timing_functions:
                timing_functions[tag] = timing_function()
            timing_func = timing_functions[tag]
            data_3 = data_2['timer.ref']
            timing_func.process(data_3['to_seconds_ratio_den'], data_3, nlaps)

            if timing_func.length() == 0:
                del timing_functions[tag]

    return timing_functions


#==============================================================================#
def plot_timing(filename, title, timing_data_dict, disp=False):

    ntics = len(timing_data_dict)
    ytics = []

    if ntics == 0:
        print ('{} had no timing data less than the minimum time ({} s)'.format(filename,
                                                                            min_time))
        return()

    avgs = nested_dict()
    stds = nested_dict()
    for key in timing_types:
        avgs[key] = []
        stds[key] = []

    for func, obj in timing_data_dict.items():
        ytics.append('{} x [ {} counts ]'.format(func, obj.laps))
        for key in timing_types:
            data = obj[key]
            avgs[key].append(np.mean(data))
            if len(data) > 1:
                stds[key].append(np.std(data))
            else:
                stds[key].append(0.0)

    # the x locations for the groups
    ind = np.arange(ntics)
    # the thickness of the bars: can also be len(x) sequence
    thickness = 0.8

    f = plt.figure(figsize=(img_size['w'] / img_dpi, img_size['h'] / img_dpi),
                   dpi=img_dpi)
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    f.subplots_adjust(left=0.05, right=0.75, bottom=0.05, top=0.90)

    ytics.reverse()
    for key in timing_types:
        avgs[key].reverse()
        stds[key].reverse()

    iter_order = ('cpu', 'wall', 'sys')
    plots = []
    lk = None
    for key in iter_order:
        data = avgs[key]
        err = stds[key]
        p = None
        if lk is None:
            p = plt.barh(ind, data, thickness, xerr=err)
        else:
            p = plt.barh(ind, data, thickness, xerr=err, bottom=lk)
        #lk = avgs[key]
        plots.append(p)

    plt.grid()
    plt.xlabel('Time [seconds]')
    plt.title('Timing report for {}'.format(title))
    plt.yticks(ind, ytics, ha='left')
    plt.setp(ax.get_yticklabels(), fontsize='smaller')
    plt.legend(plots, iter_order)
    imgfname = filename.replace('.json', '_timing.png')
    plt.savefig(imgfname, dpi=img_dpi)
    if disp:
        plt.show()
    else:
        plt.close()


#==============================================================================#
def plot_memory(filename, title, memory_data_dict, disp=False):

    ntics = len(memory_data_dict)
    ytics = []

    if ntics == 0:
        print ('{} had no memory data less than the minimum memory ({} MB)'.format(
            filename, min_memory))
        return()

    avgs = nested_dict()
    stds = nested_dict()
    for key in memory_types:
        avgs[key] = []
        stds[key] = []

    for func, obj in memory_data_dict.items():
        ytics.append('{} x [ {} counts ]'.format(func, obj.laps))
        for key in memory_types:
            data = obj.memory[key]
            if len(data) == 0:
                avgs[key].append(0.0)
                stds[key].append(0.0)
                continue
            avgs[key].append(np.mean(data))
            if len(data) > 1:
                stds[key].append(np.std(data))
            else:
                stds[key].append(0.0)

    # the x locations for the groups
    ind = np.arange(ntics)
    # the thickness of the bars: can also be len(x) sequence
    thickness = 0.8

    f = plt.figure(figsize=(img_size['w'] / img_dpi, img_size['h'] / img_dpi),
                   dpi=img_dpi)
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    f.subplots_adjust(left=0.05, right=0.75, bottom=0.05, top=0.90)

    ytics.reverse()
    for key in memory_types:
        if len(avgs[key]) == 0:
            del avgs[key]
            del stds[key]

    for key in memory_types:
        avgs[key].reverse()
        stds[key].reverse()

    iter_order = ['self_peak_rss', 'self_current_rss']
    plots = []
    lk = None
    for key in iter_order:
        data = avgs[key]
        if len(data) == 0:
            continue
        err = stds[key]
        p = None
        if lk is None:
            p = plt.barh(ind, data, thickness, xerr=err)
        else:
            p = plt.barh(ind, data, thickness, xerr=err, bottom=lk)
        #lk = avgs[key]
        plots.append(p)

    if len(plots) == 0:
        return

    plt.grid()
    plt.xlabel('Memory [MB]')
    plt.title('Memory report for {}'.format(title))
    plt.yticks(ind, ytics, ha='left')
    plt.setp(ax.get_yticklabels(), fontsize='smaller')
    plt.legend(plots, iter_order)
    imgfname = filename.replace('.json', '_memory.png')
    plt.savefig(imgfname, dpi=img_dpi)
    if disp:
        plt.show()
    else:
        plt.close()


#==============================================================================#
def main(args):
    global concurrency
    global mpi_size

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs='*', help="File input")
    parser.add_argument("-d", "--display", required=False, action='store_true',
                        help="Display plot", dest='display_plot')
    parser.set_defaults(display_plot=False)

    args = parser.parse_args()
    print('Files: {}'.format(args.files))

    file_data = dict()
    file_title = dict()
    for filename in args.files:
        print ('Reading {}...'.format(filename))
        file_data[filename] = read(filename)
        title = filename.replace('timing_report_', '')
        title = title.replace('.json', '')
        title = '"{}"\n@ MPI procs = {}, Threads/proc = {}'.format(title, mpi_size,
                                                                int(concurrency))
        file_title[filename] = title

    for filename, data in file_data.items():
        print ('Plotting {}...'.format(filename))
        plot_timing(filename, file_title[filename], data, args.display_plot)
        plot_memory(filename, file_title[filename], data, args.display_plot)


#==============================================================================#
if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print ('Exception - {}'.format(e))

    print ('Done - {}'.format(sys.argv[0]))
    sys.exit(0)
