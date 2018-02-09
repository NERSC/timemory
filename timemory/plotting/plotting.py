#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2018, The Regents of the University of California, 
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

## @file plotting.py
## Plotting routines for TiMemory module
##

from __future__ import absolute_import
from __future__ import division

import json
import sys
import traceback
import collections
import numpy as np
import os
import copy

timing_types = ('wall', 'sys', 'user', 'cpu', 'perc')
memory_types = ('total_peak_rss', 'total_current_rss', 'self_peak_rss',
    'self_current_rss')

concurrency = 1
mpi_size = 1
min_time = 0.005
min_memory = 1
img_dpi = 75
img_size = {'w': 1200, 'h': 800}

plotted_files = []

#==============================================================================#
def echo_dart_tag(name, filepath):
    print('<DartMeasurementFile name="{}" '.format(name) +
    'type="image/png">{}</DartMeasurementFile>'.format(filepath))


#==============================================================================#
def add_plotted_files(name, filepath, echo_dart):
    global plotted_files
    if echo_dart:
        echo_dart_tag(name, filepath)
    found = False
    for p in plotted_files:
        if p[0] == name and p[1] == filepath:
            found = True
            break
    if found is False:
        plotted_files.append([ name, filepath ])


#==============================================================================#
def make_output_directory(directory):
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)


#==============================================================================#
def nested_dict():
    return collections.defaultdict(nested_dict)


#==============================================================================#
class timemory_data():

    def __init__(self, types):
        self.data = nested_dict()
        self.types = types
        self.sums = {}
        for key in self.types:
            self.data[key] = []

    def append(self, _data):
        n = 0
        for key in self.types:
            self.data[key].append(_data[n])
            if not key in self.sums.keys():
                self.sums[key] = 0.0
            self.sums[key] += _data[n]
            n += 1

    def __add__(self, rhs):
        for key in self.types:
            self.data[key].extend(rhs.data[key])
            for k, v in rhs.data.items():
                if not k in self.sums.keys():
                    self.sums[k] = 0.0
                self.sums[k] += v

    def reset(self):
        self.data = nested_dict()
        for key in self.types:
            self.data[key] = []
            self.sums[key] = 0.0

    def __getitem__(self, key):
        return self.data[key]

    def get_order(self, include_keys):
        order = []
        sorted_keys = sorted(self.sums, key=lambda x: self.sums[x])
        for key in sorted_keys:
            if key in include_keys:
                order.append(key)
        return order


#==============================================================================#
class timemory_function():

    def __init__(self):
        self.timing = timemory_data(timing_types)
        self.memory = timemory_data(memory_types)
        self.laps = 0

    def process(self, denom, obj, nlap):
        _wall = obj['wall_elapsed'] / denom
        _user = obj['user_elapsed'] / denom
        _sys = obj['system_elapsed'] / denom
        _cpu = obj['cpu_elapsed'] / denom
        _tpeak = obj['rss_max']['peak']
        _tcurr = obj['rss_max']['current']
        _speak = obj['rss_self']['peak']
        _scurr = obj['rss_self']['current']
        _perc = (_cpu / _wall) * 100.0 if _wall > 0.0 else 100.0
        if _wall > min_time or abs(_speak) > min_memory or abs(_scurr) > min_memory:
            self.timing.append([_wall, _sys, _user, _cpu, _perc])
            self.memory.append([_tpeak, _tcurr, _speak, _scurr])
        self.laps += nlap

    def __getitem__(self, key):
        return self.timing[key]

    def length(self):
        return max(len(self.timing['cpu']), len(self.memory['self_peak_rss']))


#==============================================================================#
class plot_data():

    def __init__(self, filename = "output", concurrency = 1, mpi_size = 1,
                 timemory_functions = [], title = ""):
        self.filename = filename
        self.concurrency = concurrency
        self.mpi_size = mpi_size
        self.timemory_functions = timemory_functions
        self.title = title


    def __str__(self):
        return '\tFilename: {}\n\tConcurrency: {}\n\tMPI ranks: {}\n\t# functions: {}\n\tTitle: {}'.format(
            self.filename, self.concurrency, self.mpi_size, len(self.timemory_functions), self.title)


    def get_title(self):
        return '"{}"\n@ MPI procs = {}, Threads/proc = {}'.format(self.title,
                mpi_size, int(concurrency))


#==============================================================================#
def read(json_obj):

    data_0 = json_obj

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
    timemory_functions = nested_dict()
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

            if not tag in timemory_functions:
                timemory_functions[tag] = timemory_function()
            timemory_func = timemory_functions[tag]
            data_3 = data_2['timer.ref']
            timemory_func.process(data_3['to_seconds_ratio_den'], data_3, nlaps)

            if timemory_func.length() == 0:
                del timemory_functions[tag]

    return plot_data(concurrency = concurrency,
                     mpi_size = mpi_size,
                     timemory_functions = timemory_functions)


#==============================================================================#
def plot_timing(_plot_data, disp=False, output_dir=".", echo_dart=False):

    try:
        import tornado
        import matplotlib
        import matplotlib.pyplot as plt
    except:
        _matplotlib_backend = 'agg'
        import matplotlib
        matplotlib.use(_matplotlib_backend)
        import matplotlib.pyplot as plt

    iter_order = ['wall', 'cpu', 'sys']

    filename = _plot_data.filename
    title = _plot_data.get_title()
    timing_data_dict = _plot_data.timemory_functions

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

    _f = True
    for func, obj in timing_data_dict.items():
        if _f is True:
            iter_order = obj.timing.get_order(iter_order)
            _f = False
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

    plots = []
    lk = None
    iter_order.reverse()
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
    if disp:
        print('Displaying plot...')
        plt.show()
    else:
        make_output_directory(output_dir)
        imgfname = os.path.basename(filename)
        imgfname = imgfname.replace('.', '_timing.')
        if not '_timing.' in imgfname:
            imgfname += "_timing."
        imgfname = imgfname.replace('.json', '.png')
        imgfname = imgfname.replace('.py', '.png')
        if not '.png' in imgfname:
            imgfname += '.png'

        add_plotted_files(imgfname, os.path.join(output_dir, imgfname), echo_dart)

        imgfname = os.path.join(output_dir, imgfname)
        print('Saving plot: "{}"...'.format(imgfname))
        plt.savefig(imgfname, dpi=img_dpi)
        plt.close()


#==============================================================================#
def plot_memory(_plot_data, disp=False, output_dir=".", echo_dart=False):

    global plotted_files

    try:
        import tornado
        import matplotlib
        import matplotlib.pyplot as plt
    except:
        _matplotlib_backend = 'agg'
        import matplotlib
        matplotlib.use(_matplotlib_backend)
        import matplotlib.pyplot as plt

    iter_order = ['self_peak_rss', 'self_current_rss']
    filename = _plot_data.filename
    title = _plot_data.get_title()
    memory_data_dict = _plot_data.timemory_functions

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

    _f = True
    for func, obj in memory_data_dict.items():
        if _f is True:
            iter_order = obj.memory.get_order(iter_order)
            _f = False
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

    plots = []
    lk = None
    iter_order.reverse()
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
    if disp:
        #print('Displaying plot...')
        plt.show()
    else:
        make_output_directory(output_dir)
        imgfname = os.path.basename(filename)
        imgfname = imgfname.replace('.', '_memory.')
        if not '_memory.' in imgfname:
            imgfname += "_memory."
        imgfname = imgfname.replace('.json', '.png')
        imgfname = imgfname.replace('.py', '.png')
        if not '.png' in imgfname:
            imgfname += '.png'

        add_plotted_files(imgfname, os.path.join(output_dir, imgfname), echo_dart)

        imgfname = os.path.join(output_dir, imgfname)
        print('Saving plot: "{}"...'.format(imgfname))
        plt.savefig(imgfname, dpi=img_dpi)
        plt.close()


#==============================================================================#
def plot(data = [], files = [], display=False, output_dir='.', echo_dart=None):

    import timemory.options as options
    if echo_dart is None and options.echo_dart:
        echo_dart = True
    else:
        echo_dart = False

    if len(files) > 0:
        for filename in files:
            print ('Reading {}...'.format(filename))
            f = open(filename, "r")
            _data = read(json.load(f))
            f.close()
            _data.filename = filename
            _data.title = filename
            data.append(_data)

    for _data in data:
        try:
            print ('Plotting {}...'.format(_data.filename))
            plot_timing(_data, display, output_dir, echo_dart)
            plot_memory(_data, display, output_dir, echo_dart)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
            print ('Exception - {}'.format(e))
            print ('Error! Unable to plot "{}"...'.format(_data.filename))


#==============================================================================#
if __name__ == "__main__":
    import argparse
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--files", nargs='*', help="File input")
        parser.add_argument("-d", "--display", required=False, action='store_true',
                            help="Display plot", dest='display_plot')
        parser.add_argument("-t", "--titles", nargs='*', help="Plot titles")

        parser.set_defaults(display_plot=False)

        args = parser.parse_args()
        print('Files: {}'.format(args.files))
        if len(args.titles) != 1 and len(args.titles) != len(args.files):
            raise Exception("Error must provide one title or a title for each file")
        data = []
        for i in range(len(args.files)):
            f = open(args.files[i], "r")
            _data = read(json.load(f))
            if len(args.titles) == 1:
                _data.title = args.titles[0]
            else:
                _data.title = args.titles[i]
        plot(data, args.display)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print ('Exception - {}'.format(e))

    print ('Done - {}'.format(sys.argv[0]))
    sys.exit(0)
