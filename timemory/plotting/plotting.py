#!@PYTHON_EXECUTABLE@
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
import os
import copy
import warnings

_matplotlib_backend = None

#------------------------------------------------------------------------------#
# check with timemory.options
#
try:
    import timemory.options
    if timemory.options.matplotlib_backend != "default":
        _matplotlib_backend = timemory.options.matplotlib_backend
except:
    pass


#------------------------------------------------------------------------------#
# if not display variable we probably want to use agg
#
if (os.environ.get("DISPLAY") is None and
    os.environ.get("MPLBACKEND") is None and
    _matplotlib_backend is None):
    os.environ.setdefault("MPLBACKEND", "agg")


#------------------------------------------------------------------------------#
# tornado helps set the matplotlib backend but is not necessary
#
try:
    import tornado
except:
    pass


#------------------------------------------------------------------------------#
# import matplotlib and pyplot but don't fail
#
try:
    import matplotlib
    import matplotlib.pyplot as plt
    _matplotlib_backend = matplotlib.get_backend()
except:
    try:
        import matplotlib
        matplotlib.use("agg", warn=False)
        import matplotlib.pyplot as plt
        _matplotlib_backend = matplotlib.get_backend()
    except:
        pass


#------------------------------------------------------------------------------#
#
#
""" Default timing data to extract from JSON """
_default_timing_types = ['wall', 'sys', 'user', 'cpu', 'perc']

""" Default memory data to extract from JSON """
_default_memory_types = ['total_peak_rss', 'total_current_rss',
                         'self_peak_rss', 'self_current_rss']


""" Default fields for reducing # of timing plot functions displayed """
_default_timing_fields = ['wall', 'sys', 'user']

""" Default fields for reducing # of memory plot functions displayed """
_default_memory_fields = ['total_peak_rss', 'total_current_rss',
                          'self_peak_rss', 'self_current_rss']

""" Default minimum percent of max when reducing # of timing functions plotted """
_default_timing_min_percent = 0.05 # 5% of max

""" Default minimum percent of max when reducing # of memory functions plotted """
_default_memory_min_percent = 0.05 # 5% of max

""" Default image dots-per-square inch """
_default_img_dpi = 75

""" Default image size """
_default_img_size = {'w': 1600, 'h': 800}

""" Default image type """
_default_img_type = 'jpeg'

""" A list of all files that have been plotted """
plotted_files = []

""" Data fields stored in timemory_data """
timemory_types = _default_timing_types + _default_memory_types

#==============================================================================#
"""
A class for reducing the amount of data in plot by specifying a minimum
percentage of the max value and the fields to check against
"""
class plot_parameters():

    """ Global timing plotting params (these should be modified instead of _default_*) """
    timing_types = copy.copy(_default_timing_types)
    timing_fields = copy.copy(_default_timing_fields)
    timing_min_percent = copy.copy(_default_timing_min_percent)

    """ Global memory plotting params (these should be modified instead of _default_*) """
    memory_types = copy.copy(_default_memory_types)
    memory_fields = copy.copy(_default_memory_fields)
    memory_min_percent = copy.copy(_default_memory_min_percent)

    """ Global image plotting params (these should be modified instead of _default_*) """
    img_dpi = copy.copy(_default_img_dpi)
    img_size = copy.copy(_default_img_size)
    img_type = copy.copy(_default_img_type)

    def __init__(self,
                 # timing
                 _timing_types = timing_types,
                 _timing_min_percent = timing_min_percent,
                 _timing_fields = timing_fields,
                 # memory
                 _memory_types = memory_types,
                 _memory_min_percent = memory_min_percent,
                 _memory_fields = memory_fields,
                 # image
                 _img_dpi = img_dpi,
                 _img_size = img_size,
                 _img_type = img_type):
        # timing
        self.timing_types = _timing_types
        self.timing_min_percent = _timing_min_percent
        self.timing_fields = _timing_fields
        # memory
        self.memory_types = _memory_types
        self.memory_min_percent = _memory_min_percent
        self.memory_fields = _memory_fields
        # image
        self.img_dpi = _img_dpi
        self.img_size = _img_size
        self.img_type = _img_type
        # max values
        self.timing_max_value = 0.0
        self.memory_max_value = 0.0

    def __str__(self):
        # timing
        _a = 'Timing: {} = {}, {} = {}, {} = {}, {} = {}'.format(
            'types', self.timing_types,
            'min %', self.timing_min_percent,
            'fields', self.timing_fields,
            'max', self.timing_max_value)
        # memory
        _b = 'Memory: {} = {}, {} = {}, {} = {}, {} = {}'.format(
            'types', self.memory_types,
            'min %', self.memory_min_percent,
            'fields', self.memory_fields,
            'max', self.memory_max_value)
        # image
        _c = 'Image: {} = {}, {} = {}, {} = {}'.format(
            'dpi', self.img_dpi,
            'size', self.img_size,
            'type', self.img_type)
        return '[{}] [{}] [{}]'.format(_a, _b, _c)


#==============================================================================#
def echo_dart_tag(name, filepath, img_type=plot_parameters.img_type):
    """
    Printing this string will upload the results to CDash when running CTest
    """
    print('<DartMeasurementFile name="{}" '.format(name) +
    'type="image/{}">{}</DartMeasurementFile>'.format(img_type, filepath))


#==============================================================================#
def add_plotted_files(name, filepath, echo_dart):
    """
    Adds a file to the plotted file list and print CDash dart string
    """
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
    """
    mkdir -p
    """
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)


#==============================================================================#
def nested_dict():
    return collections.defaultdict(nested_dict)


#==============================================================================#
class timemory_data():
    """
    This class is for internal usage. It holds the JSON data
    """
    # ------------------------------------------------------------------------ #
    def __init__(self, func, types = timemory_types, extract_data=None):
        self.laps = 0
        self.types = types
        self.func = func
        self.data = nested_dict()
        self.sums = {}
        for key in self.types:
            self.data[key] = []
            self.sums[key] = 0.0
        # populate data and sums from existing data
        if extract_data is not None:
            self.laps = extract_data.laps
            for key in self.types:
                self.data[key] = extract_data.data[key]
                self.sums[key] = extract_data.sums[key]

    # ------------------------------------------------------------------------ #
    def process(self, denom, obj, nlap):
        """
        record data from JSON object
        """
        _wall = obj['wall_elapsed'] / denom
        _user = obj['user_elapsed'] / denom
        _sys = obj['system_elapsed'] / denom
        _cpu = obj['cpu_elapsed'] / denom
        _MB = (1.0)
        _tpeak = obj['rss_max']['peak'] / _MB
        _tcurr = obj['rss_max']['current'] / _MB
        _speak = obj['rss_self']['peak'] / _MB
        _scurr = obj['rss_self']['current'] / _MB
        _perc = (_cpu / _wall) * 100.0 if _wall > 0.0 else 100.0
        _dict = {
            'wall' : _wall,
            'sys' : _sys,
            'user' :_user,
            'cpu' : _cpu,
            'perc' : _perc,
            'total_peak_rss' : _tpeak,
            'total_current_rss' : _tcurr,
            'self_peak_rss' : _speak,
            'self_current_rss' : _scurr }
        self.append(_dict)
        self.laps += nlap

    # ------------------------------------------------------------------------ #
    def plottable(self, _params):
        """
        valid data above minimum
        """
        # compute the minimum values
        t_min = (0.01 * _params.timing_min_percent) * _params.timing_max_value
        m_min = (0.01 * _params.memory_min_percent) * _params.memory_max_value

        # function for checking passes test
        def is_valid(key, min_value):
            if key in self.sums.keys():
                # compute it's "percentage" w.r.t. max value
                return abs(self.sums[key]) > min_value
            return False

        # check the timing fields
        for field in _params.timing_fields:
            if is_valid(field, t_min):
                return True
        # check the memory fields
        for field in _params.memory_fields:
            if is_valid(field, m_min):
                return True

        # all values below minimum --> do not plot
        return False

    # ------------------------------------------------------------------------ #
    def append(self, _dict):
        """
        append data to dataset
        """
        for key in self.types:
            self.data[key].append(_dict[key])
            # add entry if not exist
            self.sums[key] += _dict[key]

    # ------------------------------------------------------------------------ #
    def __add__(self, rhs):
        """
        for combining results (typically from different MPI processes)
        """
        for key in self.types:
            self.data[key].extend(rhs.data[key])
            for k, v in rhs.data.items():
                self.sums[k] += v
        self.laps += rhs.laps

    # ------------------------------------------------------------------------ #
    def __sub__(self, rhs):
        """
        for differencing results (typically from two different runs)
        """
        for key in self.types:
            # if the lengths are the same, compute difference
            if len(self.data[key]) == len(rhs.data[key]):
                for i in len(self.data[key]):
                    self.data[key][i] -= rhs.data[key][i]
            else: # the lengths are different, insert the entries as neg values
                for entry in rhs.data[key]:
                    self.data[key].append(-entry)
            # compute the sums
            for k, v in rhs.data.items():
                self.sums[k] += v
        # this is a weird situation
        if self.laps != rhs.laps:
            self.laps = max(self.laps, rhs.laps)

    # ------------------------------------------------------------------------ #
    def reset(self):
        """
        clear all the data
        """
        self.data = nested_dict()
        for key in self.types:
            self.data[key] = []
            self.sums[key] = 0.0

    # ------------------------------------------------------------------------ #
    def __getitem__(self, key):
        """
        array indexing
        """
        return self.data[key]

    # ------------------------------------------------------------------------ #
    def keys(self):
        """
        get the keys
        """
        return self.data.keys()

    # ------------------------------------------------------------------------ #
    def __len__(self):
        """
        length operator
        """
        _maxlen = 0
        for key in rhs.data.keys():
            _maxlen = max(_maxlen, len(self.data[key]))
        return _maxlen

    # ------------------------------------------------------------------------ #
    def get_order(self, include_keys):
        """
        for getting an order based on a set of keys
        """
        order = []
        sorted_keys = sorted(self.sums, key=lambda x: abs(self.sums[x]))
        for key in sorted_keys:
            if key in include_keys:
                order.append(key)
        return order


#==============================================================================#
class plot_data():
    """
    A custom configuration for the data to be plotted

    Args:
        filename (str):
            the input filename (JSON file) to be processed
            also used to generate the output image name unless "output_name"
            is specified
        concurrency (int):
            the concurrency used
        mpi_size (int):
            the number of MPI processes
        title (str):
            title for plot
        plot_params (plot_parameters instance):
            the plotting parameters
        output_name (str):
    """
    def __init__(self,
                 filename = "output",
                 concurrency = 1, mpi_size = 1,
                 timemory_functions = [],
                 title = "",
                 plot_params = plot_parameters(),
                 output_name = None):

        self.filename = filename
        self.concurrency = concurrency
        self.mpi_size = mpi_size
        self.timemory_functions = timemory_functions
        self.title = title
        self.plot_params = plot_params
        self.output_name = output_name
        if self.output_name is None:
            self.output_name = self.filename
        # calc max values
        for key, obj in self.timemory_functions.items():
            # calc timing max
            for _key in self.plot_params.timing_fields:
                _max = self.plot_params.timing_max_value
                _val = obj.sums[_key]
                #print('max = {}, sum [{}] = {}'.format(_max, _key, _val))
                self.plot_params.timing_max_value = max([_max, _val])
            # calc memory max
            for _key in self.plot_params.memory_fields:
                _max = self.plot_params.memory_max_value
                _val = obj.sums[_key]
                #print('max = {}, sum [{}] = {}'.format(_max, _key, _val))
                self.plot_params.memory_max_value = max([_max, _val])

    # ------------------------------------------------------------------------ #
    def update_parameters(self, params = None):
        """
        Update plot parameters (i.e. recalculate maxes)
        """
        if params is not None:
            self.plot_params = params
        self.plot_params.timing_max_value = 0.0
        self.plot_params.timing_min_value = 0.0
        # calc max values
        for key, obj in self.timemory_functions.items():
            # calc timing max
            for _key in self.plot_params.timing_fields:
                _max = self.plot_params.timing_max_value
                _val = obj.sums[_key]
                #print('max = {}, sum [{}] = {}'.format(_max, _key, _val))
                self.plot_params.timing_max_value = max([_max, _val])
            # calc memory max
            for _key in self.plot_params.memory_fields:
                _max = self.plot_params.memory_max_value
                _val = obj.sums[_key]
                #print('max = {}, sum [{}] = {}'.format(_max, _key, _val))
                self.plot_params.memory_max_value = max([_max, _val])

    # ------------------------------------------------------------------------ #
    def get_timing(self):
        """
        Process the functions for timing plotting
        """
        self.update_parameters()
        _dict = nested_dict()
        for key, obj in self.timemory_functions.items():
            subset = timemory_data(key, types = self.plot_params.timing_types,
                                   extract_data = obj)
            if subset.plottable(self.plot_params):
                _dict[key] = subset
        return _dict

    # ------------------------------------------------------------------------ #
    def get_memory(self):
        """
        Process the functions for memory plotting
        """
        self.update_parameters()
        _dict = nested_dict()
        for key, obj in self.timemory_functions.items():
            subset = timemory_data(key, types = self.plot_params.memory_types,
                                   extract_data = obj)
            if subset.plottable(self.plot_params):
                _dict[key] = subset
        return _dict

    # ------------------------------------------------------------------------ #
    def __len__(self):
        """
        Get the length
        """
        return len(self.timemory_functions)

    # ------------------------------------------------------------------------ #
    def keys(self):
        """
        Get the dictionary keys
        """
        return self.timemory_functions.keys()

    # ------------------------------------------------------------------------ #
    def items(self):
        """
        Get the dictionary items
        """
        return self.timemory_functions.items()

    # ------------------------------------------------------------------------ #
    def __str__(self):
        """
        String repr
        """
        _list = [
            ('Filename', self.filename),
            ('Concurrency', self.concurrency),
            ('MPI ranks', self.mpi_size),
            ('# functions', len(self)),
            ('Title', self.title),
            ('Parameters', self.plot_params) ]
        _str = '\n'
        for entry in _list:
            _str = '{}\t{} : "{}"\n'.format(_str, entry[0], entry[1])
        return _str

    # ------------------------------------------------------------------------ #
    def get_title(self):
        """
        Construct the title for the plot
        """
        return '"{}"\n@ MPI procs = {}, Threads/proc = {}'.format(self.title,
                self.mpi_size, int(self.concurrency))


#==============================================================================#
def read(json_obj, plot_params=plot_parameters()):
    """
    Read the JSON data -- i.e. process JSON object of TiMemory data
    """

    # some fields
    data0 = json_obj
    max_level = 0
    concurrency_sum = 0
    manager_tag = 'manager'
    concurrency_tag = 'concurrency'
    mpi_size = len(data0['ranks'])
    timemory_functions = nested_dict()

    # ------------------------------------------------------------------------ #
    # loop over MPI ranks
    for i in range(0, len(data0['ranks'])):
        # shorthand
        data1 = data0['ranks'][i]
        # --------------------------------------------------#
        def get_manager_tag():
            # this is for backwards-compatibility
            _tag = 'timing_manager'
            try:
                _tmp = int(data1[_tag]['timers'])
            except:
                _tag = 'manager'
            return _tag
        # --------------------------------------------------#
        def get_concurrency_tag(_man_tag):
            # this is for backwards-compatibility
            _tag = 'omp_concurrency'
            try:
                _tmp = int(data1[_man_tag]['omp_concurrency'])
            except:
                _tag = 'concurrency'
            return _tag
        # --------------------------------------------------#

        manager_tag = get_manager_tag()
        concurrency_tag = get_concurrency_tag(manager_tag)

        concurrency_sum += int(data1[manager_tag][concurrency_tag])
        for j in range(0, len(data1[manager_tag]['timers'])):
            data2 = data1[manager_tag]['timers'][j]
            max_level = max(max_level, int(data2['timer.level']))

    # ------------------------------------------------------------------------ #
    # loop over ranks
    for i in range(0, len(data0['ranks'])):
        # shorthand
        data1 = data0['ranks'][i]

        # loop over timers
        for j in range(0, len(data1[manager_tag]['timers'])):
            data2 = data1[manager_tag]['timers'][j]
            indent = ""
            # for display
            for n in range(0, int(data2['timer.level'])):
                indent = '|{}'.format(indent)

            # construct the tag
            tag = '{} {}'.format(indent, data2['timer.tag'])

            # create timemory_data object if doesn't exist yet
            if not tag in timemory_functions:
                #print('Tag {} does not exist'.format(tag))
                timemory_functions[tag] = timemory_data(func = tag)
            # get timemory_data object
            timemory_func = timemory_functions[tag]

            # process the function data
            timemory_func.process(data2['timer.ref']['to_seconds_ratio_den'],
                                  data2['timer.ref'],
                                  int(data2['timer.ref']['laps']))

    # ------------------------------------------------------------------------ #
    # return a plot_data object
    return plot_data(concurrency = concurrency_sum / mpi_size,
                     mpi_size = mpi_size,
                     timemory_functions = timemory_functions,
                     plot_params = plot_params)


#==============================================================================#
def plot_generic(_plot_data, _types, _data_dict,
                 _type_str, _type_min, _type_unit):

    if _matplotlib_backend is None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except:
            warnings.warn("Matplotlib could not find a suitable backend. Skipping plotting...")
            return

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    filename = _plot_data.filename
    plot_params = _plot_data.plot_params
    title = _plot_data.get_title()
    nitem = len(_types)
    ntics = len(_data_dict) * nitem
    ytics = []

    if ntics == 0:
        print ('{} had no {} data less than the minimum time ({} {})'.format(
               filename, _type_str, _type_min, _type_unit))
        return False

    avgs = nested_dict()
    stds = nested_dict()
    for key in _types:
        avgs[key] = []
        stds[key] = []

    # _n is the major index
    _n = 0
    _l = ''
    for func, obj in _data_dict.items():
        #print('Plotting function: {}'.format(func))
        # _c is the minor index
        _c = 0
        for key in _types:
            if _n % nitem == 0:
                ytics.append('{} x [ {} counts ]'.format(func, obj.laps))
                _l = ytics[len(ytics)-1]
            else:
                #ytics.append('{} x [ {} counts ]'.format(func, obj.laps))
                ytics.append('')

            data = obj[key]
            for _i in range(0, nitem):
                if _i == _c:
                    avgs[key].append(np.mean(data))
                    stds[key].append(np.std(data) if len(data) > 1 else 0.0)
                else:
                    avgs[key].append(0.0)
                    stds[key].append(0.0)
            _n += 1
            _c += 1

    ytics[len(ytics)-1] = ''
    ytics.append('')

    # the x locations for the groups
    ind = np.arange(ntics)
    # the thickness of the bars: can also be len(x) sequence
    thickness = 1.0

    f = plt.figure(figsize=(plot_params.img_size['w'] / plot_params.img_dpi,
                            plot_params.img_size['h'] / plot_params.img_dpi),
                   dpi=plot_params.img_dpi)
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    f.subplots_adjust(left=0.05, right=0.75, bottom=0.05, top=0.90)

    # put largest at top
    ytics.reverse()
    for key in _types:
        avgs[key].reverse()
        stds[key].reverse()

    # construct the plots
    plots = []
    for key in _types:
        p = plt.barh(ind, avgs[key], thickness, xerr=stds[key],
                     align='edge', alpha=1.0, antialiased=False)
        plots.append(p)

    if len(plots) == 0:
        return

    grid_lines = []
    for i in range(0, ntics):
        if i % nitem == 0:
            grid_lines.append(i)

    #plt.minorticks_on()
    plt.yticks(ind, ytics, ha='left')
    plt.setp(ax.get_yticklabels(), fontsize='smaller')
    #plt.legend(plots, _types, loc='lower left', bbox_to_anchor=(0.0, 0.0))
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(plots, _types, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=nitem)

    #ax.xaxis.set_major_locator(plt.IndexLocator(1, 0))
    ax.yaxis.set_major_locator(plt.IndexLocator(1, 0))
    ax.xaxis.grid()
    #ax.yaxis.grid(markevery=nitem)
    ax.yaxis.grid()

    #for xmaj in ax.xaxis.get_majorticklocs():
    #    ax.axvline(x=xmaj, ls='--')
    #for xmin in ax.xaxis.get_minorticklocs():
    #    ax.axvline(x=xmin, ls='--')

    #for ymaj in ax.yaxis.get_majorticklocs():
    #    print('y major: {}'.format(ymaj))
    #    ax.axhline(y=ymaj, ls='--')
    #for ymin in ax.yaxis.get_minorticklocs():
    #    print('y minor: {}'.format(ymin))
    #    ax.axhline(y=ymin, ls='--')
    #plt.gca().grid(b=True, which='major', axis='y', markevery=nitem)
    #plt.gca().set_markevery()
    return True


#==============================================================================#
def plot_timing(_plot_data,
                disp=False, output_dir=".", echo_dart=False):

    #
    if _matplotlib_backend is None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except:
            warnings.warn("Matplotlib could not find a suitable backend. Skipping plotting...")
            return

    import matplotlib
    import matplotlib.pyplot as plt

    filename = _plot_data.filename
    title = _plot_data.get_title()
    _params = _plot_data.plot_params
    _ext = 'timing'

    _plot_min = (_params.timing_max_value *
                 (0.01 * _params.timing_min_percent))

    _do_plot = plot_generic(_plot_data,
                            _params.timing_fields,
                            _plot_data.get_timing(),
                            _ext,
                            _plot_min,
                            's')
    if not _do_plot:
        return

    plt.xlabel('Time [seconds]')
    plt.title('Timing report for {}'.format(title))
    if disp:
        print('Displaying plot...')
        plt.show()
    else:
        make_output_directory(output_dir)
        imgfname = os.path.basename(filename)
        imgfname = imgfname.replace('.', '_{}.'.format(_ext))
        if not '_{}.'.format(_ext) in imgfname:
            imgfname += '_{}.'.format(_ext)
        imgfname = imgfname.replace('.json', '.{}'.format(_params.img_type))
        imgfname = imgfname.replace('.py', '.{}'.format(_params.img_type))
        if not '.{}'.format(_params.img_type) in imgfname:
            imgfname += '.{}'.format(_params.img_type)
        while '..' in imgfname:
            imgfname = imgfname.replace('..', '.')

        add_plotted_files(imgfname, os.path.join(output_dir, imgfname), echo_dart)

        imgfname = os.path.join(output_dir, imgfname)
        print('Saving plot: "{}"...'.format(imgfname))
        plt.savefig(imgfname, dpi=_params.img_dpi)
        plt.close()


#==============================================================================#
def plot_memory(_plot_data,
                disp=False, output_dir=".", echo_dart=False):

    #
    if _matplotlib_backend is None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except:
            warnings.warn("Matplotlib could not find a suitable backend. Skipping plotting...")
            return

    import matplotlib
    import matplotlib.pyplot as plt

    filename = _plot_data.filename
    title = _plot_data.get_title()
    _params = _plot_data.plot_params
    _ext = 'memory'

    _plot_min = (_params.timing_max_value *
                 (0.01 * _params.timing_min_percent))

    _do_plot = plot_generic(_plot_data,
                            _params.memory_fields,
                            _plot_data.get_memory(),
                            _ext,
                            _plot_min,
                            'MB')

    if not _do_plot:
        return

    plt.xlabel('Memory [MB]')
    plt.title('Memory report for {}'.format(title))
    if disp:
        print('Displaying plot...')
        plt.show()
    else:
        make_output_directory(output_dir)
        imgfname = os.path.basename(filename)
        imgfname = imgfname.replace('.', '_{}.'.format(_ext))
        if not '_{}.'.format(_ext) in imgfname:
            imgfname += '_{}.'.format(_ext)
        imgfname = imgfname.replace('.json', '.{}'.format(_params.img_type))
        imgfname = imgfname.replace('.py', '.{}'.format(_params.img_type))
        if not '.{}'.format(_params.img_type) in imgfname:
            imgfname += '.{}'.format(_params.img_type)
        while '..' in imgfname:
            imgfname = imgfname.replace('..', '.')

        add_plotted_files(imgfname, os.path.join(output_dir, imgfname), echo_dart)

        imgfname = os.path.join(output_dir, imgfname)
        print('Saving plot: "{}"...'.format(imgfname))
        plt.savefig(imgfname, dpi=_params.img_dpi)
        plt.close()


#==============================================================================#
def plot_maximums(output_name, title, data, plot_params=plot_parameters(),
                  display=False, output_dir='.', echo_dart=None):
    """
    A function to plot JSON data

    Args:
        - data (list):
            - list of "plot_data" objects
            - should contain their own plot_parameters object
        - files (list):
            - list of JSON files
            - "plot_params" argument object will be applied to these files
        - combine (bool):
            - if specified, the plot_data objects from "data" and "files"
              will be combined into one "plot_data" object
            - the plot_params object will be used for this
    """
    try:
        # try here in case running as main on C++ output
        import timemory.options as options
        if echo_dart is None and options.echo_dart:
            echo_dart = True
        elif echo_dart is None:
            echo_dart = False
    except:
        pass

    _combined = None
    for _data in data:
        if _combined is None:
            _combined = plot_data(filename = output_name,
                                  output_name = output_name,
                                  concurrency = _data.concurrency,
                                  mpi_size = _data.mpi_size,
                                  timemory_functions = {},
                                  title = title,
                                  plot_params = plot_params)

        _key = list(_data.timemory_functions.keys())[0]
        _obj = _data.timemory_functions[_key]
        _obj_name = "{}".format(_data.filename)
        _obj.func = _obj_name
        _combined.timemory_functions[_obj_name] = _obj

    try:
        print ('Plotting {}...'.format(_combined.filename))
        plot_timing(_combined, display, output_dir, echo_dart)
        plot_memory(_combined, display, output_dir, echo_dart)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print ('Exception - {}'.format(e))
        print ('Error! Unable to plot "{}"...'.format(_combined.filename))


#==============================================================================#
def plot(data = [], files = [], plot_params=plot_parameters(),
         combine=False, display=False, output_dir='.', echo_dart=None):
    """
    A function to plot JSON data

    Args:
        - data (list):
            - list of "plot_data" objects
            - should contain their own plot_parameters object
        - files (list):
            - list of JSON files
            - "plot_params" argument object will be applied to these files
        - combine (bool):
            - if specified, the plot_data objects from "data" and "files"
              will be combined into one "plot_data" object
            - the plot_params object will be used for this
    """
    try:
        # try here in case running as main on C++ output
        import timemory.options as options
        if echo_dart is None and options.echo_dart:
            echo_dart = True
        elif echo_dart is None:
            echo_dart = False
    except:
        pass

    if len(files) > 0:
        for filename in files:
            print ('Reading {}...'.format(filename))
            f = open(filename, "r")
            _data = read(json.load(f))
            f.close()
            _data.filename = filename
            _data.title = filename
            data.append(_data)

    data_sum = None
    if combine:
        for _data in data:
            if data_sum is None:
                data_sum = _data
            else:
                data_sum += _data
        data_sum.update_params(plot_params)
        data = [data_sum]

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
        parser.add_argument("-f", "--files", nargs='*', help="File input", type=str)
        parser.add_argument("-d", "--display", required=False, action='store_true',
                            help="Display plot", dest='display_plot')
        parser.add_argument("-t", "--titles", nargs='*', help="Plot titles", type=str)
        parser.add_argument('-c', "--combine", required=False, action='store_true',
                            help="Combined data into a single plot")
        parser.add_argument('-o', '--output-dir', help="Output directory", type=str,
                            required=False)
        parser.add_argument('-e', '--echo-dart', help="echo Dart measurement for CDash",
                            required=False, action='store_true')
        parser.add_argument('--timing-percent', required=False, type=float,
            help="Exclude plotting times below this percentage of maximum")
        parser.add_argument('--memory-percent', required=False, type=float,
            help="Exclude plotting RSS values below this percentage of maximum")
        parser.add_argument('--timing-fields', required=False, nargs='*',
            help='Timing types to plot {}'.format(plot_parameters.timing_fields))
        parser.add_argument('--memory-fields', required=False, nargs='*',
            help='Memory types to plot {}'.format(plot_parameters.memory_fields))
        parser.add_argument('--img-dpi', help="Image dots per sq inch",
            required=False, type=int)
        parser.add_argument('--img-size', help="Image dimensions", nargs=2,
            required=False, type=int)
        parser.add_argument('--img-type', help="Image type",
            required=False, type=str)
        parser.add_argument('--plot-max',
            help="Plot the maximums from a set of inputs to <filename>",
            required=False, type=str, dest='plot_max')

        parser.set_defaults(display_plot=False)
        parser.set_defaults(combine=False)
        parser.set_defaults(output_dir=".")
        parser.set_defaults(echo_dart=False)
        parser.set_defaults(timing_percent=plot_parameters.timing_min_percent)
        parser.set_defaults(memory_percent=plot_parameters.memory_min_percent)
        parser.set_defaults(timing_fields=plot_parameters.timing_fields)
        parser.set_defaults(memory_fields=plot_parameters.memory_fields)
        parser.set_defaults(img_dpi=plot_parameters.img_dpi)
        parser.set_defaults(img_size=[ plot_parameters.img_size['w'],
                                       plot_parameters.img_size['h'] ])
        parser.set_defaults(img_type=plot_parameters.img_type)
        parser.set_defaults(plot_max="")

        args = parser.parse_args()

        do_plot_max = True if len(args.plot_max) > 0 else False

        print('Files: {}'.format(args.files))
        print('Titles: {}'.format(args.titles))

        params = plot_parameters(_timing_min_percent = args.timing_percent,
                                 _timing_fields = args.timing_fields,
                                 _memory_min_percent = args.memory_percent,
                                 _memory_fields = args.memory_fields,
                                 _img_dpi = args.img_dpi,
                                 _img_size = { 'w' : args.img_size[0],
                                               'h' : args.img_size[1] },
                                 _img_type = args.img_type)

        if do_plot_max:
            if len(args.titles) != 1:
                raise Exception("Error must provide one title")
        else:
            if len(args.titles) != 1 and len(args.titles) != len(args.files):
                raise Exception("Error must provide one title or a title for each file")

        data = []
        for i in range(len(args.files)):
            f = open(args.files[i], "r")
            _data = read(json.load(f))
            _data.filename = args.files[i].replace('.json', '')
            if len(args.titles) == 1:
                _data.title = args.titles[0]
            else:
                _data.title = args.titles[i]
            _data.plot_params = params
            print('### --> Processing "{}" from "{}"...'.format(_data.title,
                                                                args.files[i]))
            data.append(_data)

        if do_plot_max:
            plot_maximums(args.plot_max,
                          args.titles[0],
                          data,
                          plot_params=params,
                          display=args.display_plot,
                          output_dir=args.output_dir,
                          echo_dart=args.echo_dart)
        else:
            plot(data=data,
                 plot_params=params,
                 display=args.display_plot,
                 combine=args.combine,
                 output_dir=args.output_dir,
                 echo_dart=args.echo_dart)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print ('Exception - {}'.format(e))

    print ('Done - {}'.format(sys.argv[0]))
    sys.exit(0)
