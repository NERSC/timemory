#!/usr/bin/env python

import os
import sys
import argparse
import matplotlib.pyplot as plt
import json
import math
import numpy
import re

from matplotlib.pyplot import text
from math import log, pi


GIGABYTE = 1024 * 1024 * 1024
try:
    import timemory
    GIGABYTE = timemory.units.gigabyte
except:
    pass


__all__ = ['smooth',
           'get_peak_flops',
           'get_peak_bandwidth',
           'get_hotspots',
           'get_color',
           'plot_parameters',
           'plot_roofline']

#==============================================================================#
def smooth(x,y):
    """
    Smooth a curve
    """
    xs = x[:]
    ys = y[:]
    d = 0
    for i in range(0,len(ys)):
        num = min(len(ys),i+d+1) - max(0,i-d)
        total = sum(ys[max(0,i-d):min(len(ys),i+d+1)])
        ys[i] = total/float(num)
    return xs,ys


#==============================================================================#
def get_peak_flops(roof_data, flops_info):
    """
    Get the peak floating point operations / sec
    """
    flops_data = []

    for element in roof_data:
        flops_data.append(element["tuple_element6"]/GIGABYTE)
    info_list = re.sub(r'[^\w]', ' ', flops_info).split()
    info = info_list[0] + " GFLOPs/sec"
    peak_flops = [max(flops_data), info]
    return peak_flops

    
#==============================================================================#
def get_peak_bandwidth(roof_data):
    """
    Get multi-level bandwidth peaks - Implementation from ERT:
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit
    """
    ref_intensity  = 0
    work_set       = []
    bandwidth_data = []

    # Read bandwidth raw data
    for element in roof_data:
        intensity = element["tuple_element7"]
        if ref_intensity == 0:
            ref_intensity = intensity
        if intensity != ref_intensity:
            break
        work_set.append(element["tuple_element0"])
        bandwidth_data.append(element["tuple_element5"]/ GIGABYTE)
    fraction = 1.05
    samples  = 10000

    max_bandwidth = max(bandwidth_data)
    begin = bandwidth_data.index(max_bandwidth)

    work_set = work_set[begin:]
    bandwidth_data = bandwidth_data[begin:]
    min_bandwidth = min(bandwidth_data)

    dband = max_bandwidth/float(samples - 1)

    counts = samples*[0]
    totals = samples*[0.0]

    work_set,bandwidth_data = smooth(work_set,bandwidth_data)

    for i in range(0,samples):
        cband = i*dband
        for bandwidth in bandwidth_data:
            if bandwidth >= cband/fraction and bandwidth <= cband*fraction:
                totals[i] += bandwidth
                counts[i] += 1

    band_list = [[1000*max_bandwidth, 1000]]
    
    maxc = -1
    maxi = -1
    
    for i in range(samples-3,1,-1):
        if counts[i] > 10:
            if counts[i] > maxc:
                maxc = counts[i]
                maxi = i
        else:
            if maxc > 1:
                value = float(totals[maxi])/max(1,counts[maxi])
                if 1.10*value < float(band_list[-1][0])/band_list[-1][1]:
                    band_list.append([totals[maxi],counts[maxi]])
                else:
                    band_list[-1][0] += totals[maxi]
                    band_list[-1][1] += counts[maxi]
            maxc = -1
            maxi = -1
            
    band_info_list = ["DRAM"]
    cache_num = len(band_list)-1
    
    for cache in range(1,cache_num+1):
        band_info_list = ["L%d" % (cache_num+1 - cache)] + band_info_list
        
    peak_bandwidths = []
    for (band,band_info) in zip(band_list,band_info_list):
        band_info = band_info + " GB/s"
        peak_bandwidths.append([float(band[0]/band[1]), band_info])
    return peak_bandwidths


#==============================================================================#
def get_hotspots(op_data, ai_data):
    """
    Get the hotspots information
    """
    op_graph_data = op_data["graph"]
    ai_graph_data = ai_data["graph"]
    hotspots   = []
    
    avg_runtime = 0.0
    max_runtime = 0.0
    for i in range(0, len(op_graph_data)):
        op_runtime = float(op_graph_data[0]["tuple_element1"]["accum"]["second"])
        ai_runtime = float(ai_graph_data[0]["tuple_element1"]["accum"]["second"])
        avg_runtime += 0.5 * (op_runtime + ai_runtime)
        max_runtime = max([max_runtime, 0.5 * (op_runtime + ai_runtime)])

    if len(op_graph_data) > 1:
        avg_runtime -= max_runtime
        avg_runtime /= len(op_graph_data) - 1
    
    for i in range(0, len(op_graph_data)):
        runtime   = float(op_graph_data[i]["tuple_element1"]["accum"]["second"])
        flop      = float(op_graph_data[i]["tuple_element1"]["accum"]["first"]["value0"])
        bandwidth = float(ai_graph_data[i]["tuple_element1"]["accum"]["first"]["value1"])
        label     = op_graph_data[i]["tuple_element2"]

        intensity  = flop / bandwidth
        flop       = flop / GIGABYTE
        proportion = runtime / avg_runtime
        label      = label.replace("> [cxx] ", "")

        # this can arise from overflow
        if flop < 0 or bandwidth < 0:
            continue
        print("{} : runtime = {}, avg = {}, proportion = {}".format(label, runtime, avg_runtime, proportion))
        hotspots.append([intensity, flop, proportion, label])
    return hotspots


#==============================================================================#
def get_color(proportion):
    if proportion < 0.01:
        color = "lawngreen"
    elif proportion < 0.1:
        color = "yellow"
    else:
        color = "red"
    return color


#==============================================================================#
class plot_parameters():
    def __init__(self, peak_flops, hotspots):
        y_digits = int(math.log10(peak_flops[0]))+1
        self.xmin = 0.01
        self.xmax = 100
        self.ymin = 1
        self.ymax = 10**y_digits
        self.xlabel = ('Arithmetic Intensity [FLOPs/Byte]')
        self.ylabel = ('Performance [GFLOPS/sec]')

        for element in (hotspots):
            intensity = element[0]
            flop      = element[1]
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
def plot_roofline(ai_data, op_data, display = False, fname = "roofline",
                  image_type = "png", output_dir = os.getcwd(), title = "Roofline Plot",
                  width = 1600, height = 1200, dpi = 75):
    """
    Plot the roofline
    """
    roof_data = ai_data["rank"]["data"]["roofline"]["ptr_wrapper"]["data"]["data"]
    flops_info = op_data["rank"]["data"]["unit_repr"]

    peak_bandwidths = get_peak_bandwidth(roof_data)
    peak_flops      = get_peak_flops(roof_data, flops_info)
    hotspots        = get_hotspots(op_data["rank"]["data"], ai_data["rank"]["data"])
    
    plot_params = plot_parameters(peak_flops, hotspots)

    f = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = f.add_subplot(111)

    plt.title(title)
    plt.yscale("log")
    plt.xscale("log")

    plt.xlabel(plot_params.xlabel)
    plt.ylabel(plot_params.ylabel)
        
    axes = plt.gca()
    axes.set_xlim([plot_params.xmin, plot_params.xmax])
    axes.set_ylim([plot_params.ymin, plot_params.ymax])
    
    # plot bandwidth roof
    x0 = plot_params.xmax
    for band in (peak_bandwidths):
        x1 = plot_params.xmin
        y1 = band[0] * plot_params.xmin
        if y1 < plot_params.ymin:
            x1 = plot_params.ymin / band[0]
            y1 = plot_params.ymin
        x2 = peak_flops[0] / band[0]
        y2 = peak_flops[0]
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

        text(x_text, y_text, "%.2f %s" % (band[0], band[1]), rotation=angle, rotation_mode='anchor')
        plt.plot([x1, x2], [y1, y2], color='magenta')

    # plot computing roof
    text(plot_params.xmax, peak_flops[0] + 2, "%.2f %s" % (peak_flops[0],
        peak_flops[1]), horizontalalignment='right')
    plt.plot([x0, plot_params.xmax], [peak_flops[0], peak_flops[0]], color='b')

    # plot hotspots
    for element in (hotspots):
        print(element[0], element[1])
        c = get_color(element[2])
        plt.scatter(element[0], element[1], color=c)
        text(element[0], element[1], "%s" % element[3], rotation=0, rotation_mode='anchor')

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
