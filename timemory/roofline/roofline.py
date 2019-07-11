#!/usr/bin/env python

import re
import matplotlib.pyplot as plt
import json

__all__ = ["plot_roofline"]

#f = open("timemory-test-cxx-roofline-output/cpu_roofline.json","r") 
#data = json.load(f)
#f.close()

def plot_roofline(data):
    work_set  = []
    bandwidth = []

    #lines=f.readlines()
    rank_data = data["rank"]
    full_data = rank_data["data"]
    roof_data = full_data["roofline"]
    print(roof_data)
    for x in roof_data:
        #ll = x.split(',')
        print(x)
        work_set.append(x["tuple_element0"])
        bandwidth.append(x["tuple_element2"])
        #work_set.append(int(ll[0].replace(" ","").replace("working-set=","")))
        #bandwidth.append(float(ll[2].replace(" ","").replace("bytes-per-sec=","")))
        #f.close()

    #print work_set
    #print bandwidth

    plt.yscale("log")
    plt.xscale("log")
    plt.plot(work_set, bandwidth)
    plt.show()

if __name__ == "__main__":
    import sys
    f = open(sys.argv[1])
    plot_roofline(json.load(f))
    
