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

import sys
import operator
import subprocess
import os
import glob
import filecmp
import math
import socket
import platform
import time
import json
import optparse
import ast

from .utils import *

def text_list_2_string(text_list):
  return reduce(operator.add,[t+" " for t in text_list])

class ert_core:

  def __init__(self):
    self.ert_version = "1.1.0"

    self.dict = {}
    self.metadata = {}

    self.metadata["ERT_VERSION"] = self.ert_version

    hostname = socket.gethostname()
    try:
      new_hostname = socket.gethostbyaddr(hostname)
    except socket.herror:
      new_hostname = hostname

    hostname = new_hostname

    hostname = os.getenv("NERSC_HOST",hostname)

    self.metadata["HOSTNAME"] = hostname
    self.metadata["UNAME"] = platform.uname()

  def build_only(self, option, opt, value, parser):
    parser.values.build = True
    parser.values.run   = False
    parser.values.post  = False

  def run_only(self, option, opt, value, parser):
    parser.values.build = False
    parser.values.run   = True
    parser.values.post  = False

  def post_only(self, option, opt, value, parser):
    parser.values.build = False
    parser.values.run   = False
    parser.values.post  = True

  def flags(self):
    parser = optparse.OptionParser(usage="%prog [-h] [--help] [options] config_file",version="%prog " + self.ert_version)

    build_group = optparse.OptionGroup(parser,"Build options");

    build_group.add_option("--build",dest="build",action="store_true",default=True,help="Build the micro-kernels [default]")
    build_group.add_option("--no-build",dest="build",action="store_false",default=True,help="Don't build the micro-kernels")
    build_group.add_option("--build-only",dest="build",action="callback",callback=self.build_only,help="Only build the micro-kernels")
    parser.add_option_group(build_group)

    run_group = optparse.OptionGroup(parser,"Run options");

    run_group.add_option("--run",dest="run",action="store_true",default=True,help="Run the micro-kernels [default]")
    run_group.add_option("--no-run",dest="run",action="store_false",default=True,help="Don't run the micro-kernels")
    run_group.add_option("--run-only",dest="run",action="callback",callback=self.run_only,help="Only run the micro-kernels")
    parser.add_option_group(run_group)

    post_group = optparse.OptionGroup(parser,"Post-processing options");

    post_group.add_option("--post",dest="post",action="store_true",default=True,help="Run the post-processing [default]")
    post_group.add_option("--no-post",dest="post",action="store_false",default=True,help="Don't run the post-processing")
    post_group.add_option("--post-only",dest="post",action="callback",callback=self.post_only,help="Only run the post-processing")

    post_group.add_option("--gnuplot",dest="gnuplot",action="store_true",default=True,help="Generate graphs using GNUplot [default]")
    post_group.add_option("--no-gnuplot",dest="gnuplot",action="store_false",default=True,help="Don't generate graphs using GNUplot")

    parser.add_option_group(post_group)

    parser.add_option("--verbose",dest="verbose",action="store",nargs=1,default=1,type=int,help="Set the verbosity of the screen output [default = %default].    = 0 : no output,                                         = 1 : outlines progress,                                 = 2 : good for debugging (prints all commands)")
    parser.add_option("--quiet",dest="verbose",action="store_const",const=0,help="Don't generate any screen output, '--verbose=0'")

    (options,args) = parser.parse_args()

    def nullusage(fd):
      fd.write("\n")

    if len(args) < 1:
      parser.print_help()
      parser.print_usage = nullusage
      parser.error("no configuration file given")

    if len(args) > 1:
      parser.print_help()
      parser.print_usage = nullusage
      parser.error("more than one configuration file given")

    self.options = options

    self.configure_filename = args[0]
    return 0

  def configure(self):
    if self.options.verbose > 0:
      print()
      print ("Reading configuration from '%s'..." % self.configure_filename)

    try:
      configure_file = open(self.configure_filename,"r")
    except IOError:
      sys.stderr.write("Unable to open '%s'...\n" % self.configure_filename)
      return 1

    self.dict["CONFIG"] = {}

    for line in configure_file:
      line = line[:-1]

      if len(line) > 0 and line[0] != "#":
        line = line.split()
        if len(line) > 0:
          target = line[0]
          value = line[1:]

          if len(target) > 0:
            self.dict["CONFIG"][target] = value

    if "ERT_MPI" not in self.dict["CONFIG"]:
      self.dict["CONFIG"]["ERT_MPI"] = [False]

    if "ERT_OPENMP" not in self.dict["CONFIG"]:
      self.dict["CONFIG"]["ERT_OPENMP"] = [False]

    if "ERT_GPU" not in self.dict["CONFIG"]:
      self.dict["CONFIG"]["ERT_GPU"] = [False]

    self.results_dir = self.dict["CONFIG"]["ERT_RESULTS"][0]
    made_results = make_dir_if_needed(self.results_dir,"results",False)

    if self.options.verbose > 0:
      if made_results:
        print ("  Making new results directory, %s..." % self.results_dir)
      else:
        print ("  Using existing results directory, %s..." % self.results_dir)

    run_files = glob.glob("%s/Run.[0-9][0-9][0-9]" % self.results_dir)
    used_run_files = []
    used_run_list = []
    no_dir = True
    for run_file in run_files:
      run_configure_filename = "%s/config.ert" % run_file
      if os.path.exists(run_configure_filename):
        if filecmp.cmp(self.configure_filename,run_configure_filename):
          self.results_dir = run_file
          no_dir = False
          if self.options.verbose > 0:
            print ("    Using existing run directory, %s..." % self.results_dir)
          break
        else:
          used_run_files.append(run_file)
          used_run_list.append(int(run_file[-3:]))
      else:
        used_run_files.append(run_file)
        used_run_list.append(int(run_file[-3:]))

    if no_dir:
      if self.options.build or self.options.run:
        if len(used_run_list) == 0:
          used_run_list = [0]
        for n in xrange(1,max(used_run_list)+2):
          if n not in used_run_list:
            self.results_dir = "%s/Run.%03d" % (self.results_dir,n)
            if self.options.verbose > 0:
              if made_results:
                print ("    Making new run directory, '%s'..." % self.results_dir)
              else:
                print
                print ("*** WARNING ***")
                print ("**")
                print ("**  Making new run directory, '%s'," % self.results_dir)
                print ("**    because the current connfiguration file, '%s' " % self.configure_filename)
                print ("**    doesn't match the configuration files, 'config.ert', under:")
                print ("**")
                for u in sorted(used_run_files):
                  print ("**      %s" % u)
                print ("**")
                print ("*** WARNING ***")

            command = ["mkdir",self.results_dir]
            if execute_noshell(command,self.options.verbose > 1) != 0:
              sys.stderr.write("Unable to make new run directory, '%s'\n" % self.results_dir)
              return 1

            command = ["cp",self.configure_filename,"%s/config.ert" % self.results_dir]
            if execute_noshell(command,self.options.verbose > 1) != 0:
              sys.stderr.write("Unable to copy configuration file, '%s', into new run directory, %s\n" % (self.configure_filename,self.results_dir))
              return 1

            break
      else:
        sys.stderr.write("\nNo run directory for '%s' found under '%s'\n" % (self.configure_filename,self.results_dir))
        return 1

    if self.options.verbose > 0:
      print()

    return 0

  def build(self):
    if self.options.build:
      if self.options.verbose > 0:
        if self.options.verbose > 1:
          print()
        print ("  Building ERT core code...")

      command_prefix =                                                       \
        self.dict["CONFIG"]["ERT_CC"]                                                + \
        self.dict["CONFIG"]["ERT_CFLAGS"]                                            + \
        ["-I%s/Kernels" % self.exe_path]                                   + \
        ["-DERT_FLOP=%d" % self.flop]                                      + \
        ["-DERT_ALIGN=%s" % self.dict["CONFIG"]["ERT_ALIGN"][0]]                     + \
        ["-DERT_MEMORY_MAX=%s" % self.dict["CONFIG"]["ERT_MEMORY_MAX"][0]]           + \
        ["-DERT_WORKING_SET_MIN=%s" % self.dict["CONFIG"]["ERT_WORKING_SET_MIN"][0]] + \
        ["-DERT_TRIALS_MIN=%s" % self.dict["CONFIG"]["ERT_TRIALS_MIN"][0]]

      if self.dict["CONFIG"]["ERT_MPI"][0] == "True":
        command_prefix += ["-DERT_MPI"] + self.dict["CONFIG"]["ERT_MPI_CFLAGS"]

      if self.dict["CONFIG"]["ERT_OPENMP"][0] == "True":
        command_prefix += ["-DERT_OPENMP"] + self.dict["CONFIG"]["ERT_OPENMP_CFLAGS"]

      if self.dict["CONFIG"]["ERT_GPU"][0] == "True":
        command_prefix += ["-DERT_GPU"] + self.dict["CONFIG"]["ERT_GPU_CFLAGS"]

      command = command_prefix + \
                ["-c","%s/Drivers/%s.c" % (self.exe_path,self.dict["CONFIG"]["ERT_DRIVER"][0])] + \
                ["-o","%s/%s.o" % (self.flop_dir,self.dict["CONFIG"]["ERT_DRIVER"][0])]
      if execute_noshell(command,self.options.verbose > 1) != 0:
        sys.stderr.write("Compiling driver, %s, failed\n" % self.dict["CONFIG"]["ERT_DRIVER"][0])
        return 1

      command = command_prefix + \
                ["-c","%s/Kernels/%s.c" % (self.exe_path,self.dict["CONFIG"]["ERT_KERNEL"][0])] + \
                ["-o","%s/%s.o" % (self.flop_dir,self.dict["CONFIG"]["ERT_KERNEL"][0])]
      if execute_noshell(command,self.options.verbose > 1) != 0:
        sys.stderr.write("Compiling kernel, %s, failed\n" % self.dict["CONFIG"]["ERT_KERNEL"][0])
        return 1

      command = self.dict["CONFIG"]["ERT_LD"]      + \
                self.dict["CONFIG"]["ERT_LDFLAGS"]

      if self.dict["CONFIG"]["ERT_MPI"][0] == "True":
        command += self.dict["CONFIG"]["ERT_MPI_LDFLAGS"]

      if self.dict["CONFIG"]["ERT_OPENMP"][0] == "True":
        command += self.dict["CONFIG"]["ERT_OPENMP_LDFLAGS"]

      if self.dict["CONFIG"]["ERT_GPU"][0] == "True":
        command += self.dict["CONFIG"]["ERT_GPU_LDFLAGS"]

      command += ["%s/%s.o" % (self.flop_dir,self.dict["CONFIG"]["ERT_DRIVER"][0])] + \
                 ["%s/%s.o" % (self.flop_dir,self.dict["CONFIG"]["ERT_KERNEL"][0])] + \
                 self.dict["CONFIG"]["ERT_LDLIBS"]                                  + \
                 ["-o","%s/%s.%s" % (self.flop_dir,self.dict["CONFIG"]["ERT_DRIVER"][0],self.dict["CONFIG"]["ERT_KERNEL"][0])]
      if execute_noshell(command,self.options.verbose > 1) != 0:
        sys.stderr.write("Linking code failed\n")
        return 1

    return 0

  def add_metadata(self,outputname):
    try:
      output = open(outputname,"a")
    except IOError:
      sys.stderr.write("Unable to open output file, %s, to add metadata\n" % outputfile)
      return 1

    for k,v in self.metadata.items():
      output.write("%s  %s\n" % (k,v))

    for k,v in self.dict.items():
      output.write("%s  %s\n" % (k,v))

    output.close()

    return 0

  def run(self):
    if self.options.run:
      if self.options.verbose > 0:
        if self.options.verbose > 1:
          print()
        print ("  Running ERT core code...")

    self.run_list = []

    if self.dict["CONFIG"]["ERT_MPI"][0] == "True":
      mpi_procs_list = parse_int_list(self.dict["CONFIG"]["ERT_MPI_PROCS"][0])
    else:
      mpi_procs_list = [1]

    if self.dict["CONFIG"]["ERT_OPENMP"][0] == "True":
      openmp_threads_list = parse_int_list(self.dict["CONFIG"]["ERT_OPENMP_THREADS"][0])
    else:
      openmp_threads_list = [1]

    if self.dict["CONFIG"]["ERT_MPI"][0] == "True":
      if self.dict["CONFIG"]["ERT_OPENMP"][0] == "True":
        procs_threads_list = parse_int_list(self.dict["CONFIG"]["ERT_PROCS_THREADS"][0])
      else:
        procs_threads_list = mpi_procs_list
    else:
      if self.dict["CONFIG"]["ERT_OPENMP"][0] == "True":
        procs_threads_list = openmp_threads_list
      else:
        procs_threads_list = [1]

    if self.dict["CONFIG"]["ERT_GPU"][0] == "True":
      gpu_blocks_list = parse_int_list(self.dict["CONFIG"]["ERT_GPU_BLOCKS"][0])
    else:
      gpu_blocks_list = [1]

    if self.dict["CONFIG"]["ERT_GPU"][0] == "True":
      gpu_threads_list = parse_int_list(self.dict["CONFIG"]["ERT_GPU_THREADS"][0])
    else:
      gpu_threads_list = [1]

    if self.dict["CONFIG"]["ERT_GPU"][0] == "True":
      blocks_threads_list = parse_int_list(self.dict["CONFIG"]["ERT_BLOCKS_THREADS"][0])
    else:
      blocks_threads_list = [1]

    num_experiments = int(self.dict["CONFIG"]["ERT_NUM_EXPERIMENTS"][0])

    base_command = list_2_string(self.dict["CONFIG"]["ERT_RUN"])

    for mpi_procs in mpi_procs_list:
      for openmp_threads in openmp_threads_list:
        if mpi_procs * openmp_threads in procs_threads_list:
          for gpu_blocks in gpu_blocks_list:
            for gpu_threads in gpu_threads_list:
              if gpu_blocks * gpu_threads in blocks_threads_list:
                print_str = ("")

                if self.dict["CONFIG"]["ERT_MPI"][0] == "True":
                  mpi_dir = "%s/MPI.%04d" % (self.flop_dir,mpi_procs)
                  print_str += "MPI %d, " % mpi_procs
                else:
                  mpi_dir = self.flop_dir

                if self.options.run:
                  make_dir_if_needed(mpi_dir,"run",self.options.verbose > 1)

                if self.dict["CONFIG"]["ERT_OPENMP"][0] == "True":
                  openmp_dir = "%s/OpenMP.%04d" % (mpi_dir,openmp_threads)
                  print_str += "OpenMP %d, " % openmp_threads
                else:
                  openmp_dir = mpi_dir

                if self.options.run:
                  make_dir_if_needed(openmp_dir,"run",self.options.verbose > 1)

                if self.dict["CONFIG"]["ERT_GPU"][0] == "True":
                  gpu_dir = "%s/GPU_Blocks.%04d" % (openmp_dir,gpu_blocks)
                  print_str += "GPU blocks %d, " % gpu_blocks
                else:
                  gpu_dir = openmp_dir

                if self.options.run:
                  make_dir_if_needed(gpu_dir,"run",self.options.verbose > 1)

                if self.dict["CONFIG"]["ERT_GPU"][0] == "True":
                  run_dir = "%s/GPU_Threads.%04d" % (gpu_dir,gpu_threads)
                  print_str += "GPU threads %d, " % gpu_threads
                else:
                  run_dir = gpu_dir

                if self.options.run:
                  make_dir_if_needed(run_dir,"run",self.options.verbose > 1)

                self.run_list.append(run_dir)

                if print_str == "":
                  print_str = "serial"
                else:
                  print_str = print_str[:-2]

                if self.options.run:
                  if os.path.exists("%s/run.done" % run_dir):
                    if self.options.verbose > 1:
                      print ("    Skipping %s - already run" % print_str)
                  else:
                    if self.options.verbose > 0:
                      print ("    %s" % print_str)

                    command = base_command

                    command = command.replace("ERT_OPENMP_THREADS",str(openmp_threads))
                    command = command.replace("ERT_MPI_PROCS",str(mpi_procs))

                    if self.dict["CONFIG"]["ERT_GPU"][0] != "True":
                      command = command.replace("ERT_CODE","%s/%s.%s" % (self.flop_dir,self.dict["CONFIG"]["ERT_DRIVER"][0],self.dict["CONFIG"]["ERT_KERNEL"][0]))
                    else:
                      command = command.replace("ERT_CODE","%s/%s.%s %d %d" % (self.flop_dir,self.dict["CONFIG"]["ERT_DRIVER"][0],self.dict["CONFIG"]["ERT_KERNEL"][0],gpu_blocks,gpu_threads))

                    command = "(" + command + ") > %s/try.ERT_TRY_NUM 2>&1 " % run_dir

                    for t in xrange(1,num_experiments+1):
                      output = "%s/try.%03d" % (run_dir,t)

                      cur_command = command
                      cur_command = cur_command.replace("ERT_TRY_NUM","%03d" % t)

                      self.metadata["TIMESTAMP_DATA"] = time.time()

                      if execute_shell(cur_command,self.options.verbose > 1) != 0:
                        sys.stderr.write("Unable to complete %s, experiment %d\n" % (run_dir,t))
                        return 1

                      if self.add_metadata(output) != 0:
                        return 1

                    command = ["touch","%s/run.done" % run_dir]
                    if execute_noshell(command,self.options.verbose > 1) != 0:
                      sys.stderr.write("Unable to make 'run.done' file in %s\n" % run_dir)
                      return 1

                  if self.options.verbose > 1:
                    print()

    return 0

  def process(self):
    if self.options.post:
      if self.options.verbose > 0:
        print ("  Processing results..."
)
      for run in self.run_list:
        if self.options.verbose > 1:
          print ("   ",run)

        command = ["cat %s/try.* | %s/Scripts/preprocess.py > %s/pre" % (run,self.exe_path,run)]
        if execute_shell(command,self.options.verbose > 1) != 0:
          sys.stderr.write("Unable to process %s\n" % run)
          return 1

        command = ["%s/Scripts/maximum.py < %s/pre > %s/max" % (self.exe_path,run,run)]
        if execute_shell(command,self.options.verbose > 1) != 0:
          sys.stderr.write("Unable to process %s\n" % run)
          return 1

        command = ["%s/Scripts/summary.py < %s/max > %s/sum" % (self.exe_path,run,run)]
        if execute_shell(command,self.options.verbose > 1) != 0:
          sys.stderr.write("Unable to process %s\n" % run)
          return 1

        if self.options.verbose > 1:
          print()

    return 0

  def make_graph(self,run_dir,title,name):
    command  = "sed "
    command += "-e 's#ERT_TITLE#%s#g' " % title
    command += "-e 's#ERT_XRANGE_MIN#\*#g' "
    command += "-e 's#ERT_XRANGE_MAX#\*#g' "
    command += "-e 's#ERT_YRANGE_MIN#\*#g' "
    command += "-e 's#ERT_YRANGE_MAX#\*#g' "
    command += "-e 's#ERT_RAW_DATA#%s/pre#g' " % run_dir
    command += "-e 's#ERT_MAX_DATA#%s/max#g' " % run_dir
    command += "-e 's#ERT_GRAPH#%s/%s#g' " % (run_dir,name)

    command += "< %s/Plot/%s.gnu.template > %s/%s.gnu" % (self.exe_path,name,run_dir,name)
    if execute_shell(command,False) != 0:
      sys.stderr.write("Unable to produce a '%s' gnuplot file for %s\n" % (name,run_dir))
      return 1

    command = "echo 'load \"%s/%s.gnu\"' | %s" % (run_dir,name,self.dict["CONFIG"]["ERT_GNUPLOT"][0])
    if execute_shell(command,self.options.verbose > 1) != 0:
      sys.stderr.write("Unable to produce a '%s' for %s\n" % (name,run_dir))
      return 1

    return 0

  def graphs(self):
    if self.options.post and self.options.gnuplot:
      for run_dir in self.run_list:
        if self.make_graph(run_dir,"Graph 1 (%s)" % run_dir,"graph1") != 0:
          return 1

        if self.make_graph(run_dir,"Graph 2 (%s)" % run_dir,"graph2") != 0:
          return 1

        if self.make_graph(run_dir,"Graph 3 (%s)" % run_dir,"graph3") != 0:
          return 1

        if self.make_graph(run_dir,"Graph 4 (%s)" % run_dir,"graph4") != 0:
          return 1

        if self.options.verbose > 1:
          print()

    return 0

  def merge_metadata(self,in_sub_meta1,in_sub_meta2):
    out_meta = {}
    out_sub_meta1 = {}
    out_sub_meta2 = {}

    in_sub_k1 = set(in_sub_meta1.keys())
    in_sub_k2 = set(in_sub_meta2.keys())

    out_k = in_sub_k1 & in_sub_k2

    out_sub_k1 = in_sub_k1 - in_sub_k2
    out_sub_k2 = in_sub_k2 - in_sub_k1

    for k in out_k:
      if in_sub_meta1[k] == in_sub_meta2[k]:
        out_meta[k] = in_sub_meta1[k]
      else:
        out_sub_k1.add(k)
        out_sub_k2.add(k)

    for k in out_sub_k1:
      out_sub_meta1[k] = in_sub_meta1[k]

    for k in out_sub_k2:
      out_sub_meta2[k] = in_sub_meta2[k]

    return(out_meta,out_sub_meta1,out_sub_meta2)

  def build_database(self,gflop,gbyte):
    gflop0 = gflop[0].split()

    emp_gflops_data = []
    emp_gflops_data.append([gflop0[1],float(gflop0[0])])

    emp_gflops_metadata = {}
    for metadata in gflop[1:]:
      parts = metadata.partition(" ")
      key = parts[0].strip()
      if key != "META_DATA":
        try:
          new_value = ast.literal_eval(parts[2].strip())
        except (SyntaxError,ValueError):
          new_value = parts[2].strip()

        if key in emp_gflops_metadata:
          value = emp_gflops_metadata[key]

          if isinstance(value,list):
            value.append(new_value)
          else:
            value = [value,new_value]

          emp_gflops_metadata[key] = value
        else:
          emp_gflops_metadata[parts[0].strip()] = new_value

    emp_gflops_metadata["TIMESTAMP_DB"] = time.time()

    emp_gflops = {}
    emp_gflops['data'] = emp_gflops_data

    emp_gbytes_metadata = {}
    emp_gbytes_data = []

    for i in xrange(0,len(gbyte)):
      if gbyte[i] == "META_DATA":
        break
      else:
        gbyte_split = gbyte[i].split()
        emp_gbytes_data.append([gbyte_split[1],float(gbyte_split[0])])

    for j in xrange(i+1,len(gbyte)):
      metadata = gbyte[j]

      parts = metadata.partition(" ")
      key = parts[0].strip()
      if key != "META_DATA":
        try:
          new_value = ast.literal_eval(parts[2].strip())
        except (SyntaxError,ValueError):
          new_value = parts[2].strip()

        if key in emp_gbytes_metadata:
          value = emp_gbytes_metadata[key]

          if isinstance(value,list):
            value.append(new_value)
          else:
            value = [value,new_value]

          emp_gbytes_metadata[key] = value
        else:
          emp_gbytes_metadata[parts[0].strip()] = new_value

    emp_gbytes_metadata["TIMESTAMP_DB"] = time.time()

    emp_gbytes = {}
    emp_gbytes['data'] = emp_gbytes_data

    (emp_metadata,emp_gflops_metadata,emp_gbytes_metadata) = self.merge_metadata(emp_gflops_metadata,emp_gbytes_metadata)

    emp_gflops['metadata'] = emp_gflops_metadata
    emp_gbytes['metadata'] = emp_gbytes_metadata

    empirical = {}
    empirical['metadata'] = emp_metadata
    empirical['gflops']    = emp_gflops
    empirical['gbytes']    = emp_gbytes

    spec_gflops_data = []
    if 'ERT_SPEC_GFLOPS' in self.dict['CONFIG']:
      spec_gflops_data.append(['GFLOPs',float(self.dict['CONFIG']['ERT_SPEC_GFLOPS'][0])])

    spec_gflops = {}
    spec_gflops['data'] = spec_gflops_data

    spec_gbytes_data = []
    for k in self.dict['CONFIG']:
      if k.find('ERT_SPEC_GBYTES') == 0:
        spec_gbytes_data.append([k[len('ERT_SPEC_GBYTES')+1:],float(self.dict['CONFIG'][k][0])])

    spec_gbytes = {}
    spec_gbytes['data'] = spec_gbytes_data

    spec = {}
    spec['gflops'] = spec_gflops
    spec['gbytes'] = spec_gbytes

    result = {}
    result['empirical'] = empirical
    result['spec']      = spec

    return result

  def roofline(self):
    if self.options.post:
      if self.options.verbose > 0:
        print ("Gathering the final roofline results...")

      depth_string = "/*"
      if self.dict["CONFIG"]["ERT_MPI"][0] == "True":
        depth_string += "/*"
      if self.dict["CONFIG"]["ERT_OPENMP"][0] == "True":
        depth_string += "/*"
      if self.dict["CONFIG"]["ERT_GPU"][0] == "True":
        depth_string += "/*/*"

      command = "cat %s%s/sum | %s/Scripts/roofline.py" % (self.results_dir,depth_string,self.exe_path)
      result = stdout_shell(command,self.options.verbose > 1)
      if result[0] != 0:
        sys.stderr.write("Unable to create final roofline results\n")
        return 1

      lines = result[1].split("\n")

      for i in xrange(0,len(lines)):
        if len(lines[i]) == 0:
          break

      gflop_lines = lines[:i]
      gbyte_lines = lines[i+1:-1]

      database = self.build_database(gflop_lines,gbyte_lines)

      database_filename = "%s/roofline.json" % self.results_dir
      try:
        database_file = open(database_filename,"w")
      except IOError:
        sys.stderr.write("Unable to open database file, %s\n" % database_filename)
        return 1

      json.dump(database,database_file,indent=3)

      database_file.close()

      line = gflop_lines[0].split()
      gflops_emp = [float(line[0]),line[1]]

      for i in xrange(0,len(gbyte_lines)):
        if gbyte_lines[i] == "META_DATA":
          break

      num_mem = i
      gbytes_emp = num_mem * [0]

      for i in xrange(0,num_mem):
        line = gbyte_lines[i].split()
        gbytes_emp[i] = [float(line[0]),line[1]]

      x = num_mem * [0.0]
      for i in xrange(0,len(gbytes_emp)):
        x[i] = gflops_emp[0]/gbytes_emp[i][0]

      if self.options.gnuplot:
        basename = "roofline"
        loadname = "%s/%s.gnu" % (self.results_dir,basename)

        xmin =   0.01
        xmax = 100.00

        ymin = 10 ** int(math.floor(math.log10(gbytes_emp[0][0] * xmin)))

        title = "Empirical Roofline Graph (%s)" % self.results_dir

        command  = "sed "
        command += "-e 's#ERT_TITLE#%s#g' " % title
        command += "-e 's#ERT_XRANGE_MIN#%le#g' " % xmin
        command += "-e 's#ERT_XRANGE_MAX#%le#g' " % xmax
        command += "-e 's#ERT_YRANGE_MIN#%le#g' " % ymin
        command += "-e 's#ERT_YRANGE_MAX#\*#g' "
        command += "-e 's#ERT_GRAPH#%s/%s#g' " % (self.results_dir,basename)

        command += "< %s/Plot/%s.gnu.template > %s" % (self.exe_path,basename,loadname)
        if execute_shell(command,False) != 0:
          sys.stderr.write("Unable to produce a '%s' gnuplot file for %s\n" % (loadname,self.results_dir))
          return 1

        try:
          plotfile = open(loadname,"a")
        except IOError:
          sys.stderr.write("Unable to open '%s'...\n" % loadname)
          return 1

        xgflops = 2.0
        label = '%.1f %s/sec (Maximum)' % (gflops_emp[0],gflops_emp[1])
        plotfile.write("set label '%s' at %.7le,%.7le left textcolor rgb '#000080'\n" % (label,xgflops,1.2*gflops_emp[0]))

        xleft  = xmin
        xright = x[0]

        xmid = math.sqrt(xleft * xright)
        ymid = gbytes_emp[0][0] * xmid

        y0gbytes = ymid
        x0gbytes = y0gbytes/gbytes_emp[0][0]

        C = x0gbytes * y0gbytes

        alpha = 1.065

        label_over = True
        for i in xrange(0,len(gbytes_emp)):
          if i > 0:
            if label_over and gbytes_emp[i-1][0] / gbytes_emp[i][0] < 1.5:
              label_over = False

            if not label_over and gbytes_emp[i-1][0] / gbytes_emp[i][0] > 3.0:
              label_over = True

          if label_over:
            ygbytes = math.sqrt(C * gbytes_emp[i][0]) / math.pow(alpha,len(gbytes_emp[i][1]))
            xgbytes = ygbytes/gbytes_emp[i][0]

            ygbytes *= 1.1
            xgbytes /= 1.1
          else:
            ygbytes = math.sqrt(C * gbytes_emp[i][0]) / math.pow(alpha,len(gbytes_emp[i][1]))
            xgbytes = ygbytes/gbytes_emp[i][0]

            ygbytes /= 1.1
            xgbytes *= 1.1

          label = "%s - %.1lf GB/s" % (gbytes_emp[i][1],gbytes_emp[i][0])

          plotfile.write("set label '%s' at %.7le,%.7le left rotate by 45 textcolor rgb '#800000'\n" % (label,xgbytes,ygbytes))

        plotfile.write("plot \\\n")

        for i in xrange(0,len(gbytes_emp)):
          plotfile.write("     (x <= %.7le ? %.7le * x : 1/0) lc 1 lw 2,\\\n" % (x[i],gbytes_emp[i][0]))

        plotfile.write("     (x >= %.7le ? %.7le : 1/0) lc 3 lw 2\n" % (x[0],gflops_emp[0]))

        plotfile.close()

        command = "echo 'load \"%s\"' | %s" % (loadname,self.dict["CONFIG"]["ERT_GNUPLOT"][0])
        if execute_shell(command,self.options.verbose > 1) != 0:
          sys.stderr.write("Unable to produce a '%s' for %s\n" % (basename,self.results_dir))
          return 1

      if self.options.verbose > 0:
        print()
        print ("+-------------------------------------------------")
        if self.options.gnuplot:
          print ("| Empirical roofline graph:    '%s/roofline.ps'"   % self.results_dir)
        print ("| Empirical roofline database: '%s/roofline.json'" % self.results_dir)
        print ("+-------------------------------------------------")
        print()

    return 0
