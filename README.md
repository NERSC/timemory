# TiMEmory
C++ and Python Timing + Memory Utilities including auto-timers and temporary memory calculation

Timing Results
==============

Overview
--------

There are essentially two components of the output:

- a text file (e.g. timing_report_XXX.out file)

  - general ASCII report

- a JSON file with more detailed data

  - used for plotting purposes
  - example provided in `examples/timing_plot.py`

- Implementation uses “auto-timers”. Essentially, at the beginning of a function, you create a timer. 
- The timer starts automatically and when the timer is “destroyed”, i.e. goes out of scope at the end of the function, it stops the timer and records the time difference and also some memory measurements. 
- The way the auto-timers are setup is that they will automatically record the name of the function they were created in.
- Additional info is sometimes added when you have similar function names, for example, a python “__init__” function will want to create an auto-timer that provides the class the function is being called from. 
- All this info will show up with an ensuing “@‘ tag on the end of the function name. Other options are the name of the file, etc.

Example
-------

For the interpretation of text output, here is an example and the explanation of it’s structure

::

    > rank 0
    > [pyc] main@'toast_ground_sim_simple.py'             : 17.650 wall, 21.920 user +  1.610 system = 23.530 CPU [seconds] (133.3%) [ total rss curr|peak = 655.8|944.1 MB ] [ self rss curr|peak = 582.7|871.0 MB ]
    > [pyc]   create_observations                         :  1.283 wall,  1.250 user +  0.030 system =  1.280 CPU [seconds] ( 99.7%) [ total rss curr|peak =  91.8|102.6 MB ] [ self rss curr|peak =  18.4| 29.2 MB ]
    > [pyc]     __init__@TODGround                        :  1.280 wall,  1.250 user +  0.030 system =  1.280 CPU [seconds] (100.0%) [ total rss curr|peak =  93.2|102.6 MB ] [ self rss curr|peak =   4.7|  5.4 MB ] (total # of laps: 2)
    > [pyc]       simulate_scan@TODGround                 :  0.035 wall,  0.040 user +  0.010 system =  0.050 CPU [seconds] (143.8%) [ total rss curr|peak =  88.5| 97.3 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [pyc]       translate_pointing@TODGround            :  1.239 wall,  1.210 user +  0.020 system =  1.230 CPU [seconds] ( 99.3%) [ total rss curr|peak = 102.6|102.6 MB ] [ self rss curr|peak =  14.0|  5.4 MB ] (total # of laps: 2)
    > [pyc]         rotate                                :  0.004 wall,  0.010 user +  0.000 system =  0.010 CPU [seconds] (252.6%) [ total rss curr|peak =  89.6| 97.3 MB ] [ self rss curr|peak =   0.9|  0.0 MB ] (total # of laps: 2)
    > [pyc]         radec2quat@TODGround                  :  0.014 wall,  0.010 user +  0.010 system =  0.020 CPU [seconds] (142.7%) [ total rss curr|peak =  97.1| 97.3 MB ] [ self rss curr|peak =   6.9|  0.0 MB ] (total # of laps: 2)
    > [pyc]           rotation                            :  0.009 wall,  0.000 user +  0.010 system =  0.010 CPU [seconds] (110.4%) [ total rss curr|peak =  94.4| 97.3 MB ] [ self rss curr|peak =   0.9|  0.0 MB ] (total # of laps: 6)
    > [cxx]             ctoast_qarray_from_axisangle      :  0.006 wall,  0.000 user +  0.010 system =  0.010 CPU [seconds] (160.5%) [ total rss curr|peak =  94.4| 97.3 MB ] [ self rss curr|peak =   0.9|  0.0 MB ] (total # of laps: 6)
    > [pyc]   expand_pointing                             :  0.918 wall,  1.110 user +  0.150 system =  1.260 CPU [seconds] (137.3%) [ total rss curr|peak = 374.2|374.9 MB ] [ self rss curr|peak = 282.3|272.2 MB ]
    > [pyc]     exec@OpPointingHpix                       :  0.917 wall,  1.110 user +  0.150 system =  1.260 CPU [seconds] (137.4%) [ total rss curr|peak = 374.9|374.9 MB ] [ self rss curr|peak = 283.0|272.2 MB ]
    > [pyc]       read_times@TODGround                    :  0.006 wall,  0.000 user +  0.010 system =  0.010 CPU [seconds] (170.0%) [ total rss curr|peak = 234.5|234.5 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [pyc]         _get_times@TODGround                  :  0.006 wall,  0.000 user +  0.010 system =  0.010 CPU [seconds] (173.7%) [ total rss curr|peak = 234.5|234.5 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [pyc]       read_pntg@TODGround                     :  0.109 wall,  0.100 user +  0.010 system =  0.110 CPU [seconds] (101.2%) [ total rss curr|peak = 372.6|372.6 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]         _get_pntg@TODGround                   :  0.101 wall,  0.100 user +  0.000 system =  0.100 CPU [seconds] ( 99.0%) [ total rss curr|peak = 372.6|372.6 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]           mult                                :  0.094 wall,  0.090 user +  0.000 system =  0.090 CPU [seconds] ( 95.9%) [ total rss curr|peak = 372.6|372.6 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [cxx]             ctoast_qarray_mult                :  0.035 wall,  0.070 user +  0.000 system =  0.070 CPU [seconds] (200.3%) [ total rss curr|peak = 372.6|372.6 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [cxx]       ctoast_pointing_healpix_matrix          :  0.716 wall,  0.940 user +  0.120 system =  1.060 CPU [seconds] (148.1%) [ total rss curr|peak = 374.9|374.9 MB ] [ self rss curr|peak =   2.3|  2.3 MB ] (total # of laps: 122)
    > [pyc]   get_submaps                                 :  0.092 wall,  0.090 user +  0.000 system =  0.090 CPU [seconds] ( 98.0%) [ total rss curr|peak = 374.4|374.9 MB ] [ self rss curr|peak =   0.2|  0.0 MB ]
    > [pyc]     exec@OpLocalPixels                        :  0.091 wall,  0.090 user +  0.000 system =  0.090 CPU [seconds] ( 98.4%) [ total rss curr|peak = 374.4|374.9 MB ] [ self rss curr|peak =   0.2|  0.0 MB ]
    > [pyc]   scan_signal                                 :  0.676 wall,  0.500 user +  0.110 system =  0.610 CPU [seconds] ( 90.3%) [ total rss curr|peak = 536.5|540.2 MB ] [ self rss curr|peak = 162.2|165.3 MB ]
    > [pyc]     read_healpix_fits@DistPixels              :  0.422 wall,  0.270 user +  0.090 system =  0.360 CPU [seconds] ( 85.4%) [ total rss curr|peak = 486.5|521.2 MB ] [ self rss curr|peak = 112.2|146.4 MB ]
    > [pyc]     exec@OpSimScan                            :  0.250 wall,  0.230 user +  0.020 system =  0.250 CPU [seconds] ( 99.8%) [ total rss curr|peak = 540.2|540.2 MB ] [ self rss curr|peak =  54.6| 18.9 MB ]
    > [cxx]       ctoast_sim_map_scan_map32               :  0.029 wall,  0.020 user +  0.000 system =  0.020 CPU [seconds] ( 69.6%) [ total rss curr|peak = 540.2|540.2 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]   build_npp                                   :  3.136 wall,  2.910 user +  0.320 system =  3.230 CPU [seconds] (103.0%) [ total rss curr|peak = 777.8|944.1 MB ] [ self rss curr|peak = 243.2|403.9 MB ]
    > [pyc]     exec@OpAccumDiag                          :  0.359 wall,  0.460 user +  0.020 system =  0.480 CPU [seconds] (133.7%) [ total rss curr|peak = 542.9|542.9 MB ] [ self rss curr|peak =   8.0|  2.7 MB ]
    > [cxx]       ctoast_cov_accumulate_diagonal_invnpp   :  0.139 wall,  0.260 user +  0.020 system =  0.280 CPU [seconds] (201.6%) [ total rss curr|peak = 542.9|542.9 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [cxx]         accumulate_diagonal_invnpp            :  0.134 wall,  0.260 user +  0.020 system =  0.280 CPU [seconds] (208.3%) [ total rss curr|peak = 542.9|542.9 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]     write_healpix_fits@DistPixels             :  2.649 wall,  2.370 user +  0.250 system =  2.620 CPU [seconds] ( 98.9%) [ total rss curr|peak = 796.2|944.1 MB ] [ self rss curr|peak =   2.7|  0.0 MB ] (total # of laps: 3)
    > [pyc]     covariance_invert@'map/noise.py'          :  0.011 wall,  0.010 user +  0.010 system =  0.020 CPU [seconds] (175.3%) [ total rss curr|peak = 780.1|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [cxx]       ctoast_cov_eigendecompose_diagonal      :  0.011 wall,  0.010 user +  0.010 system =  0.020 CPU [seconds] (182.7%) [ total rss curr|peak = 780.1|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [cxx]         eigendecompose_diagonal               :  0.011 wall,  0.010 user +  0.010 system =  0.020 CPU [seconds] (183.3%) [ total rss curr|peak = 780.1|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [pyc]   exec@OpCacheCopy                            :  0.042 wall,  0.030 user +  0.020 system =  0.050 CPU [seconds] (118.4%) [ total rss curr|peak = 831.6|944.1 MB ] [ self rss curr|peak =  56.2|  0.0 MB ]
    > [pyc]   bin_maps                                    :  1.915 wall,  1.870 user +  0.180 system =  2.050 CPU [seconds] (107.1%) [ total rss curr|peak = 756.4|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [pyc]     exec@OpAccumDiag                          :  0.606 wall,  0.720 user +  0.030 system =  0.750 CPU [seconds] (123.8%) [ total rss curr|peak = 675.6|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [cxx]       ctoast_cov_accumulate_zmap              :  0.187 wall,  0.320 user +  0.010 system =  0.330 CPU [seconds] (176.2%) [ total rss curr|peak = 760.6|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 244)
    > [cxx]         accumulate_zmap                       :  0.183 wall,  0.320 user +  0.010 system =  0.330 CPU [seconds] (180.6%) [ total rss curr|peak = 760.6|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 244)
    > [pyc]     write_healpix_fits@DistPixels             :  1.240 wall,  1.100 user +  0.140 system =  1.240 CPU [seconds] (100.0%) [ total rss curr|peak = 758.7|944.1 MB ] [ self rss curr|peak = 174.2|  0.0 MB ] (total # of laps: 2)
    > [pyc]   apply_polyfilter                            :  0.367 wall,  0.570 user +  0.060 system =  0.630 CPU [seconds] (171.5%) [ total rss curr|peak = 756.3|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [pyc]     exec@OpPolyFilter                         :  0.363 wall,  0.570 user +  0.050 system =  0.620 CPU [seconds] (170.9%) [ total rss curr|peak = 756.3|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [cxx]       ctoast_filter_polyfilter                :  0.277 wall,  0.470 user +  0.050 system =  0.520 CPU [seconds] (187.5%) [ total rss curr|peak = 756.3|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]   exec@OpCacheClear                           :  0.017 wall,  0.000 user +  0.020 system =  0.020 CPU [seconds] (119.6%) [ total rss curr|peak = 441.8|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [pyc]   apply_madam                                 :  9.183 wall, 13.570 user +  0.720 system = 14.290 CPU [seconds] (155.6%) [ total rss curr|peak = 655.8|944.1 MB ] [ self rss curr|peak = 214.0|  0.0 MB ]
    > [pyc]     exec@OpMadam                              :  9.161 wall, 13.560 user +  0.710 system = 14.270 CPU [seconds] (155.8%) [ total rss curr|peak = 655.8|944.1 MB ] [ self rss curr|peak = 214.0|  0.0 MB ]
    > [pyc]   __del__@TODGround                           :  1.709 wall,  1.650 user +  0.050 system =  1.700 CPU [seconds] ( 99.5%) [ total rss curr|peak = 234.4|944.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > rank 1
    > [pyc] main@'toast_ground_sim_simple.py'             : 17.650 wall, 21.610 user +  1.910 system = 23.520 CPU [seconds] (133.3%) [ total rss curr|peak = 647.6|788.1 MB ] [ self rss curr|peak = 574.4|714.8 MB ]
    > [pyc]   load_schedule                               :  0.001 wall,  0.010 user +  0.000 system =  0.010 CPU [seconds] (697.4%) [ total rss curr|peak =  73.3| 73.3 MB ] [ self rss curr|peak =   0.1|  0.1 MB ]
    > [pyc]   create_observations                         :  1.283 wall,  1.250 user +  0.030 system =  1.280 CPU [seconds] ( 99.8%) [ total rss curr|peak =  91.5|102.4 MB ] [ self rss curr|peak =  18.1| 29.0 MB ]
    > [pyc]     __init__@TODGround                        :  1.280 wall,  1.250 user +  0.030 system =  1.280 CPU [seconds] (100.0%) [ total rss curr|peak =  93.0|102.4 MB ] [ self rss curr|peak =   5.0|  5.7 MB ] (total # of laps: 2)
    > [pyc]       simulate_scan@TODGround                 :  0.036 wall,  0.030 user +  0.000 system =  0.030 CPU [seconds] ( 84.4%) [ total rss curr|peak =  88.0| 96.8 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [pyc]       translate_pointing@TODGround            :  1.239 wall,  1.220 user +  0.030 system =  1.250 CPU [seconds] (100.9%) [ total rss curr|peak = 102.4|102.4 MB ] [ self rss curr|peak =  14.3|  5.7 MB ] (total # of laps: 2)
    > [pyc]         from_angles                           :  0.003 wall,  0.010 user +  0.000 system =  0.010 CPU [seconds] (289.8%) [ total rss curr|peak =  88.2| 96.8 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [cxx]           ctoast_qarray_from_angles           :  0.003 wall,  0.010 user +  0.000 system =  0.010 CPU [seconds] (316.4%) [ total rss curr|peak =  88.2| 96.8 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [pyc]         radec2quat@TODGround                  :  0.014 wall,  0.010 user +  0.010 system =  0.020 CPU [seconds] (146.2%) [ total rss curr|peak =  96.9| 96.9 MB ] [ self rss curr|peak =   6.9|  0.2 MB ] (total # of laps: 2)
    > [pyc]           rotation                            :  0.009 wall,  0.010 user +  0.010 system =  0.020 CPU [seconds] (222.1%) [ total rss curr|peak =  94.2| 96.8 MB ] [ self rss curr|peak =   0.9|  0.0 MB ] (total # of laps: 6)
    > [cxx]             ctoast_qarray_from_axisangle      :  0.006 wall,  0.010 user +  0.000 system =  0.010 CPU [seconds] (164.5%) [ total rss curr|peak =  94.2| 96.8 MB ] [ self rss curr|peak =   0.9|  0.0 MB ] (total # of laps: 6)
    > [pyc]   expand_pointing                             :  0.918 wall,  1.110 user +  0.150 system =  1.260 CPU [seconds] (137.3%) [ total rss curr|peak = 373.8|374.5 MB ] [ self rss curr|peak = 282.3|272.1 MB ]
    > [pyc]     exec@OpPointingHpix                       :  0.916 wall,  1.100 user +  0.140 system =  1.240 CPU [seconds] (135.4%) [ total rss curr|peak = 374.5|374.5 MB ] [ self rss curr|peak = 283.0|272.1 MB ]
    > [pyc]       read_pntg@TODGround                     :  0.108 wall,  0.090 user +  0.020 system =  0.110 CPU [seconds] (101.5%) [ total rss curr|peak = 372.2|372.2 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]         _get_pntg@TODGround                   :  0.101 wall,  0.080 user +  0.020 system =  0.100 CPU [seconds] ( 99.3%) [ total rss curr|peak = 372.2|372.2 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]           mult                                :  0.094 wall,  0.080 user +  0.020 system =  0.100 CPU [seconds] (106.6%) [ total rss curr|peak = 372.2|372.2 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [cxx]             ctoast_qarray_mult                :  0.035 wall,  0.030 user +  0.020 system =  0.050 CPU [seconds] (142.6%) [ total rss curr|peak = 372.2|372.2 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [cxx]       ctoast_pointing_healpix_matrix          :  0.718 wall,  0.930 user +  0.120 system =  1.050 CPU [seconds] (146.3%) [ total rss curr|peak = 374.5|374.5 MB ] [ self rss curr|peak =   2.3|  2.3 MB ] (total # of laps: 122)
    > [pyc]   get_submaps                                 :  0.092 wall,  0.080 user +  0.000 system =  0.080 CPU [seconds] ( 87.2%) [ total rss curr|peak = 374.0|374.5 MB ] [ self rss curr|peak =   0.2|  0.0 MB ]
    > [pyc]     exec@OpLocalPixels                        :  0.090 wall,  0.080 user +  0.000 system =  0.080 CPU [seconds] ( 88.6%) [ total rss curr|peak = 374.0|374.5 MB ] [ self rss curr|peak =   0.2|  0.0 MB ]
    > [pyc]   scan_signal                                 :  0.672 wall,  0.550 user +  0.100 system =  0.650 CPU [seconds] ( 96.8%) [ total rss curr|peak = 435.5|435.5 MB ] [ self rss curr|peak =  61.5| 61.0 MB ]
    > [pyc]     read_healpix_fits@DistPixels              :  0.422 wall,  0.320 user +  0.080 system =  0.400 CPU [seconds] ( 94.9%) [ total rss curr|peak = 379.0|379.0 MB ] [ self rss curr|peak =   5.0|  4.5 MB ]
    > [pyc]     exec@OpSimScan                            :  0.249 wall,  0.230 user +  0.020 system =  0.250 CPU [seconds] (100.2%) [ total rss curr|peak = 435.5|435.5 MB ] [ self rss curr|peak =  56.5| 56.5 MB ]
    > [cxx]       ctoast_sim_map_scan_map32               :  0.028 wall,  0.020 user +  0.000 system =  0.020 CPU [seconds] ( 70.4%) [ total rss curr|peak = 435.5|435.5 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]   build_npp                                   :  3.141 wall,  2.590 user +  0.610 system =  3.200 CPU [seconds] (101.9%) [ total rss curr|peak = 466.6|467.8 MB ] [ self rss curr|peak =  31.3| 32.2 MB ]
    > [pyc]     exec@OpAccumDiag                          :  0.359 wall,  0.460 user +  0.020 system =  0.480 CPU [seconds] (133.7%) [ total rss curr|peak = 442.6|442.6 MB ] [ self rss curr|peak =   6.8|  6.8 MB ]
    > [cxx]       ctoast_cov_accumulate_diagonal_invnpp   :  0.139 wall,  0.260 user +  0.010 system =  0.270 CPU [seconds] (194.2%) [ total rss curr|peak = 442.6|442.6 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [cxx]         accumulate_diagonal_invnpp            :  0.134 wall,  0.260 user +  0.010 system =  0.270 CPU [seconds] (200.8%) [ total rss curr|peak = 442.6|442.6 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]     write_healpix_fits@DistPixels             :  0.105 wall,  0.080 user +  0.020 system =  0.100 CPU [seconds] ( 94.8%) [ total rss curr|peak = 466.6|467.8 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 3)
    > [pyc]     covariance_invert@'map/noise.py'          :  0.013 wall,  0.010 user +  0.000 system =  0.010 CPU [seconds] ( 77.9%) [ total rss curr|peak = 466.6|467.8 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [cxx]       ctoast_cov_eigendecompose_diagonal      :  0.012 wall,  0.010 user +  0.000 system =  0.010 CPU [seconds] ( 80.9%) [ total rss curr|peak = 466.6|467.8 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [cxx]         eigendecompose_diagonal               :  0.012 wall,  0.010 user +  0.000 system =  0.010 CPU [seconds] ( 81.1%) [ total rss curr|peak = 466.6|467.8 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [pyc]   exec@OpCacheCopy                            :  0.041 wall,  0.020 user +  0.010 system =  0.030 CPU [seconds] ( 72.8%) [ total rss curr|peak = 517.7|517.7 MB ] [ self rss curr|peak =  52.9| 50.0 MB ]
    > [pyc]   bin_maps                                    :  1.917 wall,  1.760 user +  0.300 system =  2.060 CPU [seconds] (107.5%) [ total rss curr|peak = 516.8|521.7 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [pyc]     exec@OpAccumDiag                          :  0.591 wall,  0.740 user +  0.020 system =  0.760 CPU [seconds] (128.6%) [ total rss curr|peak = 521.7|521.7 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [cxx]       ctoast_cov_accumulate_zmap              :  0.185 wall,  0.360 user +  0.000 system =  0.360 CPU [seconds] (194.7%) [ total rss curr|peak = 521.7|521.7 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 244)
    > [cxx]         accumulate_zmap                       :  0.180 wall,  0.360 user +  0.000 system =  0.360 CPU [seconds] (199.6%) [ total rss curr|peak = 521.7|521.7 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 244)
    > [pyc]     write_healpix_fits@DistPixels             :  0.054 wall,  0.050 user +  0.020 system =  0.070 CPU [seconds] (130.3%) [ total rss curr|peak = 516.8|521.7 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)
    > [pyc]   apply_polyfilter                            :  0.367 wall,  0.560 user +  0.060 system =  0.620 CPU [seconds] (168.7%) [ total rss curr|peak = 512.4|521.7 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [pyc]     exec@OpPolyFilter                         :  0.367 wall,  0.560 user +  0.060 system =  0.620 CPU [seconds] (168.8%) [ total rss curr|peak = 512.5|521.7 MB ] [ self rss curr|peak =   0.0|  0.0 MB ]
    > [cxx]       ctoast_filter_polyfilter                :  0.280 wall,  0.470 user +  0.060 system =  0.530 CPU [seconds] (189.4%) [ total rss curr|peak = 512.5|521.7 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 122)
    > [pyc]   apply_madam                                 :  9.193 wall, 13.670 user +  0.650 system = 14.320 CPU [seconds] (155.8%) [ total rss curr|peak = 647.6|788.1 MB ] [ self rss curr|peak = 214.5|266.4 MB ]
    > [pyc]     exec@OpMadam                              :  9.193 wall, 13.670 user +  0.650 system = 14.320 CPU [seconds] (155.8%) [ total rss curr|peak = 647.6|788.1 MB ] [ self rss curr|peak = 214.5|266.4 MB ]
    > [pyc]   __del__@TODGround                           :  1.582 wall,  1.510 user +  0.060 system =  1.570 CPU [seconds] ( 99.2%) [ total rss curr|peak = 226.2|788.1 MB ] [ self rss curr|peak =   0.0|  0.0 MB ] (total # of laps: 2)


GENERAL LAYOUT
--------------

- The "rank" line(s) give the MPI process/rank
- The first (non ">") column tells whether the “auto-timer” originated from C++ ([cxx]) or Python ([pyc]) code
- The second column is the function name the auto-timer was created in

  - The indentation signifies the call tree

- The last column referring to “laps” is the number of times the function was invoked

  - If the number of laps are not noted, the total number of laps is implicitly one

TIMING FIELDS
-------------

- Then you have 5 time measurements

  (1) Wall clock time (e.g. how long it took according to a clock “on the wall”)
  (2) User time (the time spent executing the code)
  (3) System time (thread-specific CPU time, e.g. an idle thread waiting for synchronization, etc.)
  (4) CPU time (user + system time)
  (5) Percent CPU utilization (cpu / wall * 100)
  
- For perfect speedup on 4 threads, the CPU time would be 4x as long as the wall clock time and would have a % CPU utilization of 400%

  - This also includes vectorization. If each thread ran a calculation that calculated 4 values with a single CPU instruction (SIMD), we would have a speed up of 16x (4 threads x 4 values at one time == 16x) 

- Relative time (i.e. self-cost) for a function at a certain indent level (i.e. indented with 2*level spaces from [pyc]/[cxx]) can be calculated from the function(s) at level+1 until you reach another function at the same level
- This is better understood by an example

  - function A is the main (it is “level 0”) and takes 35 seconds
  - function B is called from main (it is "level 1”)
  - function C is called from main (it is “level 1”)
  - function B does some calculations and calls function D (it is “level 2”) five times (e.g. a loop calling function D)
  - function B takes 20 seconds
  - function D, called from B, takes a total of 10 seconds (which is what is reported). The average time of function D is thus 2 seconds (10 sec / 5 laps)
  - function C does some calculations and also calls function D (again “level 2”) five times 
  - The call to function D from function C will be reported as separate from the calls to D from B thanks to a hashing technique we use to identify function calls originating from different call trees/sequences
  - function C takes 9 seconds 
  - function D, called from C, takes a total of 8 seconds (avg. of 1.6 seconds)
  - Thus we know that function B required 10 seconds of compute time by subtracting out the time spent in its calls to function D
  - We know that function C required 1 second of compute time by subtracting out the time spent in it’s calls to function D
  - We can subtract the time from function B and C to calculate the “self-cost” in function A (35 - 20 - 9 = 6 seconds)
  
    - When calculating the self-cost of A, one does not subtract the time spent in function D. These times are included in the timing of both B and C


MEMORY FIELDS
-------------

- The memory measurements are a bit confusing, admittedly. The two types “curr” ("current", which I will refer to as such from here on out) and “peak” have to do with different memory measurements

  - They are both “RSS” measurements, which stand for “resident set size”. This is the amount of physical memory in RAM that is currently private to the process
  
    - It does not include the “swap” memory, which is when the OS puts memory not currently being used onto the hard drive
    - Typical Linux implementations will start using swap when ~60% of your RAM is full (you can override this easily in Linux by switching the “swapiness” to say, 90% for better performance since swap is slower than RAM)

- All memory measurements with “laps” > 0, are the max memory measurement of each "lap"

  - The “current” and “peak” max measurements are computed independently
  - E.g. the “current” max doesn’t directly correspond to the “peak” max — one “lap” may record the largest “current” RSS measurement but that does not (necessarily) mean that the same “lap” is responsible for the max “peak” RSS measurement
  - This is due to our belief that the max values are the ones of interest — the instances we must guard against to avoid running out of memory

- With respect to “total” vs. “self”, this is fairly straightforward

  - For the “total”, I simply take a measurement of the memory usage at the creation of the timer
  - The “self” measurement is the difference in the memory measurements between the creation of the auto-timer and when it is destroyed
  
    - This measurement shows is how much persistent memory was created in the function
    - It is valuable primarily as a metric to see how much memory is being created in the function and returned to the calling function
    - For example, if function X called function Y and function Y allocated 10 MB of memory and returned an object using this memory to function X, you would see function Y have a “self-cost” of 10 MB in memory

- The difference between “current” and “peak” is how the memory is measured

  - The “peak” value is what the OS reports as the max amount of memory being used is
  - I find this to be slightly more informative than “current” which is measurement of the “pages” allocated in memory
  - The reason "current" is included is because of the following:
  
    - Essentially, a “page” of memory can be thought of as street addresses separated into “blocks”, i.e. 1242 MLK Blvd. is in the 1200 block of MLK Blvd. 
    - A “page” is thus similar to a “block” — it is a starting memory address
    - The size of the pages is defined by the OS and just like the “swappiness”, it can be modified
    - For example, the default page size may be 1 KB and when a process has memory allocation need for 5.5 KB, the OS will provide 6 “pages” 
    
      - This is why one will see performance improvements when dealing with certain applications that application require large contiguous memory blocks, larger “pages” require fewer page requests and fewer reallocations to different pages when more memory is requested for an existing object with contiguous memory)
      
    - Within the page itself, the entire page might be used or it might not be fully used
    - When a page is not entirely used, you will get a “current” RSS usage greater than the “peak” memory usage — the memory is reserved for the process but is not actually used so it is thus not contained in the “peak” RSS usage number
    - However, when several pages is requested and allocated within a function but then released when returning to the calling function (i.e. temporary/transient page usage), you will have a “peak” RSS exceeding the “current” RSS memory usage since the “current” is measured after the pages are released back to the OS
    - Thus, with these two numbers, one can then deduce how much temporary/transient memory usage is being allocated in the function — if a function reports a self-cost of 243.2 MB of “current” RSS and a “peak” RSS of 403.9 MB, then you know that the “build_npp” function created 243.2 MB of persistent memory but creating the object requiring the persistent 243.2 MB required an additional 160.7 MB of temporary/transient memory (403.9 MB - 243.2 MB).


USING AUTO-TIMERS
-----------------

If you have new Python code you would like to use the auto-timers with, here is general guide:

- Import the timing module (obvious, I guess)
- Always add the auto-timer at the very beginning of the function. 

  - You can use an variable name you wish but make sure it is a named variable (e.g. "autotimer = timemory.auto_timer()", not "timemory.auto_timer()”)
  - The auto-timer functionality requires the variable to exist for the scope of the function
  
- For free-standing function without any name conflicts, just add: “autotimer = timemory.auto_timer()”
- For functions within a class, add: “autotimer = timemory.auto_timer(type(self).__name__)”
- For the primary auto-timer, use: “autotimer = timemory.auto_timer(timemory.FILE())” — this will tag “main” with the python file name
- In some instances, you may want to include the directory of the filename, for this use: “autotimer = timemory.auto_timer(timemory.FILE(use_dirname = True))”
- Add “tman = timemory.timing_manager() ; tman.report()” at the end of your main file. 
  
  - It is generally recommended to do this in a different scope than the primary autotimer but not necessary. 
  - In other words, put all your work in a “main()” function looking like this:

  
