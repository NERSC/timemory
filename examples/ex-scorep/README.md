# ex-scorep

These examples demonstrate the usage of timemory for forwarding instrumentation for STL multi-threaded, OpenMP multi-threaded and hybrid MPI multi-threaded (both STL and OpenMP) applications to Score-P performance measurement infrastructure.

## Build

See [examples](../README.md##Build). These examples build the following corresponding binaries: `ex_scorep`, `ex_scorep.omp` and `ex_scorep.mpi`. The `ex_scorep.omp` requires `-DUSE_OPENMP=ON` to build whereas the `ex_scorep.mpi` requires `-DTIMEMORY_USE_MPI=ON`, `-DTIMEMORY_USE_MPI_INIT=ON`, `-DUSE_MPI=ON` and optionally `-DUSE_OPENMP=ON` to build.

## Expected Output

### ex_scorep

```bash
$ ./ex_scorep
#------------------------- tim::manager initialized [id=0][pid=20353] -------------------------#

Using STL threading 
0: Answer = 1981891
0: Answer = 1981891
1: Answer = 1981891
1: Answer = 1981891
2: Answer = 1981891
2: Answer = 1981891


#---------------------- tim::manager destroyed [rank=0][id=0][pid=20353] ----------------------#

$ ls scorep-*
build-test/scorep-20200730_0048_4221194970943529:
MANIFEST.md  profile.cubex  scorep.cfg
```

### ex_scorep.omp

```bash
$ ./ex_scorep.omp
#------------------------- tim::manager initialized [id=0][pid=20398] -------------------------#

Using OpenMP threading 
3: Answer = 1981891
0: Answer = 1981891
4: Answer = 1981891
1: Answer = 1981891
5: Answer = 1981891
2: Answer = 1981891
[Score-P] ../src/measurement/thread/create_wait/scorep_thread_create_wait_generic.c:637: Warning: Thread after main (location=1)


#---------------------- tim::manager destroyed [rank=0][id=0][pid=20398] ----------------------#

$ ls scorep-*
build-test/scorep-20200730_0048_4221194970943529:
MANIFEST.md  profile.cubex  scorep.cfg
```

### ex_scorep.mpi

```bash
$ mpirun -np 2 ./ex_scorep.mpi
#------------------------- tim::manager initialized [id=0][pid=20469] -------------------------#

#------------------------- tim::manager initialized [id=0][pid=20468] -------------------------#

[Score-P] ../src/adapters/mpi/SCOREP_Mpi_Env.c:230: Warning: MPI environment initialization request and provided level exceed MPI_THREAD_FUNNELED!
[Score-P] ../src/adapters/mpi/SCOREP_Mpi_Env.c:230: Warning: MPI environment initialization request and provided level exceed MPI_THREAD_FUNNELED!
0: Answer = 1981891
0: Answer = 1981891
0: Answer = 1981891
0: Answer = 1981891
1: Answer = 1981891
1: Answer = 1981891
1: Answer = 1981891
1: Answer = 1981891
2: Answer = 1981891
2: Answer = 1981891
2: Answer = 1981891
2: Answer = 1981891
0: Answer = 1981891
0: Answer = 1981891
1: Answer = 1981891
1: Answer = 1981891
2: Answer = 1981891
2: Answer = 1981891
3: Answer = 1981891
3: Answer = 1981891
4: Answer = 1981891
4: Answer = 1981891
5: Answer = 1981891
5: Answer = 1981891


#---------------------- tim::manager destroyed [rank=1][id=0][pid=20469] ----------------------#


#---------------------- tim::manager destroyed [rank=0][id=0][pid=20468] ----------------------#

$ ls scorep-*
build-test/scorep-20200730_0048_4221194970943529:
MANIFEST.md  profile.cubex  scorep.cfg
```