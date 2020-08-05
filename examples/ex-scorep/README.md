# ex-scorep

These examples demonstrate the usage of timemory to forward to Score-P instrumentation API.

## Build

See [examples](../README.md##Build). These examples build the following corresponding binaries: `ex_scorep`.

## Expected Output

```bash
$ ./ex_scorep
#------------------------- tim::manager initialized [id=0][pid=11794] -------------------------#

Answer = 267914296
Answer = 331160282
Answer = 433494437
Answer = 267914296
Answer = 331160282
Answer = 433494437
Answer = 267914296
Answer = 331160282
Answer = 433494437
Answer = 267914296


#---------------------- tim::manager destroyed [rank=0][id=0][pid=11794] ----------------------#

$ ls scorep-*
build-test/scorep-20200730_0048_4221194970943529:
MANIFEST.md  profile.cubex  scorep.cfg
```
