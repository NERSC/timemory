# timem

Command-line tool which provides the same capabilities of the UNIX command-line tool `time` but extends it to several additional metrics:

- Memory usage
- Major/minor page faults
- Context switches
- Read/write bytes
- Read/write bytes per second
- Hardware counters (when PAPI is available)

## Options

### Command Line

#### Standard Executable

The `timem` and `timem-mpi` executation wrappers support simple prefixing of the executable and any arguments
are assumed to belong to the file being executed unless `"--"` (two dashes) surrounded by spaces is detected.
If the two stand-alone dashes are detected, these executables assume every argument preceding the
`"--"` is an argument to `timem`/`timem-mpi` and every argument following the two dashes is the command
to execute, e.g. `timem <timem-arguments> -- <exe> <exe-arguments>`.

```console
$ timem --help
Usage: ./timem [tim::argparse::argument_parser arguments...] -- <CMD> <ARGS>

Options:
    -h, -?, --help                 Shows this page
    --debug                        Debug output
    -v, --verbose                  Verbose output
    -q, --quiet                    Suppress as much reporting as possible
    -d, --sample-delay             Set the delay before the sampler starts (seconds)
    -f, --sample-freq              Set the frequency of the sampler (number of interrupts per second
    --disable-sample               Disable sampling completely
    -e, --events, --papi-events    Set the hardware counter events to record (ref: `timemory-avail -H | grep PAPI`)
    --disable-papi                 Disable hardware counters
    -o, --output                   Write intermediate data to an output process-specific file. Some metrics, such as those associated with timers, may report intermediate values (e.g. starting timestamps as number of "seconds")
    -s, --shell                    Enable launching command via a shell command (if no arguments, $SHELL is used)
    --shell-flags                  Set the shell flags to use (pass as single string as leading dashes can confuse parser) [default: -i]
```

#### MPI-Compatible Executable

```console
$ ./timem-mpi --help
Usage: ./timem-mpi [tim::argparse::argument_parser arguments...] -- <CMD> <ARGS>

Options:
    -h, -?, --help                 Shows this page
    --debug                        Debug output
    -v, --verbose                  Verbose output
    -q, --quiet                    Suppress as much reporting as possible
    -d, --sample-delay             Set the delay before the sampler starts (seconds)
    -f, --sample-freq              Set the frequency of the sampler (number of interrupts per second
    --disable-sample               Disable sampling completely
    -e, --events, --papi-events    Set the hardware counter events to record (ref: `timemory-avail -H | grep PAPI`)
    --disable-papi                 Disable hardware counters
    -o, --output                   Write intermediate data to an output process-specific file. Some metrics, such as those associated with timers, may report intermediate values (e.g. starting timestamps as number of "seconds")
    -s, --shell                    Enable launching command via a shell command (if no arguments, $SHELL is used)
    --shell-flags                  Set the shell flags to use (pass as single string as leading dashes can confuse parser) [default: -i]
    --mpi                          Launch processes via MPI_Comm_spawn_multiple (reduced functionality)
    --disable-mpi                  Disable MPI_Finalize
    -i, --indiv                    Output individual results for each process (i.e. rank) instead of reporting the aggregation
```

### Environment

- `TIMEM_USE_SHELL` : enable execution via `$SHELL`
    - default: `"OFF"`
    - e.g. `/bin/bash <CMD>`
- `TIMEM_USE_SHELL_OPTIONS` : shell invocation options
    - default: `-i`
    - e.g. `/bin/bash -i <CMD>`
- `TIMEM_SAMPLE_FREQ` : expressed in 1/seconds, that sets the frequency that the timem executable samples the relevant measurements
    - default: `2.0`
- `TIMEM_SAMPLE_DELAY` : expressed in seconds, that sets the length of time the timem executable waits before starting sampling of the relevant measurements
    - default: `0.001`
- `TIMEMORY_PAPI_EVENTS` : Hardware counters. Use `papi_avail` and `papi_native_avail`

## Customization Demonstration

The ability to customize the behavior of several components without altering the components themselves in demonstrated in
`timem.hpp`. In this tool, additional type-traits are added, the printing output of a bundle of components is customized,
and new capabilities are introduced:

- Introduction of `custom_print` and `timem_sample` operations
  - `custom_print` replaces comma delimiters with new line delimiters
  - `timem_sample` invokes a
- Specialization of the `trait::custom_label_printing` type-trait for `papi_array_t`
  - This type-trait communicates to timemory that data labels for the components should not be included when invoking `operator<<` on this component
- Specialization of the `sample`, `start`, and `stop` operations for `papi_array_t` component
  - The default behavior of `operation::sample<T>` pushes a measurement to an array in the `storage<T>`
  - However, in this tool, we modify the operation to just update the value of the current object
- Specialization of the `operation::base_printer<T>` for `read_bytes` and `written_bytes` components
  - The default behavior of `read_bytes` and `written_bytes` is to print the bytes read/written + the bytes read/written
  per second on the same line. This is customized to print on separate lines
- Introduction of `timem_tuple` -- an extension to `component_tuple` which:
  - Reimplements the `component_tuple<T...>::sample()` member function to include the `timem_sample` operation
  - Replaces the `operator<<` to utilize the `custom_print` operation
