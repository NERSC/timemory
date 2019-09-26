# Operating System Dependent Components

These components are only available on certain operating systems. In general, all of these components are available for POSIX systems.
The components in the "resource usage" category are provided by POSIX `rusage` (`man getrusage`).

| Component Name                 | Category       | Dependencies | Description                                                                                                                                                                                    |
| ------------------------------ | -------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`read_bytes`**               | I/O            | Linux/macOS  | Attempt to count the number of bytes which this process really did cause to be fetched from the storage layer. Done at the submit_bio() level, so it is accurate for block-backed filesystems. |
| **`written_bytes`**            | I/O            | Linux/macOS  | Attempt to count the number of bytes which this process caused to be sent to the storage layer. This is done at page-dirtying time.                                                            |
| **`process_cpu_clock`**        | timing         | POSIX        | CPU timer that tracks the amount of CPU (in user- or kernel-mode) used by the calling process (excludes child processes)                                                                       |
| **`thread_cpu_clock`**         | timing         | POSIX        | CPU timer that tracks the amount of CPU (in user- or kernel-mode) used by the calling thread (excludes sibling/child threads)                                                                  |
| **`process_cpu_util`**         | timing         | POSIX        | Percentage of process CPU time (`process_cpu_clock`) vs. wall-clock time                                                                                                                       |
| **`thread_cpu_util`**          | timing         | POSIX        | Percentage of thread CPU time (`thread_cpu_clock`) vs. `wall_clock`                                                                                                                            |
| **`monotonic_clock`**          | timing         | POSIX        | Real-clock timer that increments monotonically, unaffected by frequency or time adjustments, that increments while system is asleep                                                            |
| **`monotonic_raw_clock`**      | timing         | POSIX        | Real-clock timer that increments monotonically, unaffected by frequency or time adjustments                                                                                                    |
| **`data_rss`**                 | resource usage | POSIX        | Unshared memory residing the data segment of a process                                                                                                                                         |
| **`stack_rss`**                | resource usage | POSIX        | Integral value of the amount of unshared memory residing in the stack segment of a process                                                                                                     |
| **`num_io_in`**                | resource usage | POSIX        | Number of times the file system had to perform input                                                                                                                                           |
| **`num_io_out`**               | resource usage | POSIX        | Number of times the file system had to perform output                                                                                                                                          |
| **`num_major_page_faults`**    | resource usage | POSIX        | Number of page faults serviced that required I/O activity                                                                                                                                      |
| **`num_minor_page_faults`**    | resource usage | POSIX        | Number of page faults serviced without any I/O activity<sup>[[1]](#fn1)</sup>                                                                                                                  |
| **`num_msg_recv`**             | resource usage | POSIX        | Number of IPC messages received                                                                                                                                                                |
| **`num_msg_sent`**             | resource usage | POSIX        | Number of IPC messages sent                                                                                                                                                                    |
| **`num_signals`**              | resource usage | POSIX        | Number of signals delivered                                                                                                                                                                    |
| **`num_swap`**                 | resource_usage | POSIX        | Number of swaps out of main memory                                                                                                                                                             |
| **`priority_context_switch`**  | resource usage | POSIX        | Number of times a context switch resulted due to a higher priority process becoming runnable or bc the current process exceeded its time slice.                                                |
| **`voluntary_context_switch`** | resource usage | POSIX        | Number of times a context switch resulted due to a process voluntarily giving up the processor before its time slice was completed<sup>[[2]](#fn2)</sup>                                       |

<a name="fn1">[1]</a>: Here I/O activity is avoided by reclaiming a page frame from the list of pages awaiting reallocation

<a name="fn2">[2]</a>: Usually to await availability of a resource
