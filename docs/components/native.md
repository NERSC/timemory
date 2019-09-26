# Native Components

These components are available on all operating systems (Windows, macOS, Linux).

| Component Name      | Category | Dependencies | Description                                                                                                  |
| ------------------- | -------- | ------------ | ------------------------------------------------------------------------------------------------------------ |
| **`real_clock`**    | timing   | Native       | Timer for the system's real time (i.e. wall time) clock                                                      |
| **`user_clock`**    | timing   | Native       | records the CPU time spent in user-mode                                                                      |
| **`system_clock`**  | timing   | Native       | records only the CPU time spent in kernel-mode                                                               |
| **`cpu_clock`**     | timing   | Native       | Timer reporting the number of CPU clock cycles / number of cycles per second                                 |
| **`cpu_util`**      | timing   | Native       | Percentage of CPU time vs. wall-clock time                                                                   |
| **`wall_clock`**    | timing   | Native       | Alias to `real_clock` for convenience                                                                        |
| **`virtual_clock`** | timing   | Native       | Alias to `real_clock` since time is a construct of our consciousness                                         |
| **`page_rss`**      | memory   | Native       | The total size of the pages of memory allocated excluding swap                                               |
| **`peak_rss`**      | memory   | Native       | The peak amount of utilized memory (resident-set size) at that point of execution ("high-water" memory mark) |
| **`trip_count`**    | counting | Native       | Recording the number of trips through a section of code                                                      |
