# timemory-run

Dynamic instrumentation and binary re-writing command-line tool.

## General Syntax

```console
# run exe as a subprocess
timemory-run <OPTIONS> -- <EXECUTABLE> <ARGS>
# binary rewriting
timemory-run <OPTIONS> -o <OUTPUT_EXECUTABLE> -- <EXECUTABLE>

```

- `<OPTIONS>` : use `timemory-run --help`
- `<EXECUTABLE>`
  - Binary file which will be instrumented with timemory
    - Will not work with a script (e.g. bash script, Python script, etc.)
- `<ARGS>`
  - Command line arguments to executable

## Dynamic Instrumentation Mode

Dynamic instrumentation mode enables launching the `<EXECUTABLE>` as a subprocess.
