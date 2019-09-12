# Using GOTCHA

[GOTCHA](https://github.com/LLNL/GOTCHA) is a library that wraps functions and is used to place hook into external libraries.
It is similar to LD_PRELOAD, but operates via a programmable API.
This enables easy methods of accomplishing tasks like code instrumentation or wholesale replacement of mechanisms in programs without disrupting their source code.

The `gotcha` component in timemory supports implicit extraction of the wrapped function return type and arguments and
significantly reduces the complexity of a traditional GOTCHA specification.
Additionally, limited support for C++ function mangling required to intercept C++ function calls.

| Component Name                | Category                       | Dependences                              | Description                                           |
| ----------------------------- | ------------------------------ | ---------------------------------------- | ----------------------------------------------------- |
| **`gotcha<Size,Tools,Diff>`** | All (specify other components) | [GOTCHA](https://github.com/LLNL/GOTCHA) | Wrap external function calls with timemory components |

Requires at least two template parameters: `gotcha<Size, Tools, Diff = void>`
where `Size` is the maximum number of external functions to be wrapped,
`Tools` is a [variadic component wrapper](#variadic-component-wrappers), and
`Diff` is an optional template parameter for differentiating `gotcha` components with equivalent `Size` and `Tools`
parameters but wrap different functions. Note: the `Tools` type cannot contain other `gotcha` components.
