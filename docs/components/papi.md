# PAPI Components

> Namespace: `tim::component`

| Component Name   | Category | Template Specification      | Dependencies | Description                                                                             |
| ---------------- | -------- | --------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| **`papi_tuple`** | CPU      | `papi_tuple<EventTypes...>` | PAPI         | Variadic list of compile-time specified list of PAPI preset types (e.g. `PAPI_TOT_CYC`) |
| **`papi_array`** | CPU      | `papi_array<N>`             | PAPI         | Variable set of PAPI counters up to size _N_. Supports native hardware counter types    |

## Pre-defined Types

- `papi_array_t`
  - Alias to `papi_array<32>`.

