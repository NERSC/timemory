# Timemory Components

## Overview

### `timemory/components/base.hpp`

Defines the static polymorphic base class for all timemory components.

### `timemory/components/placeholder.hpp`

Unused

### `timemory/components/skeletons.hpp`

Unused

### `timemory/components/properties.hpp`

Defines the generic base class for metadata properties for a component.
Generate a component specific overload with `TIMEMORY_PROPERTIES_SPECIALIZATION` macro.

### `timemory/components/types.hpp`

This file is used to declare component types without providing the definition.
New components should not be directly declared here; instead, new components
should use the component-generator in `cmake/Scripts/component-generator` which
will create a standard layout folder for the component:

```console
./generator.sh mycomponent
mv mycomponent ../../../source/timemory/components/
```

and the resulting `types.hpp` file in the new folder should be included:

```cpp
#include "timemory/components/mycomponent/types.hpp"
```

## Component Folder Layout

The components provided by timemory use a standard folder layout for various reasons:

- Minimize unnecessary header file inclusion
- Avoid inadvertant template instantiations by not providing definition until necessary
- Easy look-up and reference for component names and type-traits

Until the actual component definition is needed (e.g. when instantiating a variadic bundle of tools),
only the declarations of the component types and their type-traits are required.
The declaration of the components and the type-traits are fully specified via
`#include "timemory/components/<CATEGORY>/types.hpp`. This file always includes the corresponding
`traits.hpp` and `properties.hpp`.

- `mycomponent/`
  - `types.hpp` : declaration of all component(s)
  - `traits.hpp` : overloading of all type-traits for the component(s)
  - `components.hpp` : definition of component(s)
  - `properties.hpp` : properties specialization (for `timemory-avail`)
  - `backends.hpp` : routines for handling functions used by component(s)
  - `extern.hpp` : common header that includes all the headers in the extern folder
  - `extern/`
    - The extern folder is used for extern template declarations and instantiations which, when utilized,
      will reduce the memory usage during compilation
    - `base.hpp` : provides extern template declarations or extern template instantiation for the templated base class
    - `base.cpp` : defines `TIMEMORY_SOURCE` which converts `TIMEMORY_EXTERN_TEMPLATE(...)` in `base.hpp` to an instantiation
    - `storage.hpp` : provides extern template declarations or extern template instantiation for the templated `tim::storage` class
    - `storage.cpp` : defines `TIMEMORY_SOURCE` which converts `TIMEMORY_EXTERN_STORAGE(...)` in `storage.hpp` to an instantiation
    - `operations.hpp` : provides extern template declarations or extern template instantiation for the templated `tim::operation` classes
    - `operations.cpp` : defines `TIMEMORY_SOURCE` which converts `TIMEMORY_EXTERN_OPERATIONS(...)` in `operations.hpp` to an instantiation
