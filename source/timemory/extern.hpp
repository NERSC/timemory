// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

//
#if defined(TIMEMORY_USE_CORE_EXTERN)
// clang-format off
#    include "timemory/hash/extern.hpp"
#    include "timemory/environment/extern.hpp"
#    include "timemory/settings/extern.hpp"
#    include "timemory/plotting/extern.hpp"
// clang-format on
#endif
//
#if defined(TIMEMORY_USE_MANAGER_EXTERN)
#    include "timemory/manager/extern.hpp"
#endif
//
#if defined(TIMEMORY_USE_ERT_EXTERN)
#    include "timemory/ert/extern.hpp"
#endif
//
#if defined(TIMEMORY_USE_CONFIG_EXTERN)
#    include "timemory/config/extern.hpp"
#endif
//
#if defined(TIMEMORY_USE_STORAGE_EXTERN)
#    include "timemory/storage/extern.hpp"
#endif
//
#if defined(TIMEMORY_USE_OPERATIONS_EXTERN)
#    include "timemory/operations/extern.hpp"
#endif
//
#if defined(TIMEMORY_USE_BACKEND_EXTERN)
#    include "timemory/backends/extern.hpp"
#    include "timemory/backends/types/mpi/extern.hpp"
#endif
//
#if defined(TIMEMORY_USE_VARIADIC_EXTERN)
#endif
//
#if defined(TIMEMORY_USE_COMPONENT_EXTERN)
#    include "timemory/components/extern.hpp"
#endif
//
#if defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
#    include "timemory/components/user_bundle/extern.hpp"
#endif
//
#if defined(TIMEMORY_USE_CONTAINERS_EXTERN)
#    include "timemory/containers/extern.hpp"
#endif
//
#if defined(TIMEMORY_USE_RUNTIME_EXTERN)
#    include "timemory/runtime/extern.hpp"
#endif
//
