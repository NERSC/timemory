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

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"

#if !defined(TIMEMORY_COMPONENT_SOURCE) && !defined(TIMEMORY_USE_PERFETTO_EXTERN)
#    if !defined(TIMEMORY_COMPONENT_PERFETTO_HEADER_ONLY_MODE)
#        define TIMEMORY_COMPONENT_PERFETTO_HEADER_ONLY_MODE 1
#    endif
#endif

TIMEMORY_DECLARE_COMPONENT(perfetto_trace)

TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_value_storage, component::perfetto_trace, false_type)

#if !defined(TIMEMORY_USE_PERFETTO)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::perfetto_trace, false_type)
#endif

// perfetto header
#if defined(TIMEMORY_USE_PERFETTO)
#    include <perfetto.h>
#endif

// provides a constexpr "category"
#if !defined(TIMEMORY_PERFETTO_API)
#    define TIMEMORY_PERFETTO_API ::tim::project::timemory
#endif

// allow including file to override by defining before inclusion
#if !defined(TIMEMORY_PERFETTO_CATEGORIES)
#    define TIMEMORY_PERFETTO_CATEGORIES                                                 \
        perfetto::Category("timemory").SetDescription("Events from the timemory API")
#endif

#if defined(TIMEMORY_USE_PERFETTO)
PERFETTO_DEFINE_CATEGORIES(TIMEMORY_PERFETTO_CATEGORIES);
#endif

TIMEMORY_PROPERTY_SPECIALIZATION(perfetto_trace, TIMEMORY_PERFETTO_TRACE,
                                 "perfetto_trace", "perfetto")
