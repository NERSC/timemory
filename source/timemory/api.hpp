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

/** \file timemory/api.hpp
 * \headerfile timemory/api.hpp "timemory/api.hpp"
 *
 * This is a declaration of API types
 *
 */

#include "timemory/defines.h"
#include "timemory/version.h"

#include <type_traits>

//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_DECLARE_API
 * \brief Declare an API category
 */

#if !defined(TIMEMORY_DECLARE_API)
#    define TIMEMORY_DECLARE_API(NAME)                                                   \
        namespace tim                                                                    \
        {                                                                                \
        namespace api                                                                    \
        {                                                                                \
        struct NAME;                                                                     \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_DEFINE_API
 * \brief Define an API category
 */

#if !defined(TIMEMORY_DEFINE_API)
#    define TIMEMORY_DEFINE_API(NAME)                                                    \
        namespace tim                                                                    \
        {                                                                                \
        namespace api                                                                    \
        {                                                                                \
        struct NAME                                                                      \
        {};                                                                              \
        }                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <>                                                                      \
        struct is_api<api::NAME> : true_type                                             \
        {};                                                                              \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
//
using true_type  = std::true_type;
using false_type = std::false_type;
//
namespace trait
{
template <typename T>
struct is_api : false_type
{};
}  // namespace trait
//
}  // namespace tim

//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DECLARE_API(native_tag)
TIMEMORY_DEFINE_API(native_tag)

TIMEMORY_DECLARE_API(python)
TIMEMORY_DEFINE_API(python)
//
//--------------------------------------------------------------------------------------//
//
namespace cereal
{
class JSONInputArchive;
class XMLInputArchive;
class XMLOutputArchive;
}  // namespace cereal

//
//--------------------------------------------------------------------------------------//
//
//                              Default pre-processor settings
//
//--------------------------------------------------------------------------------------//
//

#if !defined(TIMEMORY_DEFAULT_API)
#    define TIMEMORY_DEFAULT_API ::tim::api::native_tag
#endif

#if !defined(TIMEMORY_API)
#    define TIMEMORY_API TIMEMORY_DEFAULT_API
#endif

#if defined(DISABLE_TIMEMORY) || defined(TIMEMORY_DISABLED)
#    if !defined(TIMEMORY_DEFAULT_AVAILABLE)
#        define TIMEMORY_DEFAULT_AVAILABLE false_type
#    endif
#else
#    if !defined(TIMEMORY_DEFAULT_AVAILABLE)
#        define TIMEMORY_DEFAULT_AVAILABLE true_type
#    endif
#endif

#if !defined(TIMEMORY_DEFAULT_STATISTICS_TYPE)
#    if defined(TIMEMORY_USE_STATISTICS)
#        define TIMEMORY_DEFAULT_STATISTICS_TYPE true_type
#    else
#        define TIMEMORY_DEFAULT_STATISTICS_TYPE false_type
#    endif
#endif

#if !defined(TIMEMORY_DEFAULT_PLOTTING)
#    if defined(TIMEMORY_USE_PLOTTING)
#        define TIMEMORY_DEFAULT_PLOTTING true
#    else
#        define TIMEMORY_DEFAULT_PLOTTING false
#    endif
#endif

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#    define TIMEMORY_DEFAULT_ENABLED true
#endif

#if !defined(TIMEMORY_PYTHON_PLOTTER)
#    define TIMEMORY_PYTHON_PLOTTER "python"
#endif

#if !defined(TIMEMORY_USE_XML_ARCHIVE)
//
#    if !defined(TIMEMORY_DEFAULT_INPUT_ARCHIVE)
#        define TIMEMORY_DEFAULT_INPUT_ARCHIVE cereal::JSONInputArchive
#    endif
//
#    if !defined(TIMEMORY_DEFAULT_OUTPUT_ARCHIVE)
#        define TIMEMORY_DEFAULT_OUTPUT_ARCHIVE ::tim::type_list<>
#    endif
//
#else
//
#    if !defined(TIMEMORY_DEFAULT_INPUT_ARCHIVE)
#        define TIMEMORY_DEFAULT_INPUT_ARCHIVE cereal::XMLInputArchive
#    endif
//
#    if !defined(TIMEMORY_DEFAULT_OUTPUT_ARCHIVE)
#        define TIMEMORY_DEFAULT_OUTPUT_ARCHIVE cereal::XMLOutputArchive
#    endif
//
#endif

#if !defined(TIMEMORY_INPUT_ARCHIVE)
#    define TIMEMORY_INPUT_ARCHIVE TIMEMORY_DEFAULT_INPUT_ARCHIVE
#endif

#if !defined(TIMEMORY_OUTPUT_ARCHIVE)
#    define TIMEMORY_OUTPUT_ARCHIVE TIMEMORY_DEFAULT_OUTPUT_ARCHIVE
#endif
