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

/**
 * \file timemory/components/likwid/types.hpp
 * \brief Declare the likwid component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

//======================================================================================//
//
TIMEMORY_DECLARE_COMPONENT(likwid_marker)
TIMEMORY_DECLARE_COMPONENT(likwid_nvmarker)
//
//======================================================================================//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_LIKWID)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::likwid_marker, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::likwid_nvmarker, false_type)
#else
#    if !defined(TIMEMORY_USE_CUDA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::likwid_nvmarker, false_type)
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              REQUIRES PREFIX
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::likwid_marker, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(requires_prefix, component::likwid_nvmarker, true_type)
//
//--------------------------------------------------------------------------------------//
//
//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(likwid_marker, LIKWID_MARKER, "likwid_marker", "")
//
TIMEMORY_PROPERTY_SPECIALIZATION(likwid_nvmarker, LIKWID_NVMARKER, "likwid_nvmarker", "")
