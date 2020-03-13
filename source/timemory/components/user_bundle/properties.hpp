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
 * \file timemory/components/user_bundle/properties.hpp
 * \brief Specialization of the properties for the user_bundle components
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/enum.h"

//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_global_bundle, USER_GLOBAL_BUNDLE,
                                 "user_global_bundle", "global_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_list_bundle, USER_LIST_BUNDLE, "user_list_bundle",
                                 "list_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_tuple_bundle, USER_TUPLE_BUNDLE,
                                 "user_tuple_bundle", "tuple_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_ompt_bundle, USER_OMPT_BUNDLE, "user_ompt_bundle",
                                 "ompt", "omp_tools", "openmp", "openmp_tools")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_mpip_bundle, USER_MPIP_BUNDLE,
                                 "user_mpip_bundle"
                                 "mpip",
                                 "mpi_tools", "mpi")
//
//======================================================================================//
