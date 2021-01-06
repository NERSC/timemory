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

#include "timemory/containers/declaration.hpp"
#include "timemory/containers/macros.hpp"
#include "timemory/containers/types.hpp"
#if !defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_CONTAINERS_EXTERN)
#    include "timemory/operations/definition.hpp"
#    include "timemory/storage/definition.hpp"
#else
#    include "timemory/operations/extern.hpp"
#    include "timemory/storage/extern.hpp"
#endif
#include "timemory/runtime/configure.hpp"

#include "timemory/containers/types/complete_list.hpp"
#include "timemory/containers/types/full_auto_timer.hpp"
#include "timemory/containers/types/minimal_auto_timer.hpp"
