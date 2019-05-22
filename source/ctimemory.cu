//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <numeric>
#include <string>

#include "timemory/clocks.hpp"
#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/papi.hpp"
#include "timemory/rusage.hpp"
#include "timemory/serializer.hpp"
#include "timemory/singleton.hpp"
#include "timemory/storage.hpp"
#include "timemory/units.hpp"
#include "timemory/utility.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/cupti.hpp"
#endif

//======================================================================================//

namespace tim
{
namespace cupti
{
//--------------------------------------------------------------------------------------//

void initialize()
{
#if defined(TIMEMORY_USE_CUPTI)
    int ndevices = 0;
    cudaGetDeviceCount(&ndevices);
    for(int i = 0; i < ndevices; ++i)
    {
        DRIVER_API_CALL(cuInit(i));
    }
#endif
}

//--------------------------------------------------------------------------------------//
}  // namespace cupti
}  // namespace tim

//======================================================================================//
