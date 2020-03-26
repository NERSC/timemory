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

#include "timemory/backends/device.hpp"
#include "timemory/components/cuda/backends.hpp"
#include "timemory/components/timing/components.hpp"
#include "timemory/ert/configuration.hpp"
#include "timemory/ert/counter.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/operations/definition.hpp"
#include "timemory/plotting.hpp"

namespace tim
{
namespace ert
{
//
//
// template class exec_data<component::wall_clock>;
//
template class counter<device::cpu, float, component::wall_clock>;
template class counter<device::cpu, double, component::wall_clock>;
//
// template struct configuration<device::cpu, float, component::wall_clock>;
// template struct configuration<device::cpu, double, component::wall_clock>;
//
//
}  // namespace ert
}  // namespace tim