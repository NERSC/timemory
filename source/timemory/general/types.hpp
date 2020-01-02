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

/** \file general/types.hpp
 * \headerfile general/types.hpp "timemory/general/types.hpp"
 * Provides some additional info for timemory/components/types.hpp
 *
 */

#pragma once

#include "timemory/enum.h"
#include "timemory/mpl/apply.hpp"
#include "timemory/settings.hpp"

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tim
{
//======================================================================================//
// generate a master instance and a nullptr on the first pass
// generate a worker instance on subsequent and return master and worker
//
template <typename _Tp, typename _Ptr = std::shared_ptr<_Tp>,
          typename _Pair = std::pair<_Ptr, _Ptr>>
_Pair&
get_shared_ptr_pair()
{
    static auto              _master = std::make_shared<_Tp>();
    static std::atomic<int>  _counter(0);
    static thread_local auto _worker   = _Ptr((_counter++ == 0) ? nullptr : new _Tp());
    static thread_local auto _instance = _Pair{ _master, _worker };
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Ptr = std::shared_ptr<_Tp>,
          typename _Pair = std::pair<_Ptr, _Ptr>>
_Ptr
get_shared_ptr_pair_instance()
{
    static thread_local auto& _pinst = get_shared_ptr_pair<_Tp>();
    static thread_local auto& _inst  = _pinst.second.get() ? _pinst.second : _pinst.first;
    return _inst;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Ptr = std::shared_ptr<_Tp>,
          typename _Pair = std::pair<_Ptr, _Ptr>>
_Ptr
get_shared_ptr_pair_master_instance()
{
    static auto& _pinst = get_shared_ptr_pair<_Tp>();
    static auto  _inst  = _pinst.first;
    return _inst;
}

}  // namespace tim
