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

#include "timemory/api.hpp"

#include <array>
#include <mutex>

namespace tim
{
/// \typedef std::recursive_mutex mutex_t
/// \brief Recursive mutex is used for convenience since the
/// performance penalty vs. a regular mutex is not really an issue since there are not
/// many locks in general.
using mutex_t = std::recursive_mutex;

/// \typedef std::unique_lock<std::recursive_mutex> auto_lock_t
/// \brief Unique lock type around \ref mutex_t
using auto_lock_t = std::unique_lock<mutex_t>;

/// \fn mutex_t& type_mutex(uint64_t)
/// \tparam Tp data type for lock
/// \tparam ApiT API for lock
/// \tparam N max size
/// \tparam MutexT mutex data type
///
/// \brief A simple way to get a mutex for a class or common behavior, e.g.
/// `type_mutex<decltype(std::cout)>()` provides a mutex for synchronizing output streams.
/// Recommend using in conjunction with auto-lock:
/// `tim::auto_lock_t _lk{ type_mutex<Foo>() }`.
template <typename Tp, typename ApiT = TIMEMORY_API, size_t N = 4,
          typename MutexT = mutex_t>
MutexT&
type_mutex(uint64_t _n = 0)
{
    static std::array<MutexT, N> _mutexes{};
    return _mutexes.at(_n % N);
}

}  // namespace tim
