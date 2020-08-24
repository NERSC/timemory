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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/**
 * \file timemory/components/rusage/backends.hpp
 * \brief Implementation of the rusage functions/utilities
 */

#pragma once

#include "timemory/backends/process.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/variadic/macros.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iosfwd>
#include <string>
#include <type_traits>

#if defined(_UNIX)
#    include <sys/resource.h>
#    include <unistd.h>
#    if defined(_MACOS)
#        include <libproc.h>
#        include <mach/mach.h>
#    endif
#endif

//======================================================================================//
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
struct io_cache
{
    static inline auto get_filename()
    {
        return TIMEMORY_JOIN('/', "/proc", process::get_target_id(), "io");
    }

    template <size_t N, size_t... Idx>
    static inline auto& read(std::ifstream& ifs, std::array<int64_t, N>& _data,
                             std::index_sequence<Idx...>)
    {
        static_assert(N <= 6, "Error! Only six entries in the /proc/<PID>/io");
        static_assert(N > 0, "Error! array size is zero");
        static_assert(sizeof...(Idx) <= N,
                      "Error! Number of indexes to read exceeds the array size");
        if(ifs)
        {
            std::string label = "";
            TIMEMORY_FOLD_EXPRESSION(ifs >> label >> std::get<Idx>(_data));
        }
        else
        {
            _data.fill(0);
        }
        return _data;
    }

    template <size_t NumReads = 6, size_t N>
    static inline auto& read(std::array<int64_t, N>& _data)
    {
        std::ifstream ifs(get_filename().c_str());
        _data = read(ifs, _data, std::make_index_sequence<NumReads>{});
        return _data;
    }

    template <size_t NumReads = 6>
    static inline auto read()
    {
        std::array<int64_t, NumReads> _data{};
        _data = read<NumReads>(_data);
        return _data;
    }

public:
#if defined(_LINUX)
    io_cache()
    : m_data(read())
    {}
#else
    io_cache() = default;
#endif

    ~io_cache()               = default;
    io_cache(const io_cache&) = delete;
    io_cache& operator=(const io_cache&) = delete;
    io_cache(io_cache&&)                 = default;
    io_cache& operator=(io_cache&&) = default;

#if !defined(_LINUX)

    inline int64_t get_char_read() const { return 0; }
    inline int64_t get_char_written() const { return 0; }
    inline int64_t get_syscall_read() const { return 0; }
    inline int64_t get_syscall_written() const { return 0; }
    inline int64_t get_bytes_read() const { return 0; }
    inline int64_t get_bytes_written() const { return 0; }

#else

    inline int64_t get_char_read() const { return std::get<0>(m_data); }
    inline int64_t get_char_written() const { return std::get<1>(m_data); }
    inline int64_t get_syscall_read() const { return std::get<2>(m_data); }
    inline int64_t get_syscall_written() const { return std::get<3>(m_data); }
    inline int64_t get_bytes_read() const { return std::get<4>(m_data); }
    inline int64_t get_bytes_written() const { return std::get<5>(m_data); }

private:
    std::array<int64_t, 6> m_data{};
#endif
};
//
//--------------------------------------------------------------------------------------//
//
int64_t
get_char_read();
int64_t
get_char_written();
int64_t
get_syscall_read();
int64_t
get_syscall_written();
int64_t
get_bytes_read();
int64_t
get_bytes_written();
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_char_read()
{
#if defined(_LINUX)
    // read one value and return it
    return io_cache::read<1>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_char_written()
{
#if defined(_LINUX)
    // read two values and return the last one
    return io_cache::read<2>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_syscall_read()
{
#if defined(_LINUX)
    // read one value and return it
    return io_cache::read<3>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_syscall_written()
{
#if defined(_LINUX)
    // read two values and return the last one
    return io_cache::read<4>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_bytes_read()
{
#if defined(_MACOS)
    rusage_info_current rusage;
    if(proc_pid_rusage(process::get_target_id(), RUSAGE_INFO_CURRENT, (void**) &rusage) ==
       0)
        return rusage.ri_diskio_bytesread;
    return 0;
#elif defined(_LINUX)
    // read three values and return the last one
    return io_cache::read<5>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
inline int64_t
tim::get_bytes_written()
{
#if defined(_MACOS)
    rusage_info_current rusage;
    if(proc_pid_rusage(process::get_target_id(), RUSAGE_INFO_CURRENT, (void**) &rusage) ==
       0)
        return rusage.ri_diskio_byteswritten;
    return 0;
#elif defined(_LINUX)
    // read four values and return the last one
    return io_cache::read<6>().back();
#else
    return 0;
#endif
}
//
//--------------------------------------------------------------------------------------//
//
