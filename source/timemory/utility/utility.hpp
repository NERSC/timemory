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

/** \file utility/utility.hpp
 * \headerfile utility/utility.hpp "timemory/utility/utility.hpp"
 * General utility functions
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/transient_function.hpp"
#include "timemory/utility/types.hpp"

#if defined(TIMEMORY_USE_LIBUNWIND)
#    include <libunwind.h>
#endif

// C library
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// I/O
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
// general
#include <functional>
#include <limits>
#include <regex>
#include <typeindex>
#include <utility>
// container
#include <vector>
// threading
#include <atomic>
#include <mutex>
#include <thread>

#if defined(TIMEMORY_UNIX)
#    include <execinfo.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#elif defined(TIMEMORY_WINDOWS)
#    include <direct.h>
using pid_t = int;
#endif

#if !defined(TIMEMORY_DEFAULT_UMASK)
#    define TIMEMORY_DEFAULT_UMASK 0777
#endif

//--------------------------------------------------------------------------------------//

namespace tim
{
// alias here for common string type
// there is also a string_view_t alias in macros/language.hpp which is std::string in
// c++14 and std::string_view in c++17 and newer
using string_t = std::string;

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline bool
isfinite(const Tp& arg)
{
#if defined(TIMEMORY_WINDOWS)
    // Windows seems to be missing std::isfinite
    return (arg == arg && arg != std::numeric_limits<Tp>::infinity() &&
            arg != -std::numeric_limits<Tp>::infinity())
               ? true
               : false;
#else
    return std::isfinite(arg);
#endif
}

//======================================================================================//
//
//  General functions
//
//======================================================================================//

namespace internal
{
template <typename Tp, typename Up = Tp>
inline auto
typeid_hash(int) -> decltype(demangle<Tp>(), size_t{})
{
    return std::type_index(typeid(Tp)).hash_code();
}
//
template <typename Tp, typename Up = Tp>
inline auto
typeid_hash(long)
{
    return 0;
}
}  // namespace internal

template <typename Tp>
inline auto
typeid_hash()
{
    return internal::typeid_hash<Tp>(0);
}

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE std::string
                        dirname(std::string _fname);

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE int
makedir(std::string _dir, int umask = TIMEMORY_DEFAULT_UMASK);

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE bool
get_bool(const std::string& strbool, bool _default = false) noexcept;

//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_UTILITY_INLINE std::vector<std::string>
                        read_command_line(pid_t _pid);

//======================================================================================//
//
//  path
//
//======================================================================================//

namespace utility
{
class path : public std::string
{
public:
    using size_type = std::string::size_type;

public:
    TIMEMORY_UTILITY_INLINE path(const std::string& _path);
    TIMEMORY_UTILITY_INLINE path(char* _path);
    TIMEMORY_UTILITY_INLINE path(const path& rhs);
    TIMEMORY_UTILITY_INLINE path(const char* _path);

    TIMEMORY_UTILITY_INLINE path& operator=(const std::string& rhs);
    TIMEMORY_UTILITY_INLINE path& operator=(const path& rhs);
    TIMEMORY_UTILITY_INLINE path& insert(size_type __pos, const std::string& __s);
    TIMEMORY_UTILITY_INLINE path& insert(size_type __pos, const path& __s);

    // OS-dependent representation
    static TIMEMORY_UTILITY_INLINE std::string osrepr(std::string _path);
    static TIMEMORY_UTILITY_INLINE std::string os();
    static TIMEMORY_UTILITY_INLINE std::string inverse();
    static TIMEMORY_UTILITY_INLINE std::string canonical(std::string _path);
};
}  // namespace utility

//--------------------------------------------------------------------------------------//

inline namespace hash
{
template <typename T>
TIMEMORY_INLINE size_t
get_hash(T&& obj)
{
    return std::hash<decay_t<T>>()(std::forward<T>(obj));
}

TIMEMORY_INLINE size_t
get_hash(string_view_cref_t str)
{
    return std::hash<string_view_t>{}(str);
}

TIMEMORY_INLINE size_t
get_hash(const char* cstr)
{
    return std::hash<string_view_t>{}(cstr);
}

template <typename T>
struct hasher
{
    inline size_t operator()(T&& val) const { return get_hash(std::forward<T>(val)); }
    inline size_t operator()(const T& val) const { return get_hash(val); }
};
}  // namespace hash
//--------------------------------------------------------------------------------------//

}  // namespace tim

#if defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/utility.cpp"
#endif
