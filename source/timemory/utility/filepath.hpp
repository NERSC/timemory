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

/** \file utility/filepath.hpp
 * \headerfile utility/filepath.hpp "timemory/utility/filepath.hpp"
 * Functions for converting OS filepaths
 *
 */

#pragma once

#include <string>

//--------------------------------------------------------------------------------------//
// base operating system

#if defined(_WIN32) || defined(_WIN64)
#    if !defined(_WINDOWS)
#        define _WINDOWS
#    endif
#elif defined(__APPLE__) || defined(__MACH__) || defined(__linux__) ||                   \
    defined(__linux) || defined(linux) || defined(__gnu_linux__)
#    if !defined(_UNIX)
#        define _UNIX
#    endif
#endif

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace filepath
{
using string_t = std::string;

#if defined(_WINDOWS)

inline string_t
os()
{
    return "\\";
}

inline string_t
inverse()
{
    return "/";
}

inline string_t
osrepr(string_t _path)
{
    // OS-dependent representation
    while(_path.find("/") != std::string::npos)
        _path.replace(_path.find("/"), 1, "\\");
    return _path;
}

#elif defined(_UNIX)

inline string_t
os()
{
    return "/";
}

inline string_t
inverse()
{
    return "\\";
}

inline string_t
osrepr(string_t _path)
{
    // OS-dependent representation
    while(_path.find("\\\\") != std::string::npos)
        _path.replace(_path.find("\\\\"), 2, "/");
    while(_path.find('\\') != std::string::npos)
        _path.replace(_path.find('\\'), 1, "/");
    return _path;
}

#endif

}  // namespace filepath
}  // namespace tim

//--------------------------------------------------------------------------------------//
