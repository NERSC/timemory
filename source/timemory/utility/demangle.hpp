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

#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <typeinfo>

#if defined(TIMEMORY_UNIX)
#    include <cxxabi.h>
#endif

namespace tim
{
//--------------------------------------------------------------------------------------//

inline std::string
demangle(const char* _mangled_name, int* _status = nullptr)
{
#if defined(TIMEMORY_ENABLE_DEMANGLE)
    // return the mangled since there is no buffer
    if(!_mangled_name)
        return std::string{};

    int         _ret = 0;
    std::string _demangled_name{ _mangled_name };
    if(!_status)
        _status = &_ret;

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A NULL-terminated character string containing the name to be demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into which the
    //      demangled name is stored. If output_buffer is not long enough, it is expanded
    //      using realloc. output_buffer may instead be NULL; in that case, the demangled
    //      name is placed in a region of memory allocated with malloc.
    //  _buflen:
    //      If length is non-NULL, the length of the buffer containing the demangled name
    //      is placed in *length.
    //  status:
    //      *status is set to one of the following values
    char* _demang = abi::__cxa_demangle(_mangled_name, nullptr, nullptr, _status);
    switch(*_status)
    {
        //  0 : The demangling operation succeeded.
        // -1 : A memory allocation failiure occurred.
        // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
        // -3 : One of the arguments is invalid.
        case 0:
        {
            if(_demang)
                _demangled_name = std::string{ _demang };
            break;
        }
        case -1:
        {
            char _msg[1024];
            ::memset(_msg, '\0', 1024 * sizeof(char));
            ::snprintf(_msg, 1024, "memory allocation failure occurred demangling %s",
                       _mangled_name);
            ::perror(_msg);
            break;
        }
        case -2: break;
        case -3:
        {
            char _msg[1024];
            ::memset(_msg, '\0', 1024 * sizeof(char));
            ::snprintf(_msg, 1024, "Invalid argument in: (\"%s\", nullptr, nullptr, %p)",
                       _mangled_name, (void*) _status);
            ::perror(_msg);
            break;
        }
        default: break;
    };

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
#else
    (void) _status;
    return _mangled_name;
#endif
}

//--------------------------------------------------------------------------------------//

inline std::string
demangle(const std::string& _str, int* _status = nullptr)
{
    return demangle(_str.c_str(), _status);
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline auto
try_demangle()
{
    // static because a type demangle will always be the same
    static auto _val = []() {
        // wrap the type in type_list and then extract ... from tim::type_list<...>
        auto _tmp = ::tim::demangle(typeid(type_list<Tp>).name());
        auto _key = std::string{ "type_list" };
        auto _idx = _tmp.find(_key);
        _idx      = _tmp.find('<', _idx);
        _tmp      = _tmp.substr(_idx + 1);
        _idx      = _tmp.find_last_of('>');
        _tmp      = _tmp.substr(0, _idx);
        // strip trailing whitespaces
        while((_idx = _tmp.find_last_of(' ')) == _tmp.length() - 1)
            _tmp = _tmp.substr(0, _idx);
        return _tmp;
    }();
    return _val;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline auto
demangle()
{
    // static because a type demangle will always be the same
    static auto _val = demangle(typeid(Tp).name());
    return _val;
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

#if defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/demangle.cpp"
#endif
