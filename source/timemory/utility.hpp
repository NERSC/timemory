// MIT License
//
// Copyright (c) 2018, The Regents of the University of California, 
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
//

/** \file utility.hpp
 * General utility functions
 *
 */

#ifndef TIMEMORY_UTIL_INTERNAL_HPP
#define TIMEMORY_UTIL_INTERNAL_HPP

#include <iostream>
#include <cstdlib>
#include <set>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <functional>
#include <deque>
#include <cmath>
#include <vector>

#include "timemory/namespace.hpp"

namespace NAME_TIM
{

//----------------------------------------------------------------------------//

typedef std::deque<std::string> str_list_t;

//----------------------------------------------------------------------------//

namespace internal
{
inline std::string dummy_str_return(std::string str) { return str; }
}

//----------------------------------------------------------------------------//

template <typename _Tp>
_Tp get_env(const std::string& env_id, _Tp _default = _Tp())
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string str_var = std::string(env_var);
        std::istringstream iss(str_var);
        _Tp var = _Tp();
        iss >> var;
        return var;
    }
    // return default if not specified in environment
    return _default;
}

//----------------------------------------------------------------------------//
//  delimit line : e.g. delimit_line("a B\t c", " \t") --> { "a", "B", "c"}
inline str_list_t
delimit(const std::string& _str, const std::string& _delims,
        std::string (*strop)(std::string) = internal::dummy_str_return)
{
    str_list_t _list;
    size_t _beg = 0;
    size_t _end = 0;
    while(true)
    {
        _beg = _str.find_first_not_of(_delims, _end);
        if(_beg == std::string::npos)
            break;
        _end = _str.find_first_of(_delims, _beg);
        if(_beg < _end)
            _list.push_back(strop(_str.substr(_beg, _end - _beg)));
    }
    return _list;
}

}


#endif
