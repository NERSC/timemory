// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include "timemory/mpl/apply.hpp"
#include "timemory/utility/type_id.hpp"
#include "timemory/utility/utility.hpp"

#include <string>
#include <tuple>

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace impl
{
template <typename... _Args>
struct mangler
{
    static std::string mangle(std::string func)
    {
        std::string ret   = "_Z";
        auto        delim = delimit(func, ":()<>");
        if(delim.size() > 1)
            ret += "N";
        for(const auto& itr : delim)
        {
            ret += std::to_string(itr.length());
            ret += itr;
        }
        ret += "E";
        auto arg_string = apply<std::string>::join("", type_id<_Args>::name()...);
        ret += arg_string;
        printf("[generated_mangle]> %s --> %s\n", func.c_str(), ret.c_str());
        return ret;
    }
};

//--------------------------------------------------------------------------------------//

template <typename... _Args>
struct mangler<std::tuple<_Args...>>
{
    static std::string mangle(std::string func)
    {
        return mangler<_Args...>::mangle(func);
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace impl

//--------------------------------------------------------------------------------------//

template <typename _Func, typename _Tuple = typename function_traits<_Func>::arg_tuple>
std::string
mangle(const std::string& func)
{
    return impl::mangler<_Tuple>::mangle(func);
}

//--------------------------------------------------------------------------------------//

}  // namespace tim
