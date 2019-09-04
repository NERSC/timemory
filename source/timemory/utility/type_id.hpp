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

//--------------------------------------------------------------------------------------//

#include "timemory/mpl/apply.hpp"
#include "timemory/utility/macros.hpp"

//--------------------------------------------------------------------------------------//

#include <array>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace cupti
{
struct result;
}  // namespace cupti

//======================================================================================//
// static functions that return a string identifying the data type (used in Python plot)
//
// * DEPRECATED *
//
template <typename _Tp>
struct type_id
{
    template <typename Type = _Tp, enable_if_t<(std::is_integral<Type>::value), int> = 0>
    static std::string value(const Type&)
    {
        return "int";
    }

    template <typename Type                                           = _Tp,
              enable_if_t<(std::is_floating_point<Type>::value), int> = 0>
    static std::string value(const Type&)
    {
        return "float";
    }

    template <typename SubType, enable_if_t<(std::is_integral<SubType>::value), int> = 0>
    static std::string value(const std::pair<SubType, SubType>&)
    {
        return "int_pair";
    }

    template <typename SubType,
              enable_if_t<(std::is_floating_point<SubType>::value), int> = 0>
    static std::string value(const std::pair<SubType, SubType>&)
    {
        return "float_pair";
    }

    template <typename... _Types>
    static std::string value(const std::tuple<_Types...>&)
    {
        using type = std::tuple<_Types...>;
        std::stringstream ss;
        ss << "tuple_" << std::tuple_size<type>::value;
        return ss.str();
    }

    template <typename SubType, std::size_t SubTypeSize,
              enable_if_t<(std::is_integral<SubType>::value), int> = 0>
    static std::string value(const std::array<SubType, SubTypeSize>&)
    {
        return std::string("int_array_") + std::to_string(SubTypeSize);
    }

    template <typename SubType, std::size_t SubTypeSize,
              enable_if_t<(std::is_floating_point<SubType>::value), int> = 0>
    static std::string value(const std::array<SubType, SubTypeSize>&)
    {
        return std::string("float_array_") + std::to_string(SubTypeSize);
    }

    template <typename _Up, typename SubType, std::size_t SubTypeSize,
              enable_if_t<(std::is_integral<SubType>::value), int> = 0>
    static std::string value(const std::pair<std::array<SubType, SubTypeSize>, _Up>&)
    {
        return std::string("pair_int_array_") + std::to_string(SubTypeSize);
    }

    template <typename _Up, typename SubType, std::size_t SubTypeSize,
              enable_if_t<(std::is_floating_point<SubType>::value), int> = 0>
    static std::string value(const std::pair<std::array<SubType, SubTypeSize>, _Up>&)
    {
        return std::string("pair_float_array_") + std::to_string(SubTypeSize);
    }

    template <typename _Up, typename SubType, typename... _Extra,
              enable_if_t<(std::is_integral<SubType>::value), int> = 0>
    static std::string value(const std::pair<std::vector<SubType, _Extra...>, _Up>&)
    {
        return std::string("pair_int_vector");
    }

    template <typename _Up, typename SubType, typename... _Extra,
              enable_if_t<(std::is_floating_point<SubType>::value), int> = 0>
    static std::string value(const std::pair<std::vector<SubType, _Extra...>, _Up>&)
    {
        return std::string("pair_float_vector");
    }

    static std::string value(const std::vector<cupti::result>&)
    {
        return "result_vector";
    }
};

}  // namespace tim
