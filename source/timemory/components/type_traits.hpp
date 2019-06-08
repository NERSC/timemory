//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

#include "timemory/components/types.hpp"
#include <type_traits>

//======================================================================================//
//
//                                 Type Traits
//
//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
/// trait that signifies that updating w.r.t. another instance should
/// be a max of the two instances
//
template <typename _Tp>
struct record_max : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// trait that signifies that data is an array type
///
template <typename _Tp>
struct array_serialization
{
    using type = std::false_type;
};

//--------------------------------------------------------------------------------------//
/// trait that signifies that an implementation (e.g. PAPI) is available
///
template <typename _Tp>
struct impl_available : std::true_type
{};

//--------------------------------------------------------------------------------------//
/// trait that registers how to construct a type
///
template <typename T>
struct construct_traits : public construct_traits<decltype(&T::operator())>
{};

template <typename ClassType, typename ReturnType, typename... Args>
struct construct_traits<ReturnType (ClassType::*)(Args...) const>
{
    using result_type           = ReturnType;
    using arg_tuple             = std::tuple<Args...>;
    static constexpr auto nargs = sizeof...(Args);
};

template <typename Ret, typename... Args>
struct construct_traits<Ret (&)(Args...)>
{
    using result_type           = Ret;
    using arg_tuple             = std::tuple<Args...>;
    static constexpr auto nargs = sizeof...(Args);
};

/*
template <typename _Tp, typename... _Args>
struct constructor
{
    using signature = void(_Args...);

    template <typename Callable>
    void Register(Callable && callable)
    {
        using ft = function_traits<Callable>;
        static_assert(
            std::is_same<
                int,
                std::decay_t<std::tuple_element_t<0, typename ft::arg_tuple>>>::value,
            "");
        static_assert(
            std::is_same<
                double,
                std::decay_t<std::tuple_element_t<1, typename ft::arg_tuple>>>::value,
            "");
        static_assert(std::is_same<void, std::decay_t<typename ft::result_type>>::value,
                      "");

        callback = callable;
    }

    std::function<Signature> callback;
};
*/
//--------------------------------------------------------------------------------------//
}  // component
}  // tim

//======================================================================================//
//
//                              Specifications
//
//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
template <>
struct record_max<peak_rss> : std::true_type
{};

template <>
struct record_max<current_rss> : std::true_type
{};

template <>
struct record_max<stack_rss> : std::true_type
{};

template <>
struct record_max<data_rss> : std::true_type
{};

template <int EventSet, int... EventTypes>
struct array_serialization<papi_tuple<EventSet, EventTypes...>>
{
    using type = std::true_type;
};

template <int EventSet, std::size_t NumEvent>
struct array_serialization<papi_array<EventSet, NumEvent>>
{
    using type = std::true_type;
};

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_PAPI
//
#if !defined(TIMEMORY_USE_PAPI)

template <int EventSet, int... EventTypes>
struct impl_available<papi_tuple<EventSet, EventTypes...>> : std::false_type
{};

template <int EventSet, std::size_t NumEvent>
struct impl_available<papi_array<EventSet, NumEvent>> : std::false_type
{};

template <int... EventTypes>
struct impl_available<cpu_roofline<EventTypes...>> : std::false_type
{};

#endif  // TIMEMORY_USE_PAPI

//--------------------------------------------------------------------------------------//
//  disable if not enabled via preprocessor TIMEMORY_USE_CUDA
//
#if !defined(TIMEMORY_USE_CUDA)

template <>
struct impl_available<cuda_event> : std::false_type
{};

#endif  // TIMEMORY_USE_CUDA

//--------------------------------------------------------------------------------------//
}  // component
}  // tim
