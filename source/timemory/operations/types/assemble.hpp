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

/**
 * \file timemory/operations/types/assemble.hpp
 * \brief Definition for various functions for assemble in operations
 */

#pragma once

#include "timemory/mpl/filters.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

#include <type_traits>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct fold_assemble;

template <template <typename...> class CompT, typename... C>
struct fold_assemble<CompT<C...>>
{
public:
    TIMEMORY_DELETED_OBJECT(fold_assemble)

    template <typename Up, typename Arg, typename... Args>
    fold_assemble(bool& b, Up& obj, Arg&& arg, Args&&...)
    {
        if(!b)
            sfinae(b, obj, 0, std::forward<Arg>(arg));
    }

private:
    //  satisfies mpl condition and accepts arguments
    template <typename Up, typename Arg>
    auto sfinae(bool& b, Up& obj, int, Arg&& arg)
        -> decltype(obj.assemble(arg.template get_component<C>()...), void())
    {
        b = obj.assemble(arg.template get_component<C>()...);
    }

    template <typename Up, typename Arg>
    void sfinae(bool&, Up&, long, Arg&&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct assemble
{
    using type = Tp;

    TIMEMORY_DELETED_OBJECT(assemble)

private:
    using derived_tuple_t                   = typename trait::derivation_types<Tp>::type;
    static constexpr size_t derived_tuple_v = std::tuple_size<derived_tuple_t>::value;
    template <size_t Idx>
    using derived_t = typename std::tuple_element<Idx, derived_tuple_t>::type;

public:
    template <typename... Args>
    explicit assemble(type& obj, Args&&... args);

    template <typename Arg, size_t N = derived_tuple_v, std::enable_if_t<(N > 0)> = 0>
    explicit assemble(type& obj, Arg&& arg)
    {
        bool b = false;
        sfinae(b, obj, make_index_sequence<N>{}, std::forward<Arg>(arg));
        if(!b)
            sfinae(obj, 0, 0, std::forward<Arg>(arg));
    }

    template <template <typename...> class BundleT, typename... Args>
    explicit assemble(type& obj, BundleT<Args...>& arg)
    {
        bool           b = false;
        constexpr auto N = derived_tuple_v;
        sfinae(b, obj, make_index_sequence<N>{}, arg);
        if(!b)
            sfinae(obj, 0, 0, arg);
    }

private:
    //  satisfies mpl condition and accepts arguments
    template <typename Up, size_t... Idx, typename... Args>
    auto sfinae(bool& b, Up& obj, index_sequence<Idx...>, Args&&... args)
    {
        TIMEMORY_FOLD_EXPRESSION(
            fold_assemble<derived_t<Idx>>(b, obj, std::forward<Args>(args)...));
    }

private:
    //  satisfies mpl condition and accepts arguments
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.assemble(std::forward<Args>(args)...), void())
    {
        obj.assemble(std::forward<Args>(args)...);
    }

    //  satisfies mpl condition but does not accept arguments
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.assemble(), void())
    {
        obj.assemble();
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {
        // SFINAE_WARNING(type);
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
assemble<Tp>::assemble(type& obj, Args&&... args)
{
    sfinae(obj, 0, 0, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
