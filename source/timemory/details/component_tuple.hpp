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

#include "timemory/manager.hpp"
#include "timemory/mpl/filters.hpp"

//======================================================================================//

template <typename... Types>
void
tim::component_tuple<Types...>::init_manager()
{
    tim::manager::instance();
}

//======================================================================================//
//
//      tim::get functions
//
namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename... _Types>
auto
get(const component_tuple<_Types...>& _obj)
    -> decltype(std::declval<component_tuple<_Types...>>().get())
{
    return _obj.get();
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
auto
get_labeled(const component_tuple<_Types...>& _obj)
    -> decltype(std::declval<component_tuple<_Types...>>().get_labeled())
{
    return _obj.get_labeled();
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//
//
//      std::get operator
//
namespace std
{
//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
typename std::tuple_element<N, std::tuple<Types...>>::type&
get(::tim::component_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const ::tim::component_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(::tim::component_tuple<Types...>&& obj)
    -> decltype(get<N>(std::forward<::tim::component_tuple<Types...>>(obj).data()))
{
    using obj_type = ::tim::component_tuple<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}  // namespace std
