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
tim::component_list<Types...>::init_manager()
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

template <typename... _Types,
          typename _Ret   = typename component_list<_Types...>::data_value_tuple,
          typename _Apply = std::tuple<
              operation::pointer_operator<_Types, operation::get_data<_Types>>...>>
_Ret
get(const component_list<_Types...>& _obj)
{
    const_cast<component_list<_Types...>&>(_obj).conditional_stop();
    _Ret _ret_data;
    apply<void>::access2<_Apply>(_obj.data(), _ret_data);
    return _ret_data;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types,
          typename _Ret   = typename component_list<_Types...>::data_label_tuple,
          typename _Apply = std::tuple<
              operation::pointer_operator<_Types, operation::get_data<_Types>>...>>
_Ret
get_labeled(const component_list<_Types...>& _obj)
{
    const_cast<component_list<_Types...>&>(_obj).conditional_stop();
    _Ret _ret_data;
    apply<void>::access2<_Apply>(_obj.data(), _ret_data);
    return _ret_data;
}

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
get(tim::component_list<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const tim::component_list<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(tim::component_list<Types...>&& obj)
    -> decltype(get<N>(std::forward<tim::component_list<Types...>>(obj).data()))
{
    using obj_type = tim::component_list<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}  // namespace std
