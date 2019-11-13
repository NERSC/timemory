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

/** \file component_tuple.hpp
 * \headerfile component_tuple.hpp "timemory/variadic/component_tuple.hpp"
 * This is the C++ class that bundles together components and enables
 * operation on the components as a single entity
 *
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include "timemory/backends/mpi.hpp"
#include "timemory/bits/settings.hpp"
#include "timemory/components.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/storage.hpp"
#include "timemory/variadic/types.hpp"

//======================================================================================//

namespace tim
{
//======================================================================================//
// variadic list of components
//
template <typename... Types>
class component_tuple
{
    // manager is friend so can use above
    friend class manager;

    template <typename _TupleC, typename _ListC>
    friend class component_hybrid;

    template <typename... _Types>
    friend class auto_tuple;

public:
    template <typename... _Types>
    struct filtered
    {};

    template <template <typename...> class _TypeL, typename... _Types>
    struct filtered<_TypeL<_Types...>>
    {
        using this_type  = component_tuple<_Types...>;
        using data_type  = std::tuple<_Types...>;
        using type_tuple = std::tuple<_Types...>;

        template <typename _Archive>
        using serialize_t = _TypeL<operation::serialization<Types, _Archive>...>;
        template <typename _Scope>
        using insert_node_t = _TypeL<operation::insert_node<_Types, _Scope>...>;
        using pop_node_t    = _TypeL<operation::pop_node<_Types>...>;
        using measure_t     = _TypeL<operation::measure<_Types>...>;
        using record_t      = _TypeL<operation::record<_Types>...>;
        using reset_t       = _TypeL<operation::reset<_Types>...>;
        using plus_t        = _TypeL<operation::plus<_Types>...>;
        using minus_t       = _TypeL<operation::minus<_Types>...>;
        using multiply_t    = _TypeL<operation::multiply<_Types>...>;
        using divide_t      = _TypeL<operation::divide<_Types>...>;
        using print_t       = _TypeL<operation::print<_Types>...>;
        using prior_start_t = _TypeL<operation::priority_start<_Types>...>;
        using prior_stop_t  = _TypeL<operation::priority_stop<_Types>...>;
        using stand_start_t = _TypeL<operation::standard_start<_Types>...>;
        using stand_stop_t  = _TypeL<operation::standard_stop<_Types>...>;
        using mark_begin_t  = _TypeL<operation::mark_begin<_Types>...>;
        using mark_end_t    = _TypeL<operation::mark_end<_Types>...>;
        using customize_t   = _TypeL<operation::customize<_Types>...>;
        using set_prefix_t  = _TypeL<operation::set_prefix<_Types>...>;
        using get_data_t    = _TypeL<operation::get_data<_Types>...>;
        using auto_type     = auto_tuple<_Types...>;
    };

    using impl_unique_concat_type = available_tuple<remove_duplicates<concat<Types...>>>;

public:
    using string_t            = std::string;
    using size_type           = int64_t;
    using string_hash         = std::hash<string_t>;
    using this_type           = component_tuple<Types...>;
    using data_type           = typename filtered<impl_unique_concat_type>::data_type;
    using type_tuple          = typename filtered<impl_unique_concat_type>::type_tuple;
    using data_value_type     = get_data_value_t<data_type>;
    using data_label_type     = get_data_label_t<data_type>;
    using captured_location_t = source_location::captured;

    // used by gotcha
    using component_type = typename filtered<impl_unique_concat_type>::this_type;
    using auto_type      = auto_tuple<Types...>;

    // used by component hybrid
    static constexpr bool is_component_list   = false;
    static constexpr bool is_component_tuple  = true;
    static constexpr bool is_component_hybrid = false;

    // used by gotcha component to prevent recursion
    static constexpr bool contains_gotcha =
        (std::tuple_size<filter_gotchas<Types...>>::value != 0);

public:
    // modifier types
    // clang-format off
    template <typename _Archive>
    using serialize_t   = typename filtered<impl_unique_concat_type>::template serialize_t<_Archive>;
    template <typename _Scope>
    using insert_node_t = typename filtered<impl_unique_concat_type>::template insert_node_t<_Scope>;
    using pop_node_t    = typename filtered<impl_unique_concat_type>::pop_node_t;
    using measure_t     = typename filtered<impl_unique_concat_type>::measure_t;
    using record_t      = typename filtered<impl_unique_concat_type>::record_t;
    using reset_t       = typename filtered<impl_unique_concat_type>::reset_t;
    using plus_t        = typename filtered<impl_unique_concat_type>::plus_t;
    using minus_t       = typename filtered<impl_unique_concat_type>::minus_t;
    using multiply_t    = typename filtered<impl_unique_concat_type>::multiply_t;
    using divide_t      = typename filtered<impl_unique_concat_type>::divide_t;
    using print_t       = typename filtered<impl_unique_concat_type>::print_t;
    using prior_start_t = typename filtered<impl_unique_concat_type>::prior_start_t;
    using prior_stop_t  = typename filtered<impl_unique_concat_type>::prior_stop_t;
    using stand_start_t = typename filtered<impl_unique_concat_type>::stand_start_t;
    using stand_stop_t  = typename filtered<impl_unique_concat_type>::stand_stop_t;
    using mark_begin_t  = typename filtered<impl_unique_concat_type>::mark_begin_t;
    using mark_end_t    = typename filtered<impl_unique_concat_type>::mark_end_t;
    using customize_t   = typename filtered<impl_unique_concat_type>::customize_t;
    using set_prefix_t  = typename filtered<impl_unique_concat_type>::set_prefix_t;
    using get_data_t    = typename filtered<impl_unique_concat_type>::get_data_t;
    // clang-format on

public:
    component_tuple();

    explicit component_tuple(const string_t& key, const bool& store = false,
                             const bool& flat = settings::flat_profile());
    explicit component_tuple(const captured_location_t& loc, const bool& store = false,
                             const bool& flat = settings::flat_profile());

    ~component_tuple();

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    component_tuple(const component_tuple&) = default;
    component_tuple(component_tuple&&)      = default;

    component_tuple& operator=(const component_tuple& rhs) = default;
    component_tuple& operator=(component_tuple&&) = default;

    component_tuple clone(bool store, bool flat = settings::flat_profile());

public:
    //----------------------------------------------------------------------------------//
    // public static functions
    //
    static constexpr std::size_t size() { return std::tuple_size<type_tuple>::value; }
    static void                  print_storage();
    static void                  init_storage();

    //----------------------------------------------------------------------------------//
    // public member functions
    //
    inline void             push();
    inline void             pop();
    void                    measure();
    void                    start();
    void                    stop();
    this_type&              record();
    void                    reset();
    data_value_type         get() const;
    data_label_type         get_labeled() const;
    inline data_type&       data();
    inline const data_type& data() const;
    inline int64_t          laps() const;
    inline uint64_t&        hash();
    inline string_t&        key();
    inline const uint64_t&  hash() const;
    inline const string_t&  key() const;
    inline void             rekey(const string_t& _key);
    inline bool&            store();
    inline const bool&      store() const;

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... _Args>
    void mark_begin(_Args&&... _args)
    {
        apply<void>::access<mark_begin_t>(m_data, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... _Args>
    void mark_end(_Args&&... _args)
    {
        apply<void>::access<mark_end_t>(m_data, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // perform a customized operation (typically for GOTCHA)
    //
    template <typename... _Args>
    void customize(_Args&&... _args)
    {
        apply<void>::access<customize_t>(m_data, std::forward<_Args>(_args)...);
    }

    // get member functions taking either a type
    template <typename _Tp>
    inline _Tp& get()
    {
        return std::get<index_of<_Tp, data_type>::value>(m_data);
    }

    template <typename _Tp>
    inline const _Tp& get() const
    {
        return std::get<index_of<_Tp, data_type>::value>(m_data);
    }

    //----------------------------------------------------------------------------------//
    template <typename _Tp, typename _Func, typename... _Args,
              enable_if_t<(is_one_of<_Tp, data_type>::value == true), int> = 0>
    inline void type_apply(_Func&& _func, _Args&&... _args)
    {
        auto&& _obj = get<_Tp>();
        ((_obj).*(_func))(std::forward<_Args>(_args)...);
    }

    template <typename _Tp, typename _Func, typename... _Args,
              enable_if_t<(is_one_of<_Tp, data_type>::value == false), int> = 0>
    inline void type_apply(_Func&&, _Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    // this_type operators
    //
    this_type& operator-=(const this_type& rhs);
    this_type& operator-=(this_type& rhs);
    this_type& operator+=(const this_type& rhs);
    this_type& operator+=(this_type& rhs);

    //----------------------------------------------------------------------------------//
    // generic operators
    //
    template <typename _Op>
    this_type& operator-=(_Op&& rhs)
    {
        apply<void>::access<minus_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator+=(_Op&& rhs)
    {
        apply<void>::access<plus_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator*=(_Op&& rhs)
    {
        apply<void>::access<multiply_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator/=(_Op&& rhs)
    {
        apply<void>::access<divide_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    //----------------------------------------------------------------------------------//
    // friend operators
    //
    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        return tmp += rhs;
    }

    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        this_type tmp(lhs);
        return tmp -= rhs;
    }

    template <typename _Op>
    friend this_type operator*(const this_type& lhs, _Op&& rhs)
    {
        this_type tmp(lhs);
        return tmp *= std::forward<_Op>(rhs);
    }

    template <typename _Op>
    friend this_type operator/(const this_type& lhs, _Op&& rhs)
    {
        this_type tmp(lhs);
        return tmp /= std::forward<_Op>(rhs);
    }

    //----------------------------------------------------------------------------------//
    //
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        if(size() == 0)
            return os;
        // stop, if not already stopped
        apply<void>::access<prior_stop_t>(obj.m_data);
        apply<void>::access<stand_stop_t>(obj.m_data);
        std::stringstream ss_prefix;
        std::stringstream ss_data;
        apply<void>::access_with_indices<print_t>(obj.m_data, std::ref(ss_data), false);
        if(obj.m_print_prefix)
        {
            obj.update_width();
            std::stringstream ss_id;
            ss_id << obj.get_prefix() << " " << std::left << obj.m_key;
            ss_prefix << std::setw(output_width()) << std::left << ss_id.str() << " : ";
            os << ss_prefix.str();
        }
        os << ss_data.str();
        if(obj.m_laps > 0 && obj.m_print_laps)
            os << " [laps: " << obj.m_laps << "]";
        return os;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar(serializer::make_nvp("key", m_key), serializer::make_nvp("laps", m_laps));
        ar.setNextName("data");
        ar.startNode();
        apply<void>::access<serialize_t<Archive>>(m_data, std::ref(ar), version);
        ar.finishNode();
    }

protected:
    // protected static functions
    static int64_t output_width(int64_t = 0);

    // protected member functions
    inline data_type&       get_data();
    inline const data_type& get_data() const;
    inline string_t         get_prefix() const;
    inline void             compute_width(const string_t&);
    inline void             update_width() const;
    inline void             set_object_prefix(const string_t&);

protected:
    // objects
    bool              m_store        = false;
    bool              m_flat         = false;
    bool              m_is_pushed    = false;
    bool              m_print_prefix = true;
    bool              m_print_laps   = true;
    int64_t           m_laps         = 0;
    uint64_t          m_hash         = 0;
    string_t          m_key          = "";
    mutable data_type m_data         = data_type();

public:
};

//======================================================================================//

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

//--------------------------------------------------------------------------------------//
//
//          Member function definitions
//
//--------------------------------------------------------------------------------------//

#include "timemory/variadic/bits/component_tuple.hpp"

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

//--------------------------------------------------------------------------------------//
