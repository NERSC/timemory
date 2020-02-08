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

/** \file timemory/variadic/component_tuple.hpp
 * \headerfile variadic/component_tuple.hpp "timemory/variadic/component_tuple.hpp"
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

#include "timemory/backends/dmp.hpp"
#include "timemory/components.hpp"
#include "timemory/data/storage.hpp"
#include "timemory/general/source_location.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/variadic/generic_bundle.hpp"
#include "timemory/variadic/types.hpp"

//======================================================================================//

namespace tim
{
//======================================================================================//
// variadic list of components
//
template <typename... Types>
class component_tuple : public generic_bundle<Types...>
{
    // manager is friend so can use above
    friend class manager;

    template <typename _TupleC, typename _ListC>
    friend class component_hybrid;

    template <typename... _Types>
    friend class auto_tuple;

public:
    using bundle_type         = generic_bundle<Types...>;
    using this_type           = component_tuple<Types...>;
    using init_func_t         = std::function<void(this_type&)>;
    using captured_location_t = source_location::captured;

    using data_type       = typename bundle_type::data_type;
    using impl_type       = typename bundle_type::impl_type;
    using type_tuple      = typename bundle_type::type_tuple;
    using sample_type     = typename bundle_type::sample_type;
    using pointer_type    = typename bundle_type::pointer_type;
    using reference_type  = typename bundle_type::reference_type;
    using data_value_type = typename bundle_type::data_value_type;
    using data_label_type = typename bundle_type::data_label_type;

    using apply_v     = apply<void>;
    using size_type   = typename bundle_type::size_type;
    using string_t    = typename bundle_type::string_t;
    using string_hash = typename bundle_type::string_hash;

    template <template <typename> class Op, typename _Tuple = impl_type>
    using operation_t =
        typename bundle_type::template generic_operation<Op, _Tuple>::type;

    // used by gotcha
    using component_type = convert_t<data_type, component_tuple<>>;
    using auto_type      = convert_t<data_type, auto_tuple<>>;

    // used by component hybrid
    static constexpr bool is_component_list   = false;
    static constexpr bool is_component_tuple  = true;
    static constexpr bool is_component_hybrid = false;
    static constexpr bool is_component_type   = true;
    static constexpr bool is_auto_list        = false;
    static constexpr bool is_auto_tuple       = false;
    static constexpr bool is_auto_hybrid      = false;
    static constexpr bool is_auto_type        = false;
    static constexpr bool is_component        = false;

    //----------------------------------------------------------------------------------//
    //
    static init_func_t& get_initializer()
    {
        static init_func_t _instance = [](this_type&) {};
        return _instance;
    }

public:
    component_tuple();

    template <typename _Func = init_func_t>
    explicit component_tuple(const string_t& key, const bool& store = true,
                             const bool& flat = settings::flat_profile(),
                             const _Func&     = get_initializer());

    template <typename _Func = init_func_t>
    explicit component_tuple(const captured_location_t& loc, const bool& store = true,
                             const bool& flat = settings::flat_profile(),
                             const _Func&     = get_initializer());

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
    void                    sample();
    void                    start();
    void                    stop();
    this_type&              record();
    void                    reset();
    data_value_type         get() const;
    data_label_type         get_labeled() const;
    inline data_type&       data();
    inline const data_type& data() const;

    using bundle_type::hash;
    using bundle_type::key;
    using bundle_type::laps;
    using bundle_type::rekey;
    using bundle_type::store;

    //----------------------------------------------------------------------------------//
    // construct the objects that have constructors with matching arguments
    //
    template <typename... _Args>
    void construct(_Args&&... _args)
    {
        using construct_t = operation_t<operation::construct>;
        apply_v::access<construct_t>(m_data, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... _Args>
    void mark_begin(_Args&&... _args)
    {
        using mark_begin_t = operation_t<operation::mark_begin>;
        apply_v::access<mark_begin_t>(m_data, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // mark a beginning position in the execution (typically used by asynchronous
    // structures)
    //
    template <typename... _Args>
    void mark_end(_Args&&... _args)
    {
        using mark_end_t = operation_t<operation::mark_end>;
        apply_v::access<mark_end_t>(m_data, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // store a value
    //
    template <typename... _Args>
    void store(_Args&&... _args)
    {
        using store_t = operation_t<operation::store>;
        apply_v::access<store_t>(m_data, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // perform a auditd operation (typically for GOTCHA)
    //
    template <typename... _Args>
    void audit(_Args&&... _args)
    {
        using audit_t = operation_t<operation::audit>;
        apply_v::access<audit_t>(m_data, std::forward<_Args>(_args)...);
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
        using minus_t = operation_t<operation::minus>;
        apply_v::access<minus_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator+=(_Op&& rhs)
    {
        using plus_t = operation_t<operation::plus>;
        apply_v::access<plus_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator*=(_Op&& rhs)
    {
        using multiply_t = operation_t<operation::multiply>;
        apply_v::access<multiply_t>(m_data, std::forward<_Op>(rhs));
        return *this;
    }

    template <typename _Op>
    this_type& operator/=(_Op&& rhs)
    {
        using divide_t = operation_t<operation::divide>;
        apply_v::access<divide_t>(m_data, std::forward<_Op>(rhs));
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
    template <bool PrintPrefix = true, bool PrintLaps = true>
    void print(std::ostream& os) const
    {
        using print_t = typename bundle_type::print_t;
        if(size() == 0)
            return;
        std::stringstream ss_prefix;
        std::stringstream ss_data;
        apply_v::access_with_indices<print_t>(m_data, std::ref(ss_data), false);
        if(PrintPrefix)
        {
            auto _key = get_hash_ids()->find(m_hash)->second;
            update_width();
            std::stringstream ss_id;
            ss_id << get_prefix() << " " << std::left << _key;
            ss_prefix << std::setw(output_width()) << std::left << ss_id.str() << " : ";
            os << ss_prefix.str();
        }
        os << ss_data.str();
        if(m_laps > 0 && PrintLaps)
            os << " [laps: " << m_laps << "]";
    }

    //----------------------------------------------------------------------------------//
    //
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        obj.print<true, true>(os);
        return os;
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        std::string _key   = "";
        auto        keyitr = get_hash_ids()->find(m_hash);
        if(keyitr != get_hash_ids()->end())
            _key = keyitr->second;

        ar(cereal::make_nvp("hash", m_hash), cereal::make_nvp("key", _key),
           cereal::make_nvp("laps", m_laps));

        if(keyitr == get_hash_ids()->end())
        {
            auto _hash = add_hash_id(_key);
            if(_hash != m_hash)
                PRINT_HERE("Warning! Hash for '%s' (%llu) != %llu", _key.c_str(),
                           (unsigned long long) _hash, (unsigned long long) m_hash);
        }

        ar(cereal::make_nvp("data", m_data));
    }

public:
    int64_t         laps() const { return bundle_type::laps(); }
    std::string     key() const { return bundle_type::key(); }
    uint64_t        hash() const { return bundle_type::hash(); }
    void            rekey(const string_t& _key) { bundle_type::rekey(_key); }
    bool&           store() { return bundle_type::store(); }
    const bool&     store() const { return bundle_type::store(); }
    const string_t& prefix() const { return bundle_type::prefix(); }
    const string_t& get_prefix() const { return bundle_type::get_prefix(); }

protected:
    static int64_t output_width(int64_t w = 0) { return bundle_type::output_width(w); }
    void           update_width() const { bundle_type::update_width(); }
    void compute_width(const string_t& _key) const { bundle_type::compute_width(_key); }

protected:
    // protected member functions
    inline data_type&       get_data();
    inline const data_type& get_data() const;
    inline void             set_object_prefix(const string_t&) const;

protected:
    // objects
    using bundle_type::m_flat;
    using bundle_type::m_hash;
    using bundle_type::m_is_pushed;
    using bundle_type::m_laps;
    using bundle_type::m_store;
    mutable data_type m_data = data_type{};
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
