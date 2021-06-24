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

/** \file timemory/variadic/lightweight_tuple.hpp
 * This is the C++ class that bundles together components and enables
 * operation on the components as a single entity
 *
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/general/source_location.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/transient_function.hpp"
#include "timemory/variadic/base_bundle.hpp"
#include "timemory/variadic/functional.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

namespace tim
{
/// \class tim::lightweight_tuple
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper which provides the least amount of
/// runtime and compilation overhead.
///
///
template <typename... Types>
class lightweight_tuple
: public stack_bundle<mpl::available_t<type_list<Types...>>>
, public concepts::comp_wrapper
{
protected:
    using apply_v     = mpl::apply<void>;
    using bundle_type = stack_bundle<mpl::available_t<type_list<Types...>>>;
    using impl_type   = typename bundle_type::impl_type;

    template <typename... Tp>
    friend class impl::base_bundle;

public:
    using captured_location_t = source_location::captured;

    using this_type      = lightweight_tuple<Types...>;
    using type_list_type = type_list<Types...>;

    using data_type         = typename bundle_type::data_type;
    using tuple_type        = typename bundle_type::tuple_type;
    using sample_type       = typename bundle_type::sample_type;
    using reference_type    = typename bundle_type::reference_type;
    using user_bundle_types = typename bundle_type::user_bundle_types;

    using size_type = typename bundle_type::size_type;
    using string_t  = typename bundle_type::string_t;

    template <template <typename> class Op, typename Tuple = impl_type>
    using operation_t = typename bundle_type::template generic_operation<Op, Tuple>::type;

    template <template <typename> class Op, typename Tuple = impl_type>
    using custom_operation_t =
        typename bundle_type::template custom_operation<Op, Tuple>::type;

    using auto_type        = mpl::append_type_t<quirk::auto_start, this_type>;
    using component_type   = mpl::remove_type_t<quirk::auto_start, this_type>;
    using type             = convert_t<tuple_type, lightweight_tuple<>>;
    using initializer_type = std::function<void(this_type&)>;
    using transient_func_t = utility::transient_function<void(this_type&)>;

    static constexpr bool has_gotcha_v      = bundle_type::has_gotcha_v;
    static constexpr bool has_user_bundle_v = bundle_type::has_user_bundle_v;

public:
    static initializer_type& get_initializer()
    {
        static initializer_type _instance = [](this_type&) {};
        return _instance;
    }

    template <typename T, typename... U>
    using quirk_config = tim::variadic::impl::quirk_config<T, type_list<Types...>, U...>;

public:
    lightweight_tuple() = default;

    template <typename... T>
    explicit lightweight_tuple(const string_t& key, quirk::config<T...> = {},
                               transient_func_t = get_initializer());

    template <typename... T>
    explicit lightweight_tuple(const captured_location_t& loc, quirk::config<T...> = {},
                               transient_func_t = get_initializer());

    template <typename... T>
    explicit lightweight_tuple(size_t _hash, quirk::config<T...> = {},
                               transient_func_t = get_initializer());

    explicit lightweight_tuple(size_t _hash, scope::config _scope,
                               transient_func_t = get_initializer());

    explicit lightweight_tuple(const string_t& key, scope::config _scope,
                               transient_func_t = get_initializer());

    explicit lightweight_tuple(const captured_location_t& loc, scope::config _scope,
                               transient_func_t = get_initializer());

    ~lightweight_tuple();

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    lightweight_tuple(const lightweight_tuple&)     = default;
    lightweight_tuple(lightweight_tuple&&) noexcept = default;

    lightweight_tuple& operator=(const lightweight_tuple& rhs) = default;
    lightweight_tuple& operator=(lightweight_tuple&&) noexcept = default;

    lightweight_tuple clone(bool store, scope::config _scope = scope::get_default());

public:
    //----------------------------------------------------------------------------------//
    // public static functions
    //
    static constexpr std::size_t size() { return std::tuple_size<tuple_type>::value; }
    /// requests the component initialize their storage
    static void init_storage();

    //----------------------------------------------------------------------------------//
    // public member functions
    //
    this_type& push();
    this_type& pop();
    template <typename... Args>
    this_type& measure(Args&&...);
    template <typename... Args>
    this_type& sample(Args&&...);
    template <typename... Args>
    this_type& start(Args&&...);
    template <typename... Args>
    this_type& stop(Args&&...);
    template <typename... Args>
    this_type& record(Args&&...);
    template <typename... Args>
    this_type& reset(Args&&...);
    template <typename... Args>
    auto get(Args&&...) const;
    template <typename... Args>
    auto             get_labeled(Args&&...) const;
    data_type&       data();
    const data_type& data() const;
    this_type&       set_scope(scope::config);

    using bundle_type::get_prefix;
    using bundle_type::get_scope;
    using bundle_type::get_store;
    using bundle_type::hash;
    using bundle_type::key;
    using bundle_type::laps;
    using bundle_type::prefix;
    using bundle_type::store;

    /// when chaining together operations, this function enables executing a function
    /// inside the chain
    template <typename FuncT, typename... Args>
    decltype(auto) execute(FuncT&& func, Args&&... args)
    {
        return mpl::execute(*this,
                            std::forward<FuncT>(func)(std::forward<Args>(args)...));
    }

    //----------------------------------------------------------------------------------//
    /// construct the objects that have constructors with matching arguments
    //
    template <typename... Args>
    this_type& construct(Args&&... _args)
    {
        using construct_t = operation_t<operation::construct>;
        apply_v::access<construct_t>(m_data, std::forward<Args>(_args)...);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    /// provide preliminary info to the objects with matching arguments. This is typically
    /// used to notify a component that it has been bundled alongside another component
    /// that it can extract data from.
    //
    this_type& assemble() { invoke::assemble(m_data, *this); }

    template <typename... Args, size_t N = sizeof...(Args), enable_if_t<N != 0, int> = 0>
    this_type& assemble(Args&&... _args)
    {
        invoke::assemble(m_data, std::forward<Args>(_args)...);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    /// provide conclusive info to the objects with matching arguments. This is typically
    /// used by components to extract data from another component it has been bundled
    /// alongside, e.g. the cpu_util component can extract data from \ref
    /// tim::component::wall_clock and \ref tim::component::cpu_clock
    //
    this_type& derive() { invoke::derive(m_data, *this); }

    template <typename... Args, size_t N = sizeof...(Args), enable_if_t<N != 0, int> = 0>
    this_type& derive(Args&&... _args)
    {
        invoke::derive(m_data, std::forward<Args>(_args)...);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    /// mark a beginning position in the execution (typically used by asynchronous
    /// structures)
    //
    template <typename... Args>
    this_type& mark_begin(Args&&... _args)
    {
        invoke::mark_begin(m_data, std::forward<Args>(_args)...);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    /// mark a beginning position in the execution (typically used by asynchronous
    /// structures)
    //
    template <typename... Args>
    this_type& mark_end(Args&&... _args)
    {
        invoke::mark_end(m_data, std::forward<Args>(_args)...);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    /// store a value
    //
    template <typename... Args>
    this_type& store(Args&&... _args)
    {
        invoke::store(m_data, std::forward<Args>(_args)...);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    /// allow the components to inspect the incoming arguments before start
    /// or out-going return value before returning (typically using in GOTCHA components)
    //
    template <typename... Args>
    this_type& audit(Args&&... _args)
    {
        invoke::audit(m_data, std::forward<Args>(_args)...);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    /// apply a user-defined operation to all the components
    /// \tparam OpT Operation struct
    //
    template <template <typename> class OpT, typename... Args>
    this_type& invoke(Args&&... _args)
    {
        invoke::invoke<OpT>(m_data, std::forward<Args>(_args)...);
        return *this;
    }

    //----------------------------------------------------------------------------------//
    /// generic member function for invoking user-provided operations on a specific
    /// set of component types
    /// \tparam OpT Operation struct
    //
    template <template <typename> class OpT, typename... Tp, typename... Args>
    this_type& invoke(mpl::piecewise_select<Tp...>, Args&&... _args)
    {
        TIMEMORY_FOLD_EXPRESSION(operation::generic_operator<Tp, OpT<Tp>, TIMEMORY_API>(
            this->get<Tp>(), std::forward<Args>(_args)...));
        return *this;
    }

    //----------------------------------------------------------------------------------//
    /// get member functions taking either a type
    //
    template <typename T, enable_if_t<is_one_of<T, data_type>::value, int> = 0>
    T* get()
    {
        return &(std::get<index_of<T, data_type>::value>(m_data));
    }

    template <typename T, enable_if_t<is_one_of<T, data_type>::value, int> = 0>
    const T* get() const
    {
        return &(std::get<index_of<T, data_type>::value>(m_data));
    }

    template <typename T, enable_if_t<!is_one_of<T, data_type>::value, int> = 0>
    T* get() const
    {
        void* ptr = nullptr;
        get(ptr, typeid_hash<T>());
        return static_cast<T*>(ptr);
    }

    this_type& get(void*& ptr, size_t _hash) const
    {
        using get_t = operation_t<operation::get>;
        apply_v::access<get_t>(m_data, ptr, _hash);
        return const_cast<this_type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    /// this is a simple alternative to get<T>() when used from SFINAE in operation
    /// namespace which has a struct get also templated. Usage there can cause error
    /// with older compilers
    template <typename U, typename T = std::remove_pointer_t<decay_t<U>>,
              enable_if_t<trait::is_available<T>::value && is_one_of<T, data_type>::value,
                          int> = 0>
    auto get_component()
    {
        return get<T>();
    }

    //----------------------------------------------------------------------------------//

    template <
        typename T, typename... Args,
        enable_if_t<!is_one_of<T, reference_type>::value && has_user_bundle_v, int> = 0>
    bool init(Args&&...)
    {
        using bundle_t = decltype(std::get<0>(std::declval<user_bundle_types>()));
        this->init<bundle_t>();
        this->get<bundle_t>()->insert(component::factory::get_opaque<T>(m_scope),
                                      component::factory::get_typeids<T>());
        return true;
    }

    //----------------------------------------------------------------------------------//

    template <
        typename T, typename... Args,
        enable_if_t<!is_one_of<T, reference_type>::value && !has_user_bundle_v, int> = 0>
    bool init(Args&&...)
    {
        return is_one_of<T, reference_type>::value;
    }

    //----------------------------------------------------------------------------------//
    ///  variadic initialization
    //
    template <typename... T, typename... Args>
    auto initialize(Args&&... args)
    {
        constexpr auto N = sizeof...(T);
        return TIMEMORY_FOLD_EXPANSION(bool, N,
                                       this->init<T>(std::forward<Args>(args)...));
    }

    //----------------------------------------------------------------------------------//
    /// apply a member function to a type that is in variadic list AND is available
    ///
    template <typename T, typename Func, typename... Args,
              enable_if_t<is_one_of<T, data_type>::value, int> = 0>
    this_type& type_apply(Func&& _func, Args&&... _args)
    {
        auto&& _obj = get<T>();
        ((_obj).*(_func))(std::forward<Args>(_args)...);
        return *this;
    }

    template <typename T, typename Func, typename... Args,
              enable_if_t<!is_one_of<T, data_type>::value, int> = 0>
    this_type& type_apply(Func&&, Args&&...)
    {
        return *this;
    }

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
    template <typename Op>
    this_type& operator-=(Op&& rhs)
    {
        using minus_t = operation_t<operation::minus>;
        apply_v::access<minus_t>(m_data, std::forward<Op>(rhs));
        return *this;
    }

    template <typename Op>
    this_type& operator+=(Op&& rhs)
    {
        using plus_t = operation_t<operation::plus>;
        apply_v::access<plus_t>(m_data, std::forward<Op>(rhs));
        return *this;
    }

    template <typename Op>
    this_type& operator*=(Op&& rhs)
    {
        using multiply_t = operation_t<operation::multiply>;
        apply_v::access<multiply_t>(m_data, std::forward<Op>(rhs));
        return *this;
    }

    template <typename Op>
    this_type& operator/=(Op&& rhs)
    {
        using divide_t = operation_t<operation::divide>;
        apply_v::access<divide_t>(m_data, std::forward<Op>(rhs));
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

    template <typename Op>
    friend this_type operator*(const this_type& lhs, Op&& rhs)
    {
        this_type tmp(lhs);
        return tmp *= std::forward<Op>(rhs);
    }

    template <typename Op>
    friend this_type operator/(const this_type& lhs, Op&& rhs)
    {
        this_type tmp(lhs);
        return tmp /= std::forward<Op>(rhs);
    }

    //----------------------------------------------------------------------------------//
    //
    template <bool PrintPrefix = true, bool PrintLaps = true>
    this_type& print(std::ostream& os, bool skip_wo_hash = true) const
    {
        using printer_t = typename bundle_type::print_type;
        if(size() == 0)
            return const_cast<this_type&>(*this);
        if(m_hash == 0 && skip_wo_hash)
            return const_cast<this_type&>(*this);
        std::stringstream ss_data;
        apply_v::access_with_indices<printer_t>(m_data, std::ref(ss_data), false);
        IF_CONSTEXPR(PrintPrefix)
        {
            bundle_type::update_width();
            auto _key = key();
            if(_key.length() > 0)
            {
                std::stringstream ss_prefix;
                std::stringstream ss_id;
                ss_id << get_prefix() << " " << std::left << _key;
                ss_prefix << std::setw(bundle_type::output_width()) << std::left
                          << ss_id.str() << " : ";
                os << ss_prefix.str();
            }
        }
        if(ss_data.str().length() > 0)
        {
            os << ss_data.str();
            if(m_laps > 0 && PrintLaps)
                os << " [laps: " << m_laps << "]";
        }
        return const_cast<this_type&>(*this);
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
        std::string _key   = {};
        auto        keyitr = get_hash_ids()->find(m_hash);
        if(keyitr != get_hash_ids()->end())
            _key = keyitr->second;

        ar(cereal::make_nvp("hash", m_hash), cereal::make_nvp("key", _key),
           cereal::make_nvp("laps", m_laps));

        if(keyitr == get_hash_ids()->end())
        {
            auto _hash = add_hash_id(_key);
            if(_hash != m_hash)
            {
                PRINT_HERE("Warning! Hash for '%s' (%llu) != %llu", _key.c_str(),
                           (unsigned long long) _hash, (unsigned long long) m_hash);
            }
        }

        ar.setNextName("data");
        ar.startNode();
        invoke::serialize(ar, m_data);
        ar.finishNode();
    }

public:
    int64_t     laps() const { return bundle_type::laps(); }
    std::string key() const { return bundle_type::key(); }
    uint64_t    hash() const { return bundle_type::hash(); }
    bool&       store() { return bundle_type::store(); }
    const bool& store() const { return bundle_type::store(); }
    auto        prefix() const { return bundle_type::prefix(); }
    auto        get_prefix() const { return bundle_type::get_prefix(); }

    TIMEMORY_INLINE void rekey(const string_t& _key);
    TIMEMORY_INLINE void rekey(const captured_location_t& _loc);
    TIMEMORY_INLINE void rekey(uint64_t _hash);

protected:
    // protected member functions
    data_type&       get_data();
    const data_type& get_data() const;
    void             set_prefix(const string_t&) const;
    void             set_prefix(size_t) const;

protected:
    // objects
    using bundle_type::m_config;
    using bundle_type::m_hash;
    using bundle_type::m_is_active;
    using bundle_type::m_is_pushed;
    using bundle_type::m_laps;
    using bundle_type::m_scope;
    using bundle_type::m_store;
    mutable data_type m_data = data_type{};
};

//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
void
lightweight_tuple<Types...>::rekey(const string_t& _key)
{
    m_hash = add_hash_id(_key);
    set_prefix(_key);
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
void
lightweight_tuple<Types...>::rekey(const captured_location_t& _loc)
{
    m_hash = _loc.get_hash();
    set_prefix(_loc.get_hash());
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
void
lightweight_tuple<Types...>::rekey(uint64_t _hash)
{
    m_hash = _hash;
    set_prefix(_hash);
}
//
//======================================================================================//

template <typename... Types>
auto
get(const lightweight_tuple<Types...>& _obj)
    -> decltype(std::declval<lightweight_tuple<Types...>>().get())
{
    return _obj.get();
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
auto
get_labeled(const lightweight_tuple<Types...>& _obj)
    -> decltype(std::declval<lightweight_tuple<Types...>>().get_labeled())
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
get(::tim::lightweight_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const ::tim::lightweight_tuple<Types...>& obj)
{
    return get<N>(obj.data());
}

//--------------------------------------------------------------------------------------//

template <std::size_t N, typename... Types>
auto
get(::tim::lightweight_tuple<Types...>&& obj)
    -> decltype(get<N>(std::forward<::tim::lightweight_tuple<Types...>>(obj).data()))
{
    using obj_type = ::tim::lightweight_tuple<Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}

//======================================================================================//
}  // namespace std

//--------------------------------------------------------------------------------------//
