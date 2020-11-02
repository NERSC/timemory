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

/** \file timemory/variadic/component_tuple.cpp
 * \brief Implementation for various component_tuple member functions
 *
 */

#include "timemory/variadic/component_tuple.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types/set.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/functional.hpp"
#include "timemory/variadic/types.hpp"

//======================================================================================//
//
//      tim::get functions
//
namespace tim
{
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_tuple<Types...>::component_tuple()
{
    if(settings::enabled())
        init_storage();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename... T, typename Func>
component_tuple<Types...>::component_tuple(const string_t&     key,
                                           quirk::config<T...> config,
                                           const Func&         init_func)
: bundle_type(((settings::enabled()) ? add_hash_id(key) : 0), quirk::config<T...>{})
, m_data(invoke::construct<data_type>(key, config))
{
    if(settings::enabled())
    {
        IF_CONSTEXPR(!quirk_config<quirk::no_store, T...>::value) { init_storage(); }
        IF_CONSTEXPR(!quirk_config<quirk::no_init, T...>::value) { init_func(*this); }
        set_prefix(get_hash_ids()->find(m_hash)->second);
        invoke::set_scope(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start, T...>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename... T, typename Func>
component_tuple<Types...>::component_tuple(const captured_location_t& loc,
                                           quirk::config<T...>        config,
                                           const Func&                init_func)
: bundle_type(loc.get_hash(), quirk::config<T...>{})
, m_data(invoke::construct<data_type>(loc, config))
{
    if(settings::enabled())
    {
        IF_CONSTEXPR(!quirk_config<quirk::no_store, T...>::value) { init_storage(); }
        IF_CONSTEXPR(!quirk_config<quirk::no_init, T...>::value) { init_func(*this); }
        set_prefix(loc.get_id());
        invoke::set_scope(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start, T...>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename Func>
component_tuple<Types...>::component_tuple(const string_t& key, const bool& store,
                                           scope::config _scope, const Func& init_func)
: bundle_type((settings::enabled()) ? add_hash_id(key) : 0, store, _scope)
, m_data(invoke::construct<data_type>(key, store, _scope))
{
    if(settings::enabled())
    {
        if(store)
        {
            init_storage();
        }
        IF_CONSTEXPR(!quirk_config<quirk::no_init>::value) { init_func(*this); }
        set_prefix(get_hash_ids()->find(m_hash)->second);
        invoke::set_scope(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename Func>
component_tuple<Types...>::component_tuple(const captured_location_t& loc,
                                           const bool& store, scope::config _scope,
                                           const Func& init_func)
: bundle_type(loc.get_hash(), store, _scope)
, m_data(invoke::construct<data_type>(loc, store, _scope))
{
    if(settings::enabled())
    {
        if(store)
        {
            init_storage();
        }
        IF_CONSTEXPR(!quirk_config<quirk::no_init>::value) { init_func(*this); }
        set_prefix(loc.get_hash());
        invoke::set_scope(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename Func>
component_tuple<Types...>::component_tuple(size_t hash, const bool& store,
                                           scope::config _scope, const Func& init_func)
: bundle_type(hash, store, _scope)
, m_data(invoke::construct<data_type>(hash, store, _scope))
{
    if(settings::enabled())
    {
        if(store)
        {
            init_storage();
        }
        IF_CONSTEXPR(!quirk_config<quirk::no_init>::value) { init_func(*this); }
        set_prefix(hash);
        invoke::set_scope(m_data, m_scope);
        IF_CONSTEXPR(quirk_config<quirk::auto_start>::value) { start(); }
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_tuple<Types...>::~component_tuple()
{
    if(m_is_active())
        stop();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_tuple<Types...>
component_tuple<Types...>::clone(bool _store, scope::config _scope)
{
    component_tuple tmp(*this);
    tmp.m_store(_store);
    tmp.m_scope = _scope;
    return tmp;
}

//--------------------------------------------------------------------------------------//
// insert into graph
//
template <typename... Types>
void
component_tuple<Types...>::push()
{
    if(!m_is_pushed())
    {
        // reset the data
        invoke::reset(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(true);
        // insert node or find existing node
        invoke::push(m_data, m_scope, m_hash);
    }
}

//--------------------------------------------------------------------------------------//
// pop out of graph
//
template <typename... Types>
void
component_tuple<Types...>::pop()
{
    if(m_is_pushed())
    {
        // set the current node to the parent node
        invoke::pop(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed(false);
    }
}

//--------------------------------------------------------------------------------------//
// measure functions
//
template <typename... Types>
template <typename... Args>
void
component_tuple<Types...>::measure(Args&&... args)
{
    invoke::measure(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// sample functions
//
template <typename... Types>
template <typename... Args>
void
component_tuple<Types...>::sample(Args&&... args)
{
    sample_type _samples;
    apply_v::access2<operation_t<operation::sample>>(m_data, _samples,
                                                     std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// start/stop functions with no push/pop or assemble/derive
//
template <typename... Types>
template <typename... Args>
void
component_tuple<Types...>::start(mpl::lightweight, Args&&... args)
{
    assemble(*this);
    invoke::start(m_data, std::forward<Args>(args)...);
    m_is_active(true);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename... Args>
void
component_tuple<Types...>::stop(mpl::lightweight, Args&&... args)
{
    invoke::stop(m_data, std::forward<Args>(args)...);
    ++m_laps;
    derive(*this);
    m_is_active(false);
}

//--------------------------------------------------------------------------------------//
// start/stop functions
//
template <typename... Types>
template <typename... Args>
void
component_tuple<Types...>::start(Args&&... args)
{
    // push components into the call-stack
    if(m_store())
        push();

    // start components
    start(mpl::lightweight{}, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
template <typename... Args>
void
component_tuple<Types...>::stop(Args&&... args)
{
    // stop components
    stop(mpl::lightweight{}, std::forward<Args>(args)...);

    // pop components off of the call-stack stack
    if(m_store())
        pop();
}

//--------------------------------------------------------------------------------------//
// recording
//
template <typename... Types>
template <typename... Args>
component_tuple<Types...>&
component_tuple<Types...>::record(Args&&... args)
{
    ++m_laps;
    invoke::record(m_data, std::forward<Args>(args)...);
    return *this;
}

//--------------------------------------------------------------------------------------//
// reset data
//
template <typename... Types>
template <typename... Args>
void
component_tuple<Types...>::reset(Args&&... args)
{
    invoke::reset(m_data, std::forward<Args>(args)...);
    m_laps = 0;
}

//--------------------------------------------------------------------------------------//
// get data
//
template <typename... Types>
template <typename... Args>
auto
component_tuple<Types...>::get(Args&&... args) const
{
    return invoke::get(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// reset data
//
template <typename... Types>
template <typename... Args>
auto
component_tuple<Types...>::get_labeled(Args&&... args) const
{
    return invoke::get_labeled(m_data, std::forward<Args>(args)...);
}

//--------------------------------------------------------------------------------------//
// this_type operators
//
template <typename... Types>
component_tuple<Types...>&
component_tuple<Types...>::operator-=(const this_type& rhs)
{
    apply_v::access2<operation_t<operation::minus>>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_tuple<Types...>&
component_tuple<Types...>::operator-=(this_type& rhs)
{
    apply_v::access2<operation_t<operation::minus>>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_tuple<Types...>&
component_tuple<Types...>::operator+=(const this_type& rhs)
{
    apply_v::access2<operation_t<operation::plus>>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_tuple<Types...>&
component_tuple<Types...>::operator+=(this_type& rhs)
{
    apply_v::access2<operation_t<operation::plus>>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_tuple<Types...>::print_storage()
{
    apply_v::type_access<operation::print_storage, data_type>();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
typename component_tuple<Types...>::data_type&
component_tuple<Types...>::data()
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
const typename component_tuple<Types...>::data_type&
component_tuple<Types...>::data() const
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_tuple<Types...>::set_prefix(const string_t& _key) const
{
    invoke::set_prefix(m_data, m_hash, _key);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_tuple<Types...>::set_prefix(size_t _hash) const
{
    auto itr = get_hash_ids()->find(_hash);
    if(itr != get_hash_ids()->end())
        invoke::set_prefix(m_data, _hash, itr->second);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_tuple<Types...>::set_scope(scope::config val)
{
    m_scope = val;
    invoke::set_scope(m_data, m_scope);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
void
component_tuple<Types...>::init_storage()
{
    static thread_local bool _once = []() {
        apply_v::type_access<operation::init_storage, data_type>();
        return true;
    }();
    consume_parameters(_once);
}

//--------------------------------------------------------------------------------------//

}  // namespace tim
