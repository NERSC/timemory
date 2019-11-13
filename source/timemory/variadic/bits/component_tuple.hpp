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

/** \file bits/component_tuple.hpp
 * \headerfile bits/component_tuple.hpp "timemory/variadic/bits/component_tuple.hpp"
 * Implementation for various functions
 *
 */

#pragma once

#include "timemory/manager.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/variadic/component_tuple.hpp"

//======================================================================================//
//
//      tim::get functions
//
namespace tim
{
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline component_tuple<Types...>::component_tuple()
: m_store(false)
, m_flat(false)
, m_is_pushed(false)
, m_print_prefix(true)
, m_print_laps(true)
, m_laps(0)
, m_hash(0)
, m_key("")
{}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline component_tuple<Types...>::component_tuple(const string_t& key, const bool& store,
                                                  const bool& flat)
: m_store(store && settings::enabled())
, m_flat(flat)
, m_is_pushed(false)
, m_print_prefix(true)
, m_print_laps(true)
, m_laps(0)
, m_hash((settings::enabled()) ? add_hash_id(key) : 0)
, m_key(key)
, m_data(data_type{})
{
    compute_width(key);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline component_tuple<Types...>::component_tuple(const captured_location_t& loc,
                                                  const bool& store, const bool& flat)
: m_store(store && settings::enabled())
, m_flat(flat)
, m_is_pushed(false)
, m_print_prefix(true)
, m_print_laps(true)
, m_laps(0)
, m_hash(loc.get_hash())
, m_key(loc.get_id())
, m_data(data_type())
{
    compute_width(m_key);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline component_tuple<Types...>
component_tuple<Types...>::clone(bool store, bool flat)
{
    component_tuple tmp(*this);
    tmp.m_store = store;
    tmp.m_flat  = flat;
    return tmp;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
component_tuple<Types...>::~component_tuple()
{
    pop();
}

//----------------------------------------------------------------------------------//
// insert into graph
//
template <typename... Types>
inline void
component_tuple<Types...>::push()
{
    if(m_store && !m_is_pushed)
    {
        // reset the data
        apply<void>::access<reset_t>(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed = true;
        // insert node or find existing node
        if(m_flat)
            apply<void>::access<insert_node_t<scope::flat>>(m_data, m_hash);
        else
            apply<void>::access<insert_node_t<scope::process>>(m_data, m_hash);
    }
}

//----------------------------------------------------------------------------------//
// pop out of graph
//
template <typename... Types>
inline void
component_tuple<Types...>::pop()
{
    if(m_store && m_is_pushed)
    {
        // set the current node to the parent node
        apply<void>::access<pop_node_t>(m_data);
        // avoid pushing/popping when already pushed/popped
        m_is_pushed = false;
    }
}

//----------------------------------------------------------------------------------//
// measure functions
//
template <typename... Types>
inline void
component_tuple<Types...>::measure()
{
    apply<void>::access<measure_t>(m_data);
}

//----------------------------------------------------------------------------------//
// start/stop functions
//
template <typename... Types>
inline void
component_tuple<Types...>::start()
{
    push();
    // increment laps
    ++m_laps;
    // start components
    apply<void>::access<prior_start_t>(m_data);
    apply<void>::access<stand_start_t>(m_data);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline void
component_tuple<Types...>::stop()
{
    // stop components
    apply<void>::access<prior_stop_t>(m_data);
    apply<void>::access<stand_stop_t>(m_data);
    // pop them off the running stack
    pop();
}

//----------------------------------------------------------------------------------//
// recording
//
template <typename... Types>
inline typename component_tuple<Types...>::this_type&
component_tuple<Types...>::record()
{
    ++m_laps;
    apply<void>::access<record_t>(m_data);
    return *this;
}

//----------------------------------------------------------------------------------//
// reset data
//
template <typename... Types>
inline void
component_tuple<Types...>::reset()
{
    apply<void>::access<reset_t>(m_data);
    m_laps = 0;
}

//----------------------------------------------------------------------------------//
// get data
//
template <typename... Types>
inline typename component_tuple<Types...>::data_value_type
component_tuple<Types...>::get() const
{
    const_cast<this_type&>(*this).stop();
    data_value_type _ret_data;
    apply<void>::access2<get_data_t>(m_data, _ret_data);
    return _ret_data;
}

//----------------------------------------------------------------------------------//
// reset data
//
template <typename... Types>
inline typename component_tuple<Types...>::data_label_type
component_tuple<Types...>::get_labeled() const
{
    const_cast<this_type&>(*this).stop();
    data_label_type _ret_data;
    apply<void>::access2<get_data_t>(m_data, _ret_data);
    return _ret_data;
}

//----------------------------------------------------------------------------------//
// this_type operators
//
template <typename... Types>
inline typename component_tuple<Types...>::this_type&
component_tuple<Types...>::operator-=(const this_type& rhs)
{
    apply<void>::access2<minus_t>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline typename component_tuple<Types...>::this_type&
component_tuple<Types...>::operator-=(this_type& rhs)
{
    apply<void>::access2<minus_t>(m_data, rhs.m_data);
    m_laps -= rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline typename component_tuple<Types...>::this_type&
component_tuple<Types...>::operator+=(const this_type& rhs)
{
    apply<void>::access2<plus_t>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline typename component_tuple<Types...>::this_type&
component_tuple<Types...>::operator+=(this_type& rhs)
{
    apply<void>::access2<plus_t>(m_data, rhs.m_data);
    m_laps += rhs.m_laps;
    return *this;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline void
component_tuple<Types...>::print_storage()
{
    apply<void>::type_access<operation::print_storage, data_type>();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline typename component_tuple<Types...>::data_type&
component_tuple<Types...>::data()
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline const typename component_tuple<Types...>::data_type&
component_tuple<Types...>::data() const
{
    return m_data;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline int64_t
component_tuple<Types...>::laps() const
{
    return m_laps;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline uint64_t&
component_tuple<Types...>::hash()
{
    return m_hash;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline std::string&
component_tuple<Types...>::key()
{
    return m_key;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline const uint64_t&
component_tuple<Types...>::hash() const
{
    return m_hash;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline const std::string&
component_tuple<Types...>::key() const
{
    return m_key;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline void
component_tuple<Types...>::rekey(const string_t& _key)
{
    compute_width(m_key = _key);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline bool&
component_tuple<Types...>::store()
{
    return m_store;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline const bool&
component_tuple<Types...>::store() const
{
    return m_store;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline std::string
component_tuple<Types...>::get_prefix() const
{
    auto _get_prefix = []() {
        if(!mpi::is_initialized())
            return string_t(">>> ");

        // prefix spacing
        static uint16_t width = 1;
        if(mpi::size() > 9)
            width = std::max(width, (uint16_t)(log10(mpi::size()) + 1));
        std::stringstream ss;
        ss.fill('0');
        ss << "|" << std::setw(width) << mpi::rank() << ">>> ";
        return ss.str();
    };
    static string_t _prefix = _get_prefix();
    return _prefix;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline void
component_tuple<Types...>::compute_width(const string_t& key)
{
    static string_t _prefix = get_prefix();
    output_width(key.length() + _prefix.length() + 1);
    set_object_prefix(key);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline void
component_tuple<Types...>::update_width() const
{
    const_cast<this_type&>(*this).compute_width(m_key);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline int64_t
component_tuple<Types...>::output_width(int64_t width)
{
    static std::atomic<int64_t> _instance;
    if(width > 0)
    {
        auto current_width = _instance.load(std::memory_order_relaxed);
        auto compute       = [&]() {
            current_width = _instance.load(std::memory_order_relaxed);
            return std::max(_instance.load(), width);
        };
        int64_t propose_width = compute();
        do
        {
            if(propose_width > current_width)
            {
                auto ret = _instance.compare_exchange_strong(current_width, propose_width,
                                                             std::memory_order_relaxed);
                if(!ret)
                    compute();
            }
        } while(propose_width > current_width);
    }
    return _instance.load();
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline void
component_tuple<Types...>::set_object_prefix(const string_t& key)
{
    apply<void>::access<set_prefix_t>(m_data, key);
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
inline void
component_tuple<Types...>::init_storage()
{
    apply<void>::type_access<operation::init_storage, data_type>();
}

//--------------------------------------------------------------------------------------//

}  // namespace tim
