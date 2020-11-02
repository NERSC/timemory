//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

/**
 * \file timemory/components/data_tracker/components.hpp
 * \brief Implementation of the data_tracker component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/data_tracker/types.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

//======================================================================================//
//
namespace tim
{
namespace component
{
/// \struct component::data_tracker
/// \brief This component is provided to facilitate data tracking. The first
/// template parameter is the type of data to be tracked, the second is a custom
/// tag, the third is the implementation for how to track the data.
/// Usage:
///
///         struct iteration_count_tag;
///
///         using tracker_type = data_tracker<uint64_t, iteration_count_tag>;
///         using tuple_t = tim::auto_tuple<wall_clock, data_tracker<uint64_t>>;
///
///         double err             = std::numeric_limits<double>::max();
///         const double tolerance = 1.0e-6;
///
///         tuple_t t("iteration_time");
///
///         while(err > tolerance)
///         {
///             t.store(std::plus<uint64_t>{}, 1);
///             // ... do something ...
///         }
///
template <typename InpT, typename Tag, typename Handler, typename StoreT>
struct data_tracker : public base<data_tracker<InpT, Tag, Handler, StoreT>, StoreT>
{
    using value_type      = StoreT;
    using this_type       = data_tracker<InpT, Tag, Handler, StoreT>;
    using base_type       = base<this_type, value_type>;
    using handler_type    = Handler;
    using secondary_map_t = std::unordered_map<std::string, this_type>;
    using secondary_ptr_t = std::shared_ptr<secondary_map_t>;
    using string_t        = std::string;
    using start_t =
        operation::generic_operator<this_type, operation::start<this_type>, Tag>;
    using stop_t =
        operation::generic_operator<this_type, operation::stop<this_type>, Tag>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;

    static std::string& label()
    {
        static std::string _instance = []() {
            std::stringstream ss;
            ss << demangle<Tag>() << "_" << demangle<InpT>();
            return ss.str();
        }();
        return _instance;
    }

    static std::string& description()
    {
        static std::string _instance = []() {
            std::stringstream ss;
            ss << "Data tracker for data of type " << demangle<InpT>() << " for "
               << demangle<Tag>();
            return ss.str();
        }();
        return _instance;
    }

    void start() {}
    void stop() {}

    template <typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    void store(const T& val)
    {
        handler_type::store(*this, val / get_unit());
    }

    template <typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    void store(handler_type&&, const T& val)
    {
        handler_type::store(*this, val / get_unit());
    }

    template <typename Func, typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    auto store(Func&& f, const T& val)
        -> decltype(std::declval<handler_type>().store(*this, std::forward<Func>(f), val),
                    void())
    {
        handler_type::store(*this, std::forward<Func>(f), val / get_unit());
    }

    template <typename Func, typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    auto store(handler_type&&, Func&& f, const T& val)
        -> decltype(std::declval<handler_type>().store(*this, std::forward<Func>(f), val),
                    void())
    {
        handler_type::store(*this, std::forward<Func>(f), val / get_unit());
    }

    template <typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    void mark_begin(const T& val)
    {
        handler_type::begin(*this, val / get_unit());
    }

    template <typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    void mark_end(const T& val)
    {
        handler_type::end(*this, val / get_unit());
    }

    template <typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    void mark_begin(handler_type&&, const T& val)
    {
        handler_type::begin(*this, val / get_unit());
    }

    template <typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    void mark_end(handler_type&&, const T& val)
    {
        handler_type::end(*this, val / get_unit());
    }

    auto get() const { return handler_type::get(*this); }
    auto get_display() const { return handler_type::get_display(*this); }

    void set_value(const value_type& v) { value = v; }

    template <typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    this_type* add_secondary(const string_t& _key, const T& val)
    {
        this_type _tmp;
        start_t   _start(_tmp);
        _tmp.store(val);
        stop_t _stop(_tmp);
        auto&  _map = *get_secondary_map();
        _map.insert({ _key, _tmp });
        return &(_map[_key]);
    }

    template <typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    this_type* add_secondary(const string_t& _key, handler_type&& h, const T& val)
    {
        this_type _tmp;
        start_t   _start(_tmp);
        _tmp.store(std::forward<handler_type>(h), val);
        stop_t _stop(_tmp);
        auto&  _map = *get_secondary_map();
        _map.insert({ _key, _tmp });
        return &(_map[_key]);
    }

    template <typename Func, typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    this_type* add_secondary(const string_t& _key, Func&& f, const T& val)
    {
        PRINT_HERE("%s :: adding secondary", demangle<this_type>().c_str());
        this_type _tmp;
        start_t   _start(_tmp);
        _tmp.store(std::forward<Func>(f), val);
        stop_t _stop(_tmp);
        auto&  _map = *get_secondary_map();
        _map.insert({ _key, _tmp });
        return &(_map[_key]);
    }

    template <typename Func, typename T,
              enable_if_t<concepts::is_acceptable_conversion<T, InpT>::value, int> = 0>
    this_type* add_secondary(const string_t& _key, handler_type&& h, Func&& f,
                             const T& val)
    {
        this_type _tmp;
        start_t   _start(_tmp);
        _tmp.store(std::forward<handler_type>(h), std::forward<Func>(f), val);
        stop_t _stop(_tmp);
        auto&  _map = *get_secondary_map();
        _map.insert({ _key, _tmp });
        return &(_map[_key]);
    }

    using base_type::get_unit;
    using base_type::load;
    using base_type::value;

    auto get_secondary_map()
    {
        if(!m_secondary)
            m_secondary = std::make_shared<secondary_map_t>();
        return m_secondary;
    }

    auto get_secondary() { return (m_secondary) ? *m_secondary : secondary_map_t{}; }

private:
    secondary_ptr_t m_secondary{ nullptr };
};
//
//--------------------------------------------------------------------------------------//
//
/// \typedef data_handler_t
/// \brief an alias for getting the handle_type of a data tracker
template <typename T>
using data_handler_t = typename T::handler_type;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
