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

/** \file operations.hpp
 * \headerfile operations.hpp "timemory/mpl/operations.hpp"
 * These are structs and functions that provide the operations on the
 * components
 *
 */

#pragma once

#include "timemory/components.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/utility/serializer.hpp"

#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace operation
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct init_storage
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using string_t   = std::string;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::is_available<_Up>::value), char> = 0>
    init_storage()
    {
        using storage_type    = storage<Type>;
        static auto _instance = storage_type::instance();
        consume_parameters(_instance);
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::is_available<_Up>::value == false), char> = 0>
    init_storage()
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct live_count
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using string_t   = std::string;

    live_count(base_type& obj, int64_t& _counter) { _counter = obj.m_count; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct set_prefix
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using string_t   = std::string;

    set_prefix(base_type& obj, const bool& exists, const string_t& _prefix)
    {
        if(!exists)
            obj.set_prefix(_prefix);
    }

    template <typename _Up                                           = _Tp,
              enable_if_t<(trait::requires_prefix<_Up>::value), int> = 0>
    set_prefix(Type& obj, const string_t& _prefix)
    {
        obj.prefix = _prefix;
    }

    template <typename _Up                                                    = _Tp,
              enable_if_t<(trait::requires_prefix<_Up>::value == false), int> = 0>
    set_prefix(Type&, const string_t&)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct insert_node
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    insert_node(std::size_t _N, std::size_t, base_type& obj, bool* exists,
                const int64_t& id)
    {
        obj.insert_node(exists[_N], id);
    }

    insert_node(std::size_t _N, std::size_t, base_type& obj, bool* exists,
                const int64_t& id, const string_t& _prefix)
    {
        obj.insert_node(exists[_N], id);
        if(!exists[_N])
            obj.set_prefix(_prefix);
    }

    insert_node(base_type& obj, const string_t& _prefix, const int64_t& id)
    {
        obj.insert_node(_prefix, id);
    }

    insert_node(Type& obj, const string_t& _prefix, const int64_t& id)
    {
        obj.insert_node(_prefix, id);
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct pop_node
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit pop_node(base_type& obj) { obj.pop_node(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct record
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit record(base_type& obj) { obj.value = Type::record(); }

    template <typename _Up = _Tp, enable_if_t<(trait::record_max<_Up>::value), int> = 0>
    record(base_type& obj, const base_type& rhs)
    {
        obj = std::max(obj, rhs);
    }

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::record_max<_Up>::value == false), int> = 0>
    record(base_type& obj, const base_type& rhs)
    {
        obj += rhs;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct reset
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit reset(base_type& obj) { obj.reset(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct measure
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit measure(base_type& obj) { obj.measure(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit start(base_type& obj) { obj.start(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct priority_start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                          = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value), int> = 0>
    explicit priority_start(base_type& obj)
    {
        obj.start();
    }

    template <typename _Up                                                   = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    explicit priority_start(base_type&)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct standard_start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                          = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value), int> = 0>
    explicit standard_start(base_type&)
    {
    }

    template <typename _Up                                                   = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    explicit standard_start(base_type& obj)
    {
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit stop(base_type& obj)
    {
        obj.stop();
        obj.activate_noop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct priority_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value), int> = 0>
    explicit priority_stop(base_type& obj)
    {
        obj.stop();
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == false), int> = 0>
    explicit priority_stop(base_type&)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct standard_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value), int> = 0>
    explicit standard_stop(base_type&)
    {
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == false), int> = 0>
    explicit standard_stop(base_type& obj)
    {
        obj.stop();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit conditional_start(base_type& obj) { obj.conditional_start(); }

    template <typename _Func>
    conditional_start(base_type& obj, _Func&& func)
    {
        std::forward<_Func>(func)(obj.conditional_start());
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_priority_start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                          = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value), int> = 0>
    explicit conditional_priority_start(base_type& obj)
    {
        obj.conditional_start();
    }

    template <typename _Func, typename _Up = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value), int> = 0>
    conditional_priority_start(base_type& obj, _Func&& func)
    {
        std::forward<_Func>(func)(obj.conditional_start());
    }

    template <typename _Up                                                   = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    explicit conditional_priority_start(base_type&)
    {
    }

    template <typename _Func, typename _Up = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    conditional_priority_start(base_type&, _Func&&)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_standard_start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                          = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value), int> = 0>
    explicit conditional_standard_start(base_type&)
    {
    }

    template <typename _Func, typename _Up = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value), int> = 0>
    conditional_standard_start(base_type&, _Func&&)
    {
    }

    template <typename _Up                                                   = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    explicit conditional_standard_start(base_type& obj)
    {
        obj.conditional_start();
    }

    template <typename _Func, typename _Up = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    conditional_standard_start(base_type& obj, _Func&& func)
    {
        std::forward<_Func>(func)(obj.conditional_start());
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit conditional_stop(base_type& obj) { obj.conditional_stop(); }

    template <typename _Func>
    conditional_stop(base_type& obj, _Func&& func)
    {
        std::forward<_Func>(func)(obj.conditional_stop());
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_priority_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value), int> = 0>
    explicit conditional_priority_stop(base_type& obj)
    {
        obj.conditional_stop();
    }

    template <typename _Func, typename _Up = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value), int> = 0>
    conditional_priority_stop(base_type& obj, _Func&& func)
    {
        std::forward<_Func>(func)(obj.conditional_stop());
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == false), int> = 0>
    explicit conditional_priority_stop(base_type&)
    {
    }

    template <typename _Func, typename _Up = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == false), int> = 0>
    conditional_priority_stop(base_type&, _Func&&)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_standard_stop
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value), int> = 0>
    explicit conditional_standard_stop(base_type&)
    {
    }

    template <typename _Func, typename _Up = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value), int> = 0>
    conditional_standard_stop(base_type&, _Func&&)
    {
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == false), int> = 0>
    explicit conditional_standard_stop(base_type& obj)
    {
        obj.conditional_stop();
    }

    template <typename _Func, typename _Up = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == false), int> = 0>
    conditional_standard_stop(base_type& obj, _Func&& func)
    {
        std::forward<_Func>(func)(obj.conditional_stop());
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct mark_begin
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                                       = _Tp,
              enable_if_t<(trait::supports_args<_Up, std::tuple<>>::value), int> = 0>
    explicit mark_begin(Type& obj)
    {
        obj.mark_begin();
    }

    template <typename _Up                                                        = _Tp,
              enable_if_t<!(trait::supports_args<_Up, std::tuple<>>::value), int> = 0>
    explicit mark_begin(Type&)
    {
    }

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                     = 0,
              enable_if_t<(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_begin(Type& obj, _Args&&... _args)
    {
        obj.mark_begin(std::forward<_Args>(_args)...);
    }

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                      = 0,
              enable_if_t<!(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_begin(Type&, _Args&&...)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct mark_end
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                                       = _Tp,
              enable_if_t<(trait::supports_args<_Up, std::tuple<>>::value), int> = 0>
    explicit mark_end(Type& obj)
    {
        obj.mark_end();
    }

    template <typename _Up                                                        = _Tp,
              enable_if_t<!(trait::supports_args<_Up, std::tuple<>>::value), int> = 0>
    explicit mark_end(Type&)
    {
    }

    // mark_end(Type& obj) { obj.mark_end(); }

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                     = 0,
              enable_if_t<(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_end(Type& obj, _Args&&... _args)
    {
        obj.mark_end(std::forward<_Args>(_args)...);
    }

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                      = 0,
              enable_if_t<!(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_end(Type&, _Args&&...)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename RetType, typename LhsType, typename RhsType>
struct compose
{
    using ret_value_type = typename RetType::value_type;
    using lhs_value_type = typename LhsType::value_type;
    using rhs_value_type = typename RhsType::value_type;

    using ret_base_type = typename RetType::base_type;
    using lhs_base_type = typename LhsType::base_type;
    using rhs_base_type = typename RhsType::base_type;

    static_assert(std::is_same<ret_value_type, lhs_value_type>::value,
                  "Value types of RetType and LhsType are different!");

    static_assert(std::is_same<lhs_value_type, rhs_value_type>::value,
                  "Value types of LhsType and RhsType are different!");

    static RetType generate(const lhs_base_type& _lhs, const rhs_base_type& _rhs)
    {
        RetType _ret;
        _ret.is_running   = false;
        _ret.is_on_stack  = false;
        _ret.is_transient = (_lhs.is_transient && _rhs.is_transient);
        _ret.laps         = std::min(_lhs.laps, _rhs.laps);
        _ret.value        = (_lhs.value + _rhs.value);
        _ret.accum        = (_lhs.accum + _rhs.accum);
        return _ret;
    }

    template <typename _Func, typename... _Args>
    static RetType generate(const lhs_base_type& _lhs, const rhs_base_type& _rhs,
                            const _Func& _func, _Args&&... _args)
    {
        RetType _ret(std::forward<_Args>(_args)...);
        _ret.is_running   = false;
        _ret.is_on_stack  = false;
        _ret.is_transient = (_lhs.is_transient && _rhs.is_transient);
        _ret.laps         = std::min(_lhs.laps, _rhs.laps);
        _ret.value        = _func(_lhs.value, _rhs.value);
        _ret.accum        = _func(_lhs.accum, _rhs.accum);
        return _ret;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct plus
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, enable_if_t<(trait::record_max<_Up>::value), int> = 0>
    plus(base_type& obj, const base_type& rhs)
    {
        obj = std::max(obj, rhs);
    }

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::record_max<_Up>::value == false), int> = 0>
    plus(base_type& obj, const base_type& rhs)
    {
        obj += rhs;
    }

    plus(base_type& obj, const int64_t& rhs) { obj += rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct minus
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    minus(base_type& obj, const int64_t& rhs) { obj -= rhs; }
    minus(base_type& obj, const base_type& rhs) { obj -= rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct multiply
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    multiply(base_type& obj, const int64_t& rhs) { obj *= rhs; }
    multiply(base_type& obj, const base_type& rhs) { obj *= rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct divide
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    divide(base_type& obj, const int64_t& rhs) { obj /= rhs; }
    divide(base_type& obj, const base_type& rhs) { obj /= rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct get_data
{
    using Type            = _Tp;
    using DataType        = decltype(std::declval<Type>().get());
    using LabeledDataType = std::tuple<std::string, decltype(std::declval<Type>().get())>;

    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    // shorthand for available, non-void, using internal output handling
    //
    template <typename _Up>
    struct is_enabled
    {
        using _Vp                   = typename _Up::value_type;
        static constexpr bool value = (trait::is_available<_Up>::value &&
                                       !(trait::external_output_handling<_Up>::value) &&
                                       !(std::is_same<_Vp, void>::value));
    };

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    get_data(Type& _obj, DataType& _dst)
    {
        _dst = _obj.get();
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    get_data(Type&, DataType&)
    {
    }

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    get_data(Type& _obj, LabeledDataType& _dst)
    {
        _dst = LabeledDataType(Type::label(), _obj.get());
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    get_data(Type&, LabeledDataType&)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct print
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using widths_t   = std::vector<int64_t>;

    //----------------------------------------------------------------------------------//
    // shorthand for available, non-void, using internal output handling
    //
    template <typename _Up>
    struct is_enabled
    {
        using _Vp                   = typename _Up::value_type;
        static constexpr bool value = (trait::is_available<_Up>::value &&
                                       !(trait::external_output_handling<_Up>::value) &&
                                       !(std::is_same<_Vp, void>::value));
    };

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    print(const Type& _obj, std::ostream& _os, bool _endline = false)
    {
        std::stringstream ss;
        ss << _obj;
        if(_endline)
            ss << std::endl;
        _os << ss.str();
    }

    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    print(std::size_t _N, std::size_t _Ntot, const Type& _obj, std::ostream& _os,
          bool _endline)
    {
        std::stringstream ss;
        ss << _obj;
        if(_N + 1 < _Ntot)
            ss << ", ";
        else if(_N + 1 == _Ntot && _endline)
            ss << std::endl;
        _os << ss.str();
    }

    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    print(const Type& _obj, std::ostream& _os, const string_t& _prefix, int64_t _laps,
          int64_t _depth, const widths_t& _output_widths, bool _endline,
          const string_t& _suffix = "")
    {
        std::stringstream ss_prefix;
        std::stringstream ss;
        ss_prefix << std::setw(_output_widths.at(0)) << std::left << _prefix << " : ";
        ss << ss_prefix.str() << _obj;
        if(_laps > 0 && !trait::custom_laps_printing<Type>::value)
            ss << ", " << std::setw(_output_widths.at(1)) << _laps << " laps";
        if(_endline)
        {
            ss << ", depth " << std::setw(_output_widths.at(2)) << _depth;
            if(_suffix.length() > 0)
                ss << " " << _suffix;
            ss << std::endl;
        }
        _os << ss.str();
    }

    //----------------------------------------------------------------------------------//
    // only if components are available -- pointers
    //
    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    print(const Type* _obj, std::ostream& _os, bool _endline = false)
    {
        if(_obj)
            print(*_obj, _os, _endline);
    }

    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    print(std::size_t _N, std::size_t _Ntot, const Type* _obj, std::ostream& _os,
          bool _endline)
    {
        if(_obj)
            print(_N, _Ntot, *_obj, _os, _endline);
    }

    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    print(const Type* _obj, std::ostream& _os, const string_t& _prefix, int64_t _laps,
          int64_t _depth, const widths_t& _output_widths, bool _endline,
          const string_t& _suffix = "")
    {
        if(_obj)
            print(*_obj, _os, _prefix, _laps, _depth, _output_widths, _endline, _suffix);
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(const Type&, std::ostream&, bool = false)
    {
    }

    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(std::size_t, std::size_t, const Type&, std::ostream&, bool)
    {
    }

    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(const Type&, std::ostream&, const string_t&, int64_t, int64_t, const widths_t&,
          bool, const string_t& = "")
    {
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available -- pointers
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(const Type*, std::ostream&, bool = false)
    {
    }

    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(std::size_t, std::size_t, const Type*, std::ostream&, bool)
    {
    }

    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(const Type*, std::ostream&, const string_t&, int64_t, int64_t, const widths_t&,
          bool, const string_t& = "")
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct print_storage
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    // shorthand for available, non-void, using internal output handling
    //
    template <typename _Up>
    struct is_enabled
    {
        using _Vp                   = typename _Up::value_type;
        static constexpr bool value = (trait::is_available<_Up>::value &&
                                       !(trait::external_output_handling<_Up>::value) &&
                                       !(std::is_same<_Vp, void>::value));
    };

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    print_storage()
    {
        auto _storage = tim::storage<_Tp>::noninit_instance();
        if(_storage)
            _storage->print();
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print_storage()
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Archive>
struct serialization
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    // shorthand for available, non-void, using internal output handling
    //
    template <typename _Up>
    struct is_enabled
    {
        using _Vp                   = typename _Up::value_type;
        static constexpr bool value = (trait::is_available<_Up>::value &&
                                       !(trait::external_output_handling<_Up>::value) &&
                                       !(std::is_same<_Vp, void>::value));
    };

    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    serialization(base_type& obj, _Archive& ar, const unsigned int version)
    {
        auto _disp = static_cast<const Type&>(obj).get_display();
        auto _data = static_cast<const Type&>(obj).get();
        ar(serializer::make_nvp("is_transient", obj.is_transient),
           serializer::make_nvp("laps", obj.laps),
           serializer::make_nvp("repr_data", _data),
           serializer::make_nvp("value", obj.value),
           serializer::make_nvp("accum", obj.accum),
           serializer::make_nvp("display", _disp),
           serializer::make_nvp("unit.value", Type::unit()),
           serializer::make_nvp("unit.repr", Type::display_unit()));
        consume_parameters(version);
    }

    template <typename _Up = _Tp, enable_if_t<!(is_enabled<_Up>::value), char> = 0>
    serialization(base_type&, _Archive&, const unsigned int)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct echo_measurement
{
    using Type           = _Tp;
    using value_type     = typename Type::value_type;
    using base_type      = typename Type::base_type;
    using attributes_t   = std::map<string_t, string_t>;
    using strset_t       = std::set<string_t>;
    using stringstream_t = std::stringstream;
    using strvec_t       = std::vector<string_t>;

    //----------------------------------------------------------------------------------//
    // shorthand for available, non-void, using internal output handling
    //
    template <typename _Up>
    struct is_enabled
    {
        using _Vp                   = typename _Up::value_type;
        static constexpr bool value = (trait::is_available<_Up>::value &&
                                       !(trait::external_output_handling<_Up>::value) &&
                                       !(std::is_same<_Vp, void>::value));
    };

    //----------------------------------------------------------------------------------//
    /// generate an attribute
    ///
    static string_t attribute_string(const string_t& key, const string_t& item)
    {
        return apply<string_t>::join("", key, "=", "\"", item, "\"");
    };

    //----------------------------------------------------------------------------------//
    /// replace matching values in item with str
    ///
    static string_t replace(string_t& item, const string_t& str, const strset_t& values)
    {
        for(const auto& itr : values)
        {
            while(item.find(itr) != string_t::npos)
                item = item.replace(item.find(itr), itr.length(), str);
        }
        return item;
    };

    //----------------------------------------------------------------------------------//
    /// convert to lowercase
    ///
    static string_t lowercase(string_t _str)
    {
        for(auto& itr : _str)
            itr = tolower(itr);
        return _str;
    };

    //----------------------------------------------------------------------------------//
    /// convert to uppercase
    ///
    static string_t uppercase(string_t _str)
    {
        for(auto& itr : _str)
            itr = toupper(itr);
        return _str;
    };

    //----------------------------------------------------------------------------------//
    /// check if str contains any of the string items
    ///
    static bool contains(const string_t& str, const strset_t& items)
    {
        for(const auto& itr : items)
        {
            if(lowercase(str).find(itr) != string_t::npos)
                return true;
        }
        return false;
    };

    //----------------------------------------------------------------------------------//
    /// shorthand for apply<string_t>::join(...)
    ///
    template <typename... _Args>
    static string_t join(const std::string& _delim, _Args&&... _args)
    {
        return apply<string_t>::join(_delim, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    /// generate a name attribute
    ///
    template <typename... _Args>
    static string_t generate_name(const string_t& _prefix, string_t _unit, _Args&&... _args)
    {
        auto _extra    = join("_", std::forward<_Args>(_args)...);
        auto _label    = join("", "[", uppercase(Type::label()), "]");
        _unit          = replace(_unit, "", { " " });
        string_t _name = (_extra.length() > 0) ? join("_", _extra, _prefix)
                                               : join("_", _prefix);
        auto _ret = join(" ", _label, _name);
        _ret      = replace(_ret, "_", { "__" });
        if(_ret.length() > 0 && _ret.at(_ret.length() - 1) == '_')
            _ret.erase(_ret.length() - 1);
        _ret += " (" + _unit + ")";
        return _ret;
    }

    //----------------------------------------------------------------------------------//
    /// generate a measurement tag
    ///
    template <typename _Vt>
    static void generate_measurement(std::ostream& os, const attributes_t& attributes,
                                     const _Vt& value)
    {
        os << "<DartMeasurement";
        for(const auto& itr : attributes)
            os << " " << attribute_string(itr.first, itr.second);
        os << ">" << value << "</DartMeasurement>\n";
    }

    //----------------------------------------------------------------------------------//
    /// generate the prefix
    ///
    static string_t generate_prefix(const strvec_t& hierarchy)
    {
        string_t ret_prefix = "";
        string_t add_prefix = "";
        for(const auto& itr : hierarchy)
        {
            auto prefix = itr;
            prefix      = replace(prefix, "[c]", { "[_c_]" });
            prefix      = replace(prefix, "", { "> [" });
            prefix      = replace(prefix, "", { "|_" });
            prefix      = replace(prefix, "_",
                             { "[", "]", "(", ")", ".", "/", "\\", " ", "\t", "<", ">" });
            prefix      = replace(prefix, "_", { "__" });
            ret_prefix += add_prefix + prefix;
            add_prefix = " [>>] ";
        }
        return ret_prefix;
    }

    //----------------------------------------------------------------------------------//
    /// assumes type is not a iterable
    ///
    template <typename _Up = _Tp, typename _Vt = value_type,
              enable_if_t<(is_enabled<_Up>::value), char>                      = 0,
              enable_if_t<!(tim::trait::array_serialization<_Up>::value), int> = 0>
    echo_measurement(_Up& obj, const strvec_t& hierarchy)
    {
        auto prefix = generate_prefix(hierarchy);
        auto _unit  = Type::display_unit();
        auto name   = generate_name(prefix, _unit);
        auto _data  = obj.get();

        attributes_t   attributes = { { "name", name }, { "type", "numeric/double" } };
        stringstream_t ss;
        generate_measurement(ss, attributes, _data);
        std::cout << ss.str() << std::flush;
    }

    //----------------------------------------------------------------------------------//
    /// assumes type is iterable
    ///
    template <typename _Up = _Tp, typename _Vt = value_type,
              enable_if_t<(is_enabled<_Up>::value), char>                     = 0,
              enable_if_t<(tim::trait::array_serialization<_Up>::value), int> = 0>
    echo_measurement(_Up& obj, const strvec_t& hierarchy)
    {
        auto prefix = generate_prefix(hierarchy);
        auto _data  = obj.get();

        attributes_t   attributes = { { "type", "numeric/double" } };
        stringstream_t ss;

        uint64_t idx     = 0;
        auto     _labels = obj.label_array();
        auto     _dunits = obj.display_unit_array();
        for(auto& itr : _data)
        {
            string_t _extra = (idx < _labels.size()) ? _labels.at(idx) : "";
            string_t _dunit = (idx < _labels.size()) ? _dunits.at(idx) : "";
            ++idx;
            attributes["name"] = generate_name(prefix, _dunit, _extra);
            generate_measurement(ss, attributes, itr);
        }
        std::cout << ss.str() << std::flush;
    }

    template <typename _Up = _Tp, typename _Vt = value_type,
              enable_if_t<!(is_enabled<_Up>::value), char> = 0>
    echo_measurement(_Up&, string_t, const uint64_t&)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct copy
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::is_available<_Up>::value), char> = 0>
    copy(_Up& obj, const _Up& rhs)
    {
        obj = _Up(rhs);
    }

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::is_available<_Up>::value), char> = 0>
    copy(_Up*& obj, const _Up* rhs)
    {
        if(rhs)
        {
            if(!obj)
            {
                obj = new Type(*rhs);
            }
            else
            {
                *obj = Type(*rhs);
            }
        }
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::is_available<_Up>::value == false), char> = 0>
    copy(_Up&, const _Up&)
    {
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::is_available<_Up>::value == false), char> = 0>
    copy(_Up*&, const _Up*)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Op>
struct pointer_operator
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(base_type* obj, _Args&&... _args)
    {
        if(obj)
            _Op(*obj, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(Type* obj, _Args&&... _args)
    {
        if(obj)
            _Op(*obj, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(base_type* obj, base_type* rhs, _Args&&... _args)
    {
        if(obj && rhs)
            _Op(*obj, *rhs, std::forward<_Args>(_args)...);
    }

    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value), int> = 0>
    explicit pointer_operator(Type* obj, Type* rhs, _Args&&... _args)
    {
        if(obj && rhs)
            _Op(*obj, *rhs, std::forward<_Args>(_args)...);
    }

    // if the type is not available, never do anything
    template <typename _Up = _Tp, typename... _Args,
              tim::enable_if_t<(trait::is_available<_Up>::value == false), int> = 0>
    pointer_operator(_Args&&...)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct pointer_deleter
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit pointer_deleter(Type*& obj) { delete obj; }
    explicit pointer_deleter(base_type*& obj) { delete static_cast<Type*&>(obj); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct pointer_counter
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit pointer_counter(Type* obj, uint64_t& count)
    {
        if(obj)
            ++count;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct set_width
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;

    template <typename _Up, typename _Vp = value_type,
              enable_if_t<!(std::is_same<_Vp, void>::value), int> = 0>
    set_width(const _Up& val)
    {
        _Tp::get_width() = val;
    }

    template <typename _Up, typename _Vp = value_type,
              enable_if_t<(std::is_same<_Vp, void>::value), int> = 0>
    set_width(const _Up&)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct set_precision
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;

    template <typename _Up, typename _Vp = value_type,
              enable_if_t<!(std::is_same<_Vp, void>::value), int> = 0>
    set_precision(const _Up& val)
    {
        _Tp::get_precision() = val;
    }

    template <typename _Up, typename _Vp = value_type,
              enable_if_t<(std::is_same<_Vp, void>::value), int> = 0>
    set_precision(const _Up&)
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct set_format_flags
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;

    template <typename _Vp                                        = value_type,
              enable_if_t<!(std::is_same<_Vp, void>::value), int> = 0>
    set_format_flags(const std::ios_base::fmtflags& add_flags,
                     const std::ios_base::fmtflags& remove_flags =
                         (std::ios_base::fixed & std::ios_base::scientific))
    {
        _Tp::get_format_flags() &= remove_flags;
        _Tp::get_format_flags() |= add_flags;
    }

    template <typename _Vp                                       = value_type,
              enable_if_t<(std::is_same<_Vp, void>::value), int> = 0>
    set_format_flags(const std::ios_base::fmtflags&,
                     const std::ios_base::fmtflags& = (std::ios_base::fixed &
                                                       std::ios_base::scientific))
    {
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct set_units
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;

    template <typename _Up, typename _Vp = value_type,
              enable_if_t<!(std::is_same<_Vp, void>::value), int> = 0>
    set_units(const _Up& val, const std::string& str)
    {
        _Tp::get_unit()         = val;
        _Tp::get_display_unit() = str;
    }

    template <typename _Up, typename _Vp = value_type,
              enable_if_t<!(std::is_same<_Vp, void>::value), int> = 0>
    set_units(const std::tuple<std::string, _Up>& val)
    {
        _Tp::get_display_unit() = std::get<0>(val);
        _Tp::get_unit()         = std::get<1>(val);
    }

    template <typename _Up, typename _Vp = value_type,
              enable_if_t<(std::is_same<_Vp, void>::value), int> = 0>
    set_units(const _Up&, const std::string&)
    {
    }

    template <typename _Up, typename _Vp = value_type,
              enable_if_t<(std::is_same<_Vp, void>::value), int> = 0>
    set_units(const std::tuple<std::string, _Up>&)
    {
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace operation

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

inline tim::component::cpu_clock
operator+(const tim::component::user_clock&   _user,
          const tim::component::system_clock& _sys)
{
    return tim::operation::compose<tim::component::cpu_clock, tim::component::user_clock,
                                   tim::component::system_clock>::generate(_user, _sys);
}

//--------------------------------------------------------------------------------------//
