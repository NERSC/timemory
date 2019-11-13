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

#include "timemory/bits/types.hpp"
#include "timemory/components.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
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

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct init_storage
{
    using Type         = _Tp;
    using value_type   = typename Type::value_type;
    using base_type    = typename Type::base_type;
    using string_t     = std::string;
    using storage_type = storage<Type>;

    template <typename _Up                                         = _Tp,
              enable_if_t<(trait::is_available<_Up>::value), char> = 0>
    init_storage()
    {
        static thread_local auto _instance = storage_type::instance();
        _instance->initialize();
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::is_available<_Up>::value == false), char> = 0>
    init_storage()
    {}

    using master_pointer_t = decltype(storage_type::master_instance());
    using pointer_t        = decltype(storage_type::instance());

    using get_type = std::tuple<master_pointer_t, pointer_t, bool, bool, bool>;

    template <typename U = base_type, enable_if_t<(U::implements_storage_v), int> = 0>
    static get_type get()
    {
        static auto _lambda = []() {
            static thread_local auto _main_inst = storage_type::master_instance();
            static thread_local auto _this_inst = storage_type::instance();
            if(_main_inst != _this_inst)
            {
                static bool              _main_glob = _main_inst->global_init();
                static bool              _this_glob = _this_inst->global_init();
                static thread_local bool _main_work = _main_inst->thread_init();
                static thread_local bool _this_work = _this_inst->thread_init();
                static thread_local bool _main_data = _main_inst->data_init();
                static thread_local bool _this_data = _this_inst->data_init();
                return get_type{ _main_inst, _this_inst, (_main_glob && _this_glob),
                                 (_main_work && _this_work), (_main_data && _this_data) };
            }
            else
            {
                static bool              _this_glob = _this_inst->global_init();
                static thread_local bool _this_work = _this_inst->thread_init();
                static thread_local bool _this_data = _this_inst->data_init();
                return get_type{ _main_inst, _this_inst, (_this_glob), (_this_work),
                                 (_this_data) };
            }
        };
        static thread_local auto _instance = _lambda();
        return _instance;
    }

    template <typename U = base_type, enable_if_t<!(U::implements_storage_v), int> = 0>
    static get_type get()
    {
        static auto _lambda = []() {
            static thread_local auto _main_inst = storage_type::master_instance();
            static thread_local auto _this_inst = storage_type::instance();
            return get_type{ _main_inst, _this_inst, false, false, false };
        };
        static thread_local auto _instance = _lambda();
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct set_prefix
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using string_t   = std::string;

    template <typename _Up                                           = _Tp,
              enable_if_t<(trait::requires_prefix<_Up>::value), int> = 0>
    set_prefix(Type& obj, const string_t& _prefix)
    {
        obj.set_prefix(_prefix);
    }

    template <typename _Up                                                    = _Tp,
              enable_if_t<(trait::requires_prefix<_Up>::value == false), int> = 0>
    set_prefix(Type&, const string_t&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Scope>
struct insert_node
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    //  has storage implementation
    //
    template <typename _Up = base_type, enable_if_t<(_Up::implements_storage_v), int> = 0>
    explicit insert_node(base_type& obj, const uint64_t& _hash)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);

        obj.insert_node(_Scope{}, _hash);
    }

    //----------------------------------------------------------------------------------//
    //  no storage implementation
    //
    template <typename _Up                                   = base_type,
              enable_if_t<!(_Up::implements_storage_v), int> = 0>
    explicit insert_node(base_type&, const uint64_t&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct pop_node
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    //  has storage implementation
    //
    template <typename _Up = base_type, enable_if_t<(_Up::implements_storage_v), int> = 0>
    explicit pop_node(base_type& obj)
    {
        obj.pop_node();
    }

    //----------------------------------------------------------------------------------//
    //  no storage implementation
    //
    template <typename _Up                                   = base_type,
              enable_if_t<!(_Up::implements_storage_v), int> = 0>
    explicit pop_node(base_type&)
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct record
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    explicit record(base_type& obj)
    {
        obj.value = Type::record();
    }

    template <typename _Up = _Tp, enable_if_t<(trait::record_max<_Up>::value), int> = 0,
              enable_if_t<(is_enabled<_Up>::value), char> = 0>
    record(base_type& obj, const base_type& rhs)
    {
        obj = std::max(obj, rhs);
    }

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::record_max<_Up>::value == false), int> = 0,
              enable_if_t<(is_enabled<_Up>::value), char>                = 0>
    record(base_type& obj, const base_type& rhs)
    {
        obj += rhs;
    }

    template <typename... _Args, typename _Up = _Tp,
              enable_if_t<!(is_enabled<_Up>::value), char> = 0>
    record(_Args&&...)
    {}
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

    explicit measure(base_type& obj)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
        obj.measure();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct start
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    explicit start(base_type& obj)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
        obj.start();
    }
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
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
        obj.start();
    }

    template <typename _Up                                                   = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    explicit priority_start(base_type&)
    {}
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
    {}

    template <typename _Up                                                   = _Tp,
              enable_if_t<(trait::start_priority<_Up>::value == false), int> = 0>
    explicit standard_start(base_type& obj)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
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
        // obj.activate_noop();
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
    {}
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
    {}

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::stop_priority<_Up>::value == false), int> = 0>
    explicit standard_stop(base_type& obj)
    {
        obj.stop();
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
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
        obj.mark_begin();
    }

    template <typename _Up                                                        = _Tp,
              enable_if_t<!(trait::supports_args<_Up, std::tuple<>>::value), int> = 0>
    explicit mark_begin(Type&)
    {}

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                     = 0,
              enable_if_t<(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_begin(Type& obj, _Args&&... _args)
    {
        static thread_local auto _init = init_storage<_Tp>::get();
        consume_parameters(_init);
        obj.mark_begin(std::forward<_Args>(_args)...);
    }

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(sizeof...(_Args) > 0), int>                      = 0,
              enable_if_t<!(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    mark_begin(Type&, _Args&&...)
    {}
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
    {}

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
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::customize
///
/// \brief The purpose of this operation class is for a component to provide some extra
/// customization within a GOTCHA function.
///
/// It will require overloading `tim::trait::supports_args`:
///   `template <> trait::supports_args<MyType, std::tuple<string, _Args...>> : true_type`
/// where `_Args...` are the GOTCHA function arguments. The string will be the function
/// name (possibly mangled). One such purpose may be to create a custom component
/// that intercepts a malloc and uses the arguments to get the exact allocation
/// size.
///
template <typename _Tp>
struct customize
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    customize(Type& obj, _Args&&... _args)
    {
        obj.customize(std::forward<_Args>(_args)...);
    }

    template <typename... _Args, typename _Tuple = std::tuple<decay_t<_Args>...>,
              enable_if_t<!(trait::supports_args<_Tp, _Tuple>::value), int> = 0>
    customize(Type&, _Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::compose
///
/// \brief The purpose of this operation class is operating on two components to compose
/// a result, e.g. use system-clock and user-clock to get a cpu-clock
///
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
///
/// \class operation::plus
///
/// \brief Define addition operations
///
template <typename _Tp>
struct plus
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    plus(Type& obj, const int64_t& rhs) { obj += rhs; }

    template <typename _Up = _Tp, enable_if_t<(trait::record_max<_Up>::value), int> = 0>
    plus(Type& obj, const Type& rhs)
    {
        obj.base_type::plus(rhs);
        obj = std::max(obj, rhs);
    }

    template <typename _Up                                               = _Tp,
              enable_if_t<(trait::record_max<_Up>::value == false), int> = 0>
    plus(Type& obj, const Type& rhs)
    {
        obj.base_type::plus(rhs);
        obj += rhs;
    }
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::minus
///
/// \brief Define subtraction operations
///
template <typename _Tp>
struct minus
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    minus(Type& obj, const int64_t& rhs) { obj -= rhs; }

    minus(Type& obj, const Type& rhs)
    {
        // ensures update to laps
        obj.base_type::minus(rhs);
        obj -= rhs;
    }
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
///
/// \class operation::get_data
///
/// \brief The purpose of this operation class is to combine the output types from the
/// "get()" member function for multiple components -- this is specifically used in the
/// Python interface to provide direct access to the results
///
template <typename _Tp>
struct get_data
{
    using Type            = _Tp;
    using DataType        = decltype(std::declval<Type>().get());
    using LabeledDataType = std::tuple<std::string, decltype(std::declval<Type>().get())>;

    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    get_data(const Type& _obj, DataType& _dst)
    {
        _dst = _obj.get();
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    get_data(const Type&, DataType&)
    {}

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    get_data(const Type& _obj, LabeledDataType& _dst)
    {
        _dst = LabeledDataType(Type::label(), _obj.get());
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    get_data(const Type&, LabeledDataType&)
    {}
};

//--------------------------------------------------------------------------------------//
/// \class base_printer
/// \brief invoked from the base class to provide default printing behavior
//
template <typename _Tp>
struct base_printer
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using widths_t   = std::vector<int64_t>;

    //----------------------------------------------------------------------------------//
    // invoked from the base class
    //
    explicit base_printer(std::ostream& _os, const base_type& _obj)
    {
        auto _value = static_cast<const Type&>(_obj).get_display();
        auto _label = base_type::get_label();
        auto _disp  = base_type::get_display_unit();
        auto _prec  = base_type::get_precision();
        auto _width = base_type::get_width();
        auto _flags = base_type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;
        if(!_disp.empty() && !trait::custom_unit_printing<Type>::value)
            ss_extra << " " << _disp;
        if(!_label.empty() && !trait::custom_label_printing<Type>::value)
            ss_extra << " " << _label;

        _os << ss_value.str() << ss_extra.str();
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
    {}

    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(std::size_t, std::size_t, const Type&, std::ostream&, bool)
    {}

    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(const Type&, std::ostream&, const string_t&, int64_t, int64_t, const widths_t&,
          bool, const string_t& = "")
    {}

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available -- pointers
    //
    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(const Type*, std::ostream&, bool = false)
    {}

    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(std::size_t, std::size_t, const Type*, std::ostream&, bool)
    {}

    template <typename _Up                                         = _Tp,
              enable_if_t<(is_enabled<_Up>::value == false), char> = 0>
    print(const Type*, std::ostream&, const string_t&, int64_t, int64_t, const widths_t&,
          bool, const string_t& = "")
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct print_storage
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

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
    {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Archive>
struct serialization
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename _Up = _Tp, enable_if_t<(is_enabled<_Up>::value), char> = 0>
    serialization(const base_type& obj, _Archive& ar, const unsigned int)
    {
        auto _data = static_cast<const Type&>(obj).get();
        ar(serializer::make_nvp("is_transient", obj.is_transient),
           serializer::make_nvp("laps", obj.laps),
           serializer::make_nvp("repr_data", _data),
           serializer::make_nvp("value", obj.value),
           serializer::make_nvp("accum", obj.accum));
    }

    template <typename _Up = _Tp, enable_if_t<!(is_enabled<_Up>::value), char> = 0>
    serialization(const base_type&, _Archive&, const unsigned int)
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::echo_measurement
///
/// \brief This operation class echoes DartMeasurements for a CDash dashboard
///
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
    /// generate an attribute
    ///
    static string_t attribute_string(const string_t& key, const string_t& item)
    {
        return apply<string_t>::join("", key, "=", "\"", item, "\"");
    }

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
    }

    //----------------------------------------------------------------------------------//
    /// convert to lowercase
    ///
    static string_t lowercase(string_t _str)
    {
        for(auto& itr : _str)
            itr = tolower(itr);
        return _str;
    }

    //----------------------------------------------------------------------------------//
    /// convert to uppercase
    ///
    static string_t uppercase(string_t _str)
    {
        for(auto& itr : _str)
            itr = toupper(itr);
        return _str;
    }

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
    }

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
    static string_t generate_name(const string_t& _prefix, string_t _unit,
                                  _Args&&... _args)
    {
        auto _extra = join("/", std::forward<_Args>(_args)...);
        auto _label = uppercase(Type::label());
        _unit       = replace(_unit, "", { " " });
        string_t _name =
            (_extra.length() > 0) ? join("//", _extra, _prefix) : join("//", _prefix);

        auto _ret = join("//", _label, _name);

        if(_ret.length() > 0 && _ret.at(_ret.length() - 1) == '/')
            _ret.erase(_ret.length() - 1);

        if(_unit.length() > 0 && _unit != "%")
            _ret += "//" + _unit;

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
        os << " " << attribute_string("type", "numeric/double");
        for(const auto& itr : attributes)
            os << " " << attribute_string(itr.first, itr.second);
        os << ">" << std::setprecision(Type::get_precision()) << value
           << "</DartMeasurement>\n\n";
    }

    //----------------------------------------------------------------------------------//
    /// generate a measurement tag
    ///
    template <typename _Vt, typename... _Extra>
    static void generate_measurement(std::ostream& os, attributes_t attributes,
                                     const std::vector<_Vt, _Extra...>& value)
    {
        auto _default_name = attributes["name"];
        int  i             = 0;
        for(const auto& itr : value)
        {
            std::stringstream ss;
            ss << "INDEX_" << i++ << " ";
            attributes["name"] = ss.str() + _default_name;
            generate_measurement(os, attributes, itr);
        }
    }

    //----------------------------------------------------------------------------------//
    /// generate the prefix
    ///
    static string_t generate_prefix(const strvec_t& hierarchy)
    {
        string_t              ret_prefix = "";
        string_t              add_prefix = "";
        static const strset_t repl_chars = { "\t", "\n", "<", ">" };
        for(const auto& itr : hierarchy)
        {
            auto prefix = itr;
            prefix      = replace(prefix, "", { ">>>" });
            prefix      = replace(prefix, "", { "|_" });
            prefix      = replace(prefix, "_", repl_chars);
            prefix      = replace(prefix, "_", { "__" });
            if(prefix.length() > 0 && prefix.at(prefix.length() - 1) == '_')
                prefix.erase(prefix.length() - 1);
            ret_prefix += add_prefix + prefix;
        }
        return ret_prefix;
    }

    //----------------------------------------------------------------------------------//
    /// assumes type is not a iterable
    ///
    template <typename _Up = _Tp, typename _Vt = value_type,
              enable_if_t<(is_enabled<_Up>::value), char> = 0,
              enable_if_t<!(trait::array_serialization<_Up>::value ||
                            trait::iterable_measurement<_Up>::value),
                          int>                            = 0>
    echo_measurement(_Up& obj, const strvec_t& hierarchy)
    {
        auto prefix = generate_prefix(hierarchy);
        auto _unit  = Type::display_unit();
        auto name   = generate_name(prefix, _unit);
        auto _data  = obj.get();

        attributes_t   attributes = { { "name", name } };
        stringstream_t ss;
        generate_measurement(ss, attributes, _data);
        std::cout << ss.str() << std::flush;
    }

    //----------------------------------------------------------------------------------//
    /// assumes type is iterable
    ///
    template <typename _Up = _Tp, typename _Vt = value_type,
              enable_if_t<(is_enabled<_Up>::value), char> = 0,
              enable_if_t<(trait::array_serialization<_Up>::value ||
                           trait::iterable_measurement<_Up>::value),
                          int>                            = 0>
    echo_measurement(_Up& obj, const strvec_t& hierarchy)
    {
        auto prefix = generate_prefix(hierarchy);
        auto _data  = obj.get();

        attributes_t   attributes = {};
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

    template <typename... _Args, typename _Up = _Tp, typename _Vt = value_type,
              enable_if_t<!(is_enabled<_Up>::value), char> = 0>
    echo_measurement(_Up&, _Args&&...)
    {}
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
                obj = new Type(*rhs);
            else
                *obj = Type(*rhs);
        }
    }

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::is_available<_Up>::value == false), char> = 0>
    copy(_Up&, const _Up&)
    {}

    template <typename _Up                                                  = _Tp,
              enable_if_t<(trait::is_available<_Up>::value == false), char> = 0>
    copy(_Up*&, const _Up*)
    {}
};

//--------------------------------------------------------------------------------------//
///
/// \class operation::pointer_operator
///
/// \brief This operation class enables pointer-safety for the components created
/// on the heap (e.g. within a component_list) by ensuring other operation
/// classes are not invoked on a null pointer
///
template <typename _Tp, typename _Op>
struct pointer_operator
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    pointer_operator()                        = delete;
    pointer_operator(const pointer_operator&) = delete;
    pointer_operator(pointer_operator&&)      = delete;

    pointer_operator& operator=(const pointer_operator&) = delete;
    pointer_operator& operator=(pointer_operator&&) = delete;

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
    {}
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
