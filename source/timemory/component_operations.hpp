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

#pragma once

#include "timemory/components.hpp"

//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace component
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct set_prefix
{
    using Type     = _Tp;
    using string_t = std::string;

    set_prefix(Type& obj, const bool& exists, const string_t& _prefix)
    {
        if(!exists)
            obj.set_prefix(_prefix);
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct insert_node
{
    using Type = _Tp;

    insert_node(std::size_t _N, std::size_t, Type& obj, bool* exists, const intmax_t& id)
    {
        obj.insert_node(exists[_N], id);
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct pop_node
{
    using Type = _Tp;

    pop_node(Type& obj) { obj.pop_node(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct record
{
    using Type = _Tp;

    record(Type& obj) { obj.value = _Tp::record(); }

    template <typename _Up = _Tp, enable_if_t<(record_max<_Up>::value == true)> = 0>
    record(Type& obj, const Type& rhs)
    {
        obj = std::max(obj, rhs);
    }

    template <typename _Up = _Tp, enable_if_t<(record_max<_Up>::value == false)> = 0>
    record(Type& obj, const Type& rhs)
    {
        obj += rhs;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct reset
{
    using Type = _Tp;

    reset(Type& obj) { obj.reset(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct measure
{
    using Type = _Tp;

    measure(Type& obj) { obj.measure(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct start
{
    using Type = _Tp;

    start(Type& obj) { obj.start(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct stop
{
    using Type = _Tp;

    stop(Type& obj) { obj.stop(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_start
{
    using Type = _Tp;

    template <typename _Func>
    conditional_start(Type& obj, _Func&& func)
    {
        bool did_start = obj.conditional_start();
        std::forward<_Func>(func)(did_start);
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_stop
{
    using Type = _Tp;

    conditional_stop(Type& obj) { obj.conditional_stop(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct minus
{
    using Type = _Tp;

    minus(Type& obj, const intmax_t& rhs) { obj -= rhs; }
    minus(Type& obj, const Type& rhs) { obj -= rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct plus
{
    using Type = _Tp;

    template <typename _Up = _Tp, enable_if_t<(record_max<_Up>::value == true), int> = 0>
    plus(Type& obj, const Type& rhs)
    {
        obj = std::max(obj, rhs);
    }

    template <typename _Up = _Tp, enable_if_t<(record_max<_Up>::value == false), int> = 0>
    plus(Type& obj, const Type& rhs)
    {
        obj += rhs;
    }

    plus(Type& obj, const intmax_t& rhs) { obj += rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct multiply
{
    using Type = _Tp;

    multiply(Type& obj, const intmax_t& rhs) { obj *= rhs; }
    multiply(Type& obj, const Type& rhs) { obj *= rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct divide
{
    using Type = _Tp;

    divide(Type& obj, const intmax_t& rhs) { obj /= rhs; }
    divide(Type& obj, const Type& rhs) { obj /= rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct print
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = base<Type, value_type>;

    print(std::size_t _N, std::size_t _Ntot, const Type& obj, std::ostream& os,
          bool endline)
    {
        std::stringstream ss;
        ss << obj;
        if(_N + 1 < _Ntot)
            ss << ", ";
        else if(_N + 1 == _Ntot && endline)
            ss << std::endl;
        os << ss.str();
    }

    print(const Type& obj, std::ostream& os, const string_t& _prefix, intmax_t _laps,
          intmax_t _output_width, bool endline)
    {
        std::stringstream ss_prefix;
        std::stringstream ss;
        ss_prefix << std::setw(_output_width) << std::left << _prefix << " : ";
        ss << ss_prefix.str() << static_cast<base_type>(obj) << ", " << _laps << " laps";
        if(endline)
            ss << std::endl;
        os << ss.str();
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Archive>
struct serial
{
    using value_type = typename _Tp::value_type;
    using Type       = _Tp;

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    serial(Type& obj, Archive& ar, const unsigned int version)
    {
        ar(serializer::make_nvp(_Tp::label() + ".value", obj.accum),
           serializer::make_nvp(_Tp::label() + ".unit.value", _Tp::unit()),
           serializer::make_nvp(_Tp::label() + ".unit.repr", _Tp::display_unit()));
        consume_parameters(version);
    }

    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value)> = 0>
    serial(Type& obj, Archive& ar, const unsigned int version)
    {
        auto value = static_cast<Type&>(obj).serial();
        ar(serializer::make_nvp(_Tp::label() + ".value", value),
           serializer::make_nvp(_Tp::label() + ".unit.value", _Tp::unit()),
           serializer::make_nvp(_Tp::label() + ".unit.repr", _Tp::display_unit()));
        consume_parameters(version);
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace component

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
