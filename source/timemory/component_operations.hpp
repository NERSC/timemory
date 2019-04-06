//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California,
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
struct record
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    record(base_type& obj) { obj.value = _Tp::record(); }

    template <typename _Up = _Tp, enable_if_t<(record_max<_Up>::value == true)> = 0>
    record(base_type& obj, const base_type& rhs)
    {
        obj = std::max(obj, rhs);
    }

    template <typename _Up = _Tp, enable_if_t<(record_max<_Up>::value == false)> = 0>
    record(base_type& obj, const base_type& rhs)
    {
        obj += rhs;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct reset
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    reset(base_type& obj) { obj.reset(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct start
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    start(base_type& obj) { obj.start(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct stop
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    stop(base_type& obj) { obj.stop(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_start
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    template <typename _Func>
    conditional_start(base_type& obj, _Func&& func)
    {
        bool did_start = obj.conditional_start();
        std::forward<_Func>(func)(did_start);
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct conditional_stop
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    conditional_stop(base_type& obj) { obj.conditional_stop(); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct print
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    print(std::size_t _N, std::size_t _Ntot, base_type& obj, std::ostream& os,
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
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct minus
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    minus(base_type& obj, const intmax_t& rhs) { obj -= rhs; }
    minus(base_type& obj, const base_type& rhs) { obj -= rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct plus
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    template <typename _Up = _Tp, enable_if_t<(record_max<_Up>::value == true), int> = 0>
    plus(base_type& obj, const base_type& rhs)
    {
        obj = std::max(obj, rhs);
    }

    template <typename _Up = _Tp, enable_if_t<(record_max<_Up>::value == false), int> = 0>
    plus(base_type& obj, const base_type& rhs)
    {
        obj += rhs;
    }

    plus(base_type& obj, const intmax_t& rhs) { obj += rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct multiply
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    multiply(base_type& obj, const intmax_t& rhs) { obj *= rhs; }
    multiply(base_type& obj, const base_type& rhs) { obj *= rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct divide
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    divide(base_type& obj, const intmax_t& rhs) { obj /= rhs; }
    divide(base_type& obj, const base_type& rhs) { obj /= rhs; }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Archive>
struct serial
{
    using value_type = typename _Tp::value_type;
    using base_type  = base<_Tp, value_type>;

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value)> = 0>
    serial(base_type& obj, Archive& ar, const unsigned int version)
    {
        ar(serializer::make_nvp(_Tp::label() + ".value", obj.accum),
           serializer::make_nvp(_Tp::label() + ".unit.value", _Tp::unit()),
           serializer::make_nvp(_Tp::label() + ".unit.repr", _Tp::display_unit()));
        consume_parameters(version);
    }

    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value)> = 0>
    serial(base_type& obj, Archive& ar, const unsigned int version)
    {
        auto value = static_cast<_Tp&>(obj).serial();
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
