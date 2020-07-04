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

/**
 * \file timemory/operations/types/print.hpp
 * \brief Definition for various functions for print in operations
 */

#pragma once

//======================================================================================//
//
#include "timemory/operations/macros.hpp"
//
#include "timemory/operations/types.hpp"
//
#include "timemory/operations/declaration.hpp"
//
//======================================================================================//

#include "timemory/data/stream.hpp"
#include "timemory/mpl/type_traits.hpp"

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::print
/// \brief print routines for individual components
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using widths_t   = std::vector<int64_t>;

    // only if components are available
    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value), char> = 0>
    print(const type& _obj, std::ostream& _os, bool _endline = false)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        std::stringstream ss;
        ss << _obj;
        if(_endline)
            ss << '\n';
        _os << ss.str();
    }

    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value), char> = 0>
    print(std::size_t N, std::size_t Ntot, const type& _obj, std::ostream& _os,
          bool _endline)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        std::stringstream ss;
        ss << _obj;
        if(N + 1 < Ntot)
            ss << ", ";
        else if(N + 1 == Ntot && _endline)
            ss << '\n';
        _os << ss.str();
    }

    template <typename Vp, typename Statp, typename Up = Tp,
              enable_if_t<(is_enabled<Up>::value), char> = 0>
    print(const type& _obj, utility::stream& _os, const string_t& _prefix, int64_t _laps,
          int64_t _depth, const Vp& _self, const Statp& _stats)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        auto _labels = common_utils::get_labels(_obj);
        auto _units  = common_utils::get_display_units(_obj);

        utility::write_entry(_os, "LABEL", _prefix);
        utility::write_entry(_os, "COUNT", _laps);
        utility::write_entry(_os, "DEPTH", _depth);
        utility::write_entry(_os, "METRIC", _labels, true);
        utility::write_entry(_os, "UNITS", _units, true);
        if(trait::report_sum<type>::value && trait::report_values<type>::sum())
            utility::write_entry(_os, "SUM", _obj.get());
        if(trait::report_mean<type>::value && trait::report_values<type>::mean())
            utility::write_entry(_os, "MEAN", _obj.get() / _obj.get_laps());
        print_statistics<Tp>(_obj, _os, _self, _stats, _laps);
        utility::write_entry(_os, "% SELF", _self);
    }

    //----------------------------------------------------------------------------------//
    // only if components are available -- pointers
    //
    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value), char> = 0>
    print(const type* _obj, std::ostream& _os, bool _endline = false)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        if(_obj)
            print(*_obj, _os, _endline);
    }

    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value), char> = 0>
    print(std::size_t N, std::size_t Ntot, const type* _obj, std::ostream& _os,
          bool _endline)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        if(_obj)
            print(N, Ntot, *_obj, _os, _endline);
    }

    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value), char> = 0>
    print(const type* _obj, std::ostream& _os, const string_t& _prefix, int64_t _laps,
          int64_t _depth, const widths_t& _output_widths, bool _endline,
          const string_t& _suffix = "")
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        if(_obj)
            print(*_obj, _os, _prefix, _laps, _depth, _output_widths, _endline, _suffix);
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value == false), char> = 0>
    print(const type&, std::ostream&, bool = false)
    {}

    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value == false), char> = 0>
    print(std::size_t, std::size_t, const type&, std::ostream&, bool)
    {}

    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value == false), char> = 0>
    print(const type&, std::ostream&, const string_t&, int64_t, int64_t, const widths_t&,
          bool, const string_t& = "")
    {}

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available -- pointers
    //
    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value == false), char> = 0>
    print(const type*, std::ostream&, bool = false)
    {}

    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value == false), char> = 0>
    print(std::size_t, std::size_t, const type*, std::ostream&, bool)
    {}

    template <typename Up = Tp, enable_if_t<(is_enabled<Up>::value == false), char> = 0>
    print(const type*, std::ostream&, const string_t&, int64_t, int64_t, const widths_t&,
          bool, const string_t& = "")
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
