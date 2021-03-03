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

#include "timemory/data/stream.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

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
/// \struct tim::operation::print
/// \brief print routines for individual components
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print
{
    using type     = Tp;
    using widths_t = std::vector<int64_t>;

    // only if components are available
    template <typename Up = Tp, enable_if_t<is_enabled<Up>::value, char> = 0>
    TIMEMORY_COLD print(const type& _obj, std::ostream& _os, bool _endline = false)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        std::stringstream ss;
        ss << _obj;
        if(_endline)
            ss << '\n';
        _os << ss.str();
    }

    template <typename Up = Tp, enable_if_t<is_enabled<Up>::value, char> = 0>
    TIMEMORY_COLD print(std::size_t N, std::size_t Ntot, const type& _obj,
                        std::ostream& _os, bool _endline)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        std::stringstream ss;
        ss << _obj;
        if(N + 1 < Ntot)
        {
            ss << ", ";
        }
        else if(N + 1 == Ntot && _endline)
        {
            ss << '\n';
        }
        _os << ss.str();
    }

    template <typename Vp, typename Statp, typename Up = Tp,
              enable_if_t<is_enabled<Up>::value, char> = 0>
    TIMEMORY_COLD print(const type& _obj, utility::stream& _os, const string_t& _prefix,
                        int64_t _laps, int64_t _depth, const Vp& _self,
                        const Statp& _stats)
    {
        auto _labels = common_utils::get_labels(_obj);
        auto _units  = common_utils::get_display_units(_obj);

        utility::write_entry(_os, "LABEL", _prefix);

        if(_laps > 0)
        {
            if(trait::report<type>::count())
                utility::write_entry(_os, "COUNT", _laps);
            if(trait::report<type>::depth())
                utility::write_entry(_os, "DEPTH", _depth);
            if(trait::report<type>::metric())
                utility::write_entry(_os, "METRIC", _labels, true);
            if(trait::report<type>::units())
                utility::write_entry(_os, "UNITS", _units, true);
            if(trait::report<type>::sum())
                utility::write_entry(_os, "SUM", _obj.get());
            if(trait::report<type>::mean())
                utility::write_entry(_os, "MEAN", _obj.get() / _obj.get_laps());
            if(trait::report<type>::stats())
                print_statistics<Tp>(_obj, _os, _self, _stats, _laps);
            if(trait::report<type>::self())
                utility::write_entry(_os, "% SELF", _self);
        }
        else
        {
            std::vector<std::string> _empty_data(_labels.size(), " ");
            if(trait::report<type>::count())
                utility::write_entry(_os, "COUNT", " ");
            if(trait::report<type>::depth())
                utility::write_entry(_os, "DEPTH", _depth);
            if(trait::report<type>::metric())
                utility::write_entry(_os, "METRIC", _empty_data);
            if(trait::report<type>::units())
                utility::write_entry(_os, "UNITS", _empty_data);
            if(trait::report<type>::sum())
                utility::write_entry(_os, "SUM", _empty_data);
            if(trait::report<type>::mean())
                utility::write_entry(_os, "MEAN", _empty_data);
            if(trait::report<type>::stats())
            {
                bool use_min    = get_env<bool>("TIMEMORY_PRINT_MIN", true);
                bool use_max    = get_env<bool>("TIMEMORY_PRINT_MIN", true);
                bool use_var    = get_env<bool>("TIMEMORY_PRINT_VARIANCE", false);
                bool use_stddev = get_env<bool>("TIMEMORY_PRINT_STDDEV", true);

                if(use_min)
                    utility::write_entry(_os, "MIN", _empty_data);
                if(use_max)
                    utility::write_entry(_os, "MAX", _empty_data);
                if(use_var)
                    utility::write_entry(_os, "VAR", _empty_data);
                if(use_stddev)
                    utility::write_entry(_os, "STDDEV", _empty_data);
            }
            if(trait::report<type>::self())
                utility::write_entry(_os, "% SELF", _empty_data);
        }
    }

    //----------------------------------------------------------------------------------//
    // only if components are available -- pointers
    //
    template <typename Up = Tp, enable_if_t<is_enabled<Up>::value, char> = 0>
    TIMEMORY_COLD print(const type* _obj, std::ostream& _os, bool _endline = false)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        if(_obj)
            print(*_obj, _os, _endline);
    }

    template <typename Up = Tp, enable_if_t<is_enabled<Up>::value, char> = 0>
    TIMEMORY_COLD print(std::size_t N, std::size_t Ntot, const type* _obj,
                        std::ostream& _os, bool _endline)
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        if(_obj)
            print(N, Ntot, *_obj, _os, _endline);
    }

    template <typename Up = Tp, enable_if_t<is_enabled<Up>::value, char> = 0>
    TIMEMORY_COLD print(const type* _obj, std::ostream& _os, const string_t& _prefix,
                        int64_t _laps, int64_t _depth, const widths_t& _output_widths,
                        bool _endline, const string_t& _suffix = "")
    {
        if(!trait::runtime_enabled<Tp>::get())
            return;

        if(_obj)
            print(*_obj, _os, _prefix, _laps, _depth, _output_widths, _endline, _suffix);
    }

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available
    //
    template <typename Up = Tp, enable_if_t<!is_enabled<Up>::value, char> = 0>
    print(const type&, std::ostream&, bool = false)
    {}

    template <typename Up = Tp, enable_if_t<!is_enabled<Up>::value, char> = 0>
    print(std::size_t, std::size_t, const type&, std::ostream&, bool)
    {}

    template <typename Up = Tp, enable_if_t<!is_enabled<Up>::value, char> = 0>
    print(const type&, std::ostream&, const string_t&, int64_t, int64_t, const widths_t&,
          bool, const string_t& = "")
    {}

    //----------------------------------------------------------------------------------//
    // print nothing if component is not available -- pointers
    //
    template <typename Up = Tp, enable_if_t<!is_enabled<Up>::value, char> = 0>
    print(const type*, std::ostream&, bool = false)
    {}

    template <typename Up = Tp, enable_if_t<!is_enabled<Up>::value, char> = 0>
    print(std::size_t, std::size_t, const type*, std::ostream&, bool)
    {}

    template <typename Up = Tp, enable_if_t<!is_enabled<Up>::value, char> = 0>
    print(const type*, std::ostream&, const string_t&, int64_t, int64_t, const widths_t&,
          bool, const string_t& = "")
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
