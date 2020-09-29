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
 * \file timemory/operations/types/base_printer.hpp
 * \brief Definition for various functions for base_printer in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
/// \struct operation::base_printer
/// \brief invoked from the base class to provide default printing behavior
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct base_printer : public common_utils
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using widths_t   = std::vector<int64_t>;

    template <typename Up                                        = value_type,
              enable_if_t<!(std::is_same<Up, void>::value), int> = 0>
    explicit base_printer(std::ostream& _os, const type& _obj)
    {
        auto _value = static_cast<const type&>(_obj).get_display();
        auto _disp  = type::get_display_unit();
        auto _label = type::get_label();

        sfinae(_os, 0, _value, _disp, _label);
    }

    template <typename Up                                       = value_type,
              enable_if_t<(std::is_same<Up, void>::value), int> = 0>
    explicit base_printer(std::ostream&, const type&)
    {}

private:
    template <typename Up>
    auto label_sfinae(const Up& _obj, int)
        -> decltype(_obj.label_array(), std::vector<std::string>())
    {
        auto                     arr = _obj.label_array();
        std::vector<std::string> ret;
        ret.assign(arr.begin(), arr.end());
        return ret;
    }

    template <typename Up>
    auto label_sfinae(const Up& _obj, long)
    {
        return _obj.label();
    }

    template <typename Up>
    auto disp_sfinae(const Up& _obj, int)
        -> decltype(_obj.display_unit_array(), std::vector<std::string>())
    {
        auto                     arr = _obj.display_unit_array();
        std::vector<std::string> ret;
        ret.assign(arr.begin(), arr.end());
        return ret;
    }

    template <typename Up>
    auto disp_sfinae(const Up& _obj, long)
    {
        return _obj.display_unit();
    }

    //----------------------------------------------------------------------------------//
    //
    //      Array of values + Array of labels + Array of display units
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ValueT, typename DispT, typename LabelT,
              enable_if_t<!(std::is_same<decay_t<ValueT>, std::string>::value), int> = 0,
              enable_if_t<!(std::is_same<decay_t<LabelT>, std::string>::value), int> = 0,
              enable_if_t<!(std::is_same<decay_t<DispT>, std::string>::value), int>  = 0>
    auto sfinae(std::ostream& _os, int, ValueT& _value, DispT& _disp, LabelT& _label)
        -> decltype((_value.size() + _disp.size() + _label.size()), void())
    {
        auto _prec  = type::get_precision();
        auto _width = type::get_width();
        auto _flags = type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);

        auto              _vn = _value.size();
        auto              _dn = _disp.size();
        auto              _ln = _label.size();
        auto              _n  = std::min<size_t>(_vn, std::min<size_t>(_dn, _ln));
        std::stringstream _ss;
        for(size_t i = 0; i < _n; ++i)
        {
            ss_value << std::setw(_width) << std::setprecision(_prec)
                     << _value.at(i % _vn);

            // get display returned an empty string
            if(ss_value.str().find_first_not_of(' ') == std::string::npos)
                continue;

            // check traits to see if we should print
            constexpr bool units_print = !trait::custom_unit_printing<type>::value;
            constexpr bool label_print = !trait::custom_label_printing<type>::value;

            print_tag<units_print>(ss_extra, _disp.at(i % _dn));
            print_tag<label_print>(ss_extra, _label.at(i % _ln));

            _ss << ss_value.str() << ss_extra.str();
            if(i + 1 < _n)
                _ss << ", ";
        }

        _os << _ss.str();
    }

    //----------------------------------------------------------------------------------//
    //
    //      Array of values + Array of labels + string display unit
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ValueT, typename DispT, typename LabelT,
              enable_if_t<!(std::is_same<decay_t<ValueT>, std::string>::value), int> = 0,
              enable_if_t<!(std::is_same<decay_t<LabelT>, std::string>::value), int> = 0,
              enable_if_t<(std::is_same<decay_t<DispT>, std::string>::value), int>   = 0>
    auto sfinae(std::ostream& _os, int, ValueT& _value, DispT& _disp, LabelT& _label)
        -> decltype((_value.size() + _label.size()), void())
    {
        auto _prec  = type::get_precision();
        auto _width = type::get_width();
        auto _flags = type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);

        auto              _vn = _value.size();
        auto              _ln = _label.size();
        auto              _n  = std::min<size_t>(_vn, _ln);
        std::stringstream _ss;
        for(size_t i = 0; i < _n; ++i)
        {
            ss_value << std::setw(_width) << std::setprecision(_prec)
                     << _value.at(i % _vn);

            // get display returned an empty string
            if(ss_value.str().find_first_not_of(' ') == std::string::npos)
                continue;

            // check traits to see if we should print
            constexpr bool units_print = !trait::custom_unit_printing<type>::value;
            constexpr bool label_print = !trait::custom_label_printing<type>::value;

            if(i + 1 < _n)
                print_tag<units_print>(ss_extra, _disp);
            print_tag<label_print>(ss_extra, _label.at(i % _ln));

            _ss << ss_value.str() << ss_extra.str();
            if(i + 1 < _n)
                _ss << ", ";
        }

        _os << _ss.str();
    }

    //----------------------------------------------------------------------------------//
    //
    //      Array of values + string label + string display unit
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ValueT, typename DispT, typename LabelT,
              enable_if_t<!(std::is_same<decay_t<ValueT>, std::string>::value), int> = 0,
              enable_if_t<(std::is_same<decay_t<LabelT>, std::string>::value), int>  = 0,
              enable_if_t<(std::is_same<decay_t<DispT>, std::string>::value), int>   = 0>
    auto sfinae(std::ostream& _os, int, ValueT& _value, DispT& _disp, LabelT& _label)
        -> decltype((_value.size() + _label.size()), void())
    {
        auto _prec  = type::get_precision();
        auto _width = type::get_width();
        auto _flags = type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);

        auto              _vn = _value.size();
        auto              _ln = _label.size();
        auto              _n  = std::min<size_t>(_vn, _ln);
        std::stringstream _ss;
        for(size_t i = 0; i < _n; ++i)
        {
            ss_value << std::setw(_width) << std::setprecision(_prec)
                     << _value.at(i % _vn);

            // get display returned an empty string
            if(ss_value.str().find_first_not_of(' ') == std::string::npos)
                continue;

            if(i + 1 < _n)
            {
                // check traits to see if we should print
                constexpr bool units_print = !trait::custom_unit_printing<type>::value;
                constexpr bool label_print = !trait::custom_label_printing<type>::value;

                print_tag<units_print>(ss_extra, _disp);
                print_tag<label_print>(ss_extra, _label);
            }

            _ss << ss_value.str() << ss_extra.str();
            if(i + 1 < _n)
                _ss << ", ";
        }

        _os << _ss.str();
    }

    //----------------------------------------------------------------------------------//
    //
    //      Scalar value + string label + string display unit
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ValueT, typename DispT, typename LabelT>
    auto sfinae(std::ostream& _os, long, ValueT& _value, DispT& _disp, LabelT& _label)
    {
        auto _prec  = type::get_precision();
        auto _width = type::get_width();
        auto _flags = type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;

        // get display returned an empty string
        if(ss_value.str().find_first_not_of(' ') == std::string::npos)
            return;

        // check traits to see if we should print
        constexpr bool units_print = !trait::custom_unit_printing<type>::value;
        constexpr bool label_print = !trait::custom_label_printing<type>::value;

        print_tag<units_print>(ss_extra, _disp);
        print_tag<label_print>(ss_extra, _label);

        _os << ss_value.str() << ss_extra.str();
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
