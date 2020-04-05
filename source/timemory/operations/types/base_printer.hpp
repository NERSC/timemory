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

//======================================================================================//
//
#include "timemory/operations/macros.hpp"
//
#include "timemory/operations/types.hpp"
//
#include "timemory/operations/declaration.hpp"
//
//======================================================================================//

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
/// \class operation::base_printer
/// \brief invoked from the base class to provide default printing behavior
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct base_printer : public common_utils
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;
    using widths_t   = std::vector<int64_t>;

    template <typename Up                                        = value_type,
              enable_if_t<!(std::is_same<Up, void>::value), int> = 0>
    explicit base_printer(std::ostream& _os, const type& _obj)
    {
        auto _value = static_cast<const type&>(_obj).get_display();
        auto _disp  = type::get_display_unit();
        auto _label = type::get_label();
        auto _prec  = type::get_precision();
        auto _width = type::get_width();
        auto _flags = type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;

        // get display returned an empty string
        if(ss_value.str().find_first_not_of(" ") == std::string::npos)
            return;

        // check traits to see if we should print
        constexpr bool units_print = !trait::custom_unit_printing<type>::value;
        constexpr bool label_print = !trait::custom_label_printing<type>::value;

        print_tag<units_print>(ss_extra, _disp);
        print_tag<label_print>(ss_extra, _label);

        _os << ss_value.str() << ss_extra.str();
    }

    template <typename Up                                       = value_type,
              enable_if_t<(std::is_same<Up, void>::value), int> = 0>
    explicit base_printer(std::ostream&, const type&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
