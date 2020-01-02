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

/** \file mpl/bits/operations.hpp
 * \headerfile mpl/bits/operations.hpp "timemory/mpl/bits/operations.hpp"
 * These are some extra operations included in other place that add to the standard
 * mpl/operations but are separate so they can be included elsewhere to avoid
 * a cyclic include dependency
 *
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
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
        {
            _storage->stack_clear();
            _storage->print();
        }
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
        if(settings::dart_label())
        {
            return (_unit.length() > 0 && _unit != "%") ? join("//", Type::label(), _unit)
                                                        : Type::label();
        }

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
        if(settings::dart_label())
            return string_t("");

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
        auto _unit  = Type::get_display_unit();
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

}  // namespace operation

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
