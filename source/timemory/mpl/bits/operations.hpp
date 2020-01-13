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
#include "timemory/data/statistics.hpp"
#include "timemory/mpl/stl_overload.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/mpl/zip.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/serializer.hpp"

#include <array>
#include <bitset>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace operation
{
//--------------------------------------------------------------------------------------//
/// \class common_utils
/// \brief common string manipulation utilities
///
struct common_utils
{
    using attributes_t   = std::map<string_t, string_t>;
    using strset_t       = std::set<string_t>;
    using stringstream_t = std::stringstream;
    using strvec_t       = std::vector<string_t>;

public:
    template <typename _Tp>
    static size_t get_distance(const _Tp& _data)
    {
        return get_distance_sfinae(_data);
    }

    template <typename _Tp>
    static auto get_distance_sfinae(const _Tp& _data, int)
        -> decltype(std::distance(_data.begin(), _data.end()), size_t())
    {
        return std::distance(_data.begin(), _data.end());
    }

    template <typename _Tp>
    static auto get_distance_sfinae(const _Tp&, long) -> size_t
    {
        return size_t(1);
    }

    template <typename _Tp>
    static auto get_distance_sfinae(const _Tp& _data)
        -> decltype(get_distance_sfinae(_data, 0))
    {
        return get_distance_sfinae(_data, 0);
    }

public:
    template <typename _Tp, enable_if_t<(std::is_arithmetic<_Tp>::value), int> = 0>
    static _Tp get_entry(const _Tp& _data, size_t)
    {
        return _data;
    }

    template <typename _Tp, enable_if_t<!(std::is_arithmetic<_Tp>::value), int> = 0>
    static auto get_entry(const _Tp& _data, size_t _idx)
        -> decltype(get_entry_sfinae_(_data, _idx))
    {
        return get_entry_sfinae_<_Tp>(_data, _idx);
    }

    template <typename _Tp, size_t _Idx>
    static _Tp get_entry(const _Tp& _data, size_t)
    {
        return _data;
    }

    template <typename _Tp>
    static auto get_entry_sfinae(const _Tp& _data, int, size_t _idx)
        -> decltype(_data.begin(), typename _Tp::value_type())
    {
        auto sz  = std::distance(_data.begin(), _data.end());
        auto n   = _idx % sz;
        auto itr = _data.begin();
        std::advance(itr, n);
        return *itr;
    }

    template <typename _Tp>
    static _Tp get_entry_sfinae(const _Tp& _data, long, size_t)
    {
        return _data;
    }

    template <typename _Tp>
    static auto get_entry_sfinae_(const _Tp& _data, size_t _idx)
        -> decltype(get_entry_sfinae(_data, 0, _idx))
    {
        return get_entry_sfinae<_Tp>(_data, 0, _idx);
    }

public:
    template <typename _Tp, typename _Wp, typename _Pp>
    static void write(std::vector<std::stringstream*>& _os,
                      std::ios_base::fmtflags _format, const _Tp& _data,
                      const _Wp& _width, const _Pp& _prec)
    {
        size_t num_data = get_distance(_data);

        for(size_t i = 0; i < num_data; ++i)
        {
            auto  _idata  = get_entry<_Tp>(_data, i);
            auto  _iwidth = get_entry<_Wp>(_width, i);
            auto  _iprec  = get_entry<_Pp>(_prec, i);
            auto* ss      = new std::stringstream;
            ss->setf(_format);
            (*ss) << std::setw(_iwidth) << std::setprecision(_iprec) << _idata;
            _os.emplace_back(ss);
        }
    }

    template <typename... _Tp, size_t... _Idx, typename _Wp, typename _Pp>
    static void write(std::vector<std::stringstream*>& _os,
                      std::ios_base::fmtflags _format, const std::tuple<_Tp...>& _data,
                      const _Wp& _width, const _Pp& _prec, index_sequence<_Idx...>)
    {
        using init_list_type = std::initializer_list<int>;
        auto&& ret           = init_list_type{ (
            write(_os, _format, std::get<_Idx>(_data), _width, _prec), 0)... };
        consume_parameters(ret);
    }

    template <typename... _Tp, typename _Wp, typename _Pp>
    static void write(std::vector<std::stringstream*>& _os,
                      std::ios_base::fmtflags _format, const std::tuple<_Tp...>& _data,
                      const _Wp& _width, const _Pp& _prec)
    {
        constexpr size_t _N = sizeof...(_Tp);
        write(_os, _format, _data, _width, _prec, make_index_sequence<_N>{});
    }

public:
    template <typename _Tp>
    static int64_t get_size(const _Tp& _data)
    {
        return get_labels_sfinae(_data, 0);
    }

    template <typename _Tp>
    static auto get_size_sfinae(const _Tp& _data, int)
        -> decltype(_data.label_array(), int64_t())
    {
        return _data.label_array().size();
    }

    template <typename _Tp>
    static auto get_size_sfinae(const _Tp&, long) -> int64_t
    {
        return 1;
    }

public:
    template <typename _Tp>
    static auto get_labels_sfinae(const _Tp& _data, int)
        -> decltype(_data.label_array(), strvec_t())
    {
        strvec_t _ret;
        for(const auto& itr : _data.label_array())
            _ret.push_back(itr);
        return _ret;
    }

    template <typename _Tp>
    static auto get_labels_sfinae(const _Tp&, long) -> strvec_t
    {
        return strvec_t{ _Tp::get_label() };
    }

    template <typename _Tp>
    static strvec_t get_labels(const _Tp& _data)
    {
        return get_labels_sfinae(_data, 0);
    }

public:
    template <typename T>
    static strvec_t as_string_vec(const T& _data)
    {
        return strvec_t{ _data };
    }

    template <typename _Tp>
    static std::string as_string(const _Tp& _obj)
    {
        std::stringstream ss;
        ss << _obj;
        return ss.str();
    }

    template <typename... T, size_t... _Idx>
    static strvec_t as_string_vec(const std::tuple<T...>& _obj, index_sequence<_Idx...>)
    {
        using init_list_type = std::initializer_list<std::string>;
        auto&& ret           = init_list_type{ (as_string(std::get<_Idx>(_obj)))... };
        return strvec_t(ret);
    }

    template <typename... T>
    static strvec_t as_string_vec(const std::tuple<T...>& _obj)
    {
        constexpr size_t _N = sizeof...(T);
        return as_string_vec(_obj, make_index_sequence<_N>{});
    }

public:
    template <typename _Tp>
    static auto get_display_units_sfinae(const _Tp& _data, int)
        -> decltype(_data.display_unit_array(), strvec_t())
    {
        strvec_t _ret;
        for(const auto& itr : _data.display_unit_array())
            _ret.push_back(itr);
        return _ret;
    }

    template <typename _Tp>
    static auto get_display_units_sfinae(const _Tp&, long) -> strvec_t
    {
        return as_string_vec(_Tp::get_display_unit());
    }

    template <typename _Tp>
    static strvec_t get_display_units(const _Tp& _data)
    {
        return get_display_units_sfinae(_data, 0);
    }

public:
    using sizevector_t = std::vector<size_t>;

    template <typename _Tp>
    static sizevector_t get_widths(const _Tp& _data)
    {
        return get_widths_sfinae(_data, 0);
    }

    template <typename _Tp>
    static auto get_widths_sfinae(const _Tp& _data, int)
        -> decltype(_data.width_array(), sizevector_t())
    {
        return _data.width_array();
    }

    template <typename _Tp>
    static auto get_widths_sfinae(const _Tp&, long) -> sizevector_t
    {
        return sizevector_t{ _Tp::get_width() };
    }

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

    static bool is_empty(const std::string& obj) { return obj.empty(); }

    //----------------------------------------------------------------------------------//

    template <typename _Tp, typename... _Extra>
    static bool is_empty(const std::vector<_Tp, _Extra...>& obj)
    {
        for(const auto& itr : obj)
            if(!itr.empty())
                return false;
        return true;
    }

    //----------------------------------------------------------------------------------//

    template <template <typename...> class _Tuple, typename... _Tp>
    static bool is_empty(const _Tuple<_Tp...>& obj)
    {
        using init_list_type = std::initializer_list<int>;
        using input_type     = _Tuple<_Tp...>;

        constexpr size_t _N = sizeof...(_Tp);
        std::bitset<_N>  _bits;

        auto&& ret = init_list_type{ (
            _bits[index_of<_Tp, input_type>::value] =
                (std::get<index_of<_Tp, input_type>::value>(obj).empty()),
            0)... };
        consume_parameters(ret);
        return _bits.all();
    }

    //----------------------------------------------------------------------------------//

    template <bool _Enabled, typename _Arg, enable_if_t<(_Enabled == true), int> = 0>
    static void print_tag(std::ostream& os, const _Arg& _arg)
    {
        if(!is_empty(_arg))
            os << " " << _arg;
    }

    //----------------------------------------------------------------------------------//

    template <bool _Enabled, typename _Arg, enable_if_t<(_Enabled == false), int> = 0>
    static void print_tag(std::ostream&, const _Arg&)
    {}
};

//--------------------------------------------------------------------------------------//
/// \class base_printer
/// \brief invoked from the base class to provide default printing behavior
//
template <typename _Tp>
struct base_printer : public common_utils
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using widths_t   = std::vector<int64_t>;

    explicit base_printer(std::ostream& _os, const Type& _obj)
    {
        auto _value = static_cast<const Type&>(_obj).get_display();
        auto _disp  = Type::get_display_unit();
        auto _label = Type::get_label();
        auto _prec  = Type::get_precision();
        auto _width = Type::get_width();
        auto _flags = Type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;

        // check traits to see if we should print
        constexpr bool units_print = !trait::custom_unit_printing<Type>::value;
        constexpr bool label_print = !trait::custom_label_printing<Type>::value;

        print_tag<units_print>(ss_extra, _disp);
        print_tag<label_print>(ss_extra, _label);

        _os << ss_value.str() << ss_extra.str();
    }
};

//--------------------------------------------------------------------------------------//
/// \class base_printer
/// \brief invoked from the base class to provide default printing behavior
//
template <typename _Tp>
struct print_statistics : public common_utils
{
public:
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using widths_t   = std::vector<int64_t>;

public:
    template <typename _Up, typename _Vp>
    struct stats_enabled
    {
        static constexpr bool value =
            (trait::record_statistics<_Up>::value && !(std::is_same<_Vp, void>::value) &&
             !(std::is_same<_Vp, std::tuple<>>::value) &&
             !(std::is_same<_Vp, statistics<void>>::value) &&
             !(std::is_same<_Vp, statistics<std::tuple<>>>::value));
    };

public:
    template <typename _Self, template <typename> class _Sp, typename _Vp,
              typename _Up = _Tp, enable_if_t<(stats_enabled<_Up, _Vp>::value), int> = 0>
    print_statistics(const Type&, utility::stream& _os, const _Self&,
                     const _Sp<_Vp>& _stats, uint64_t)
    {
        bool use_mean   = get_env<bool>("TIMEMORY_PRINT_MEAN", true);
        bool use_min    = get_env<bool>("TIMEMORY_PRINT_MIN", true);
        bool use_max    = get_env<bool>("TIMEMORY_PRINT_MIN", true);
        bool use_var    = get_env<bool>("TIMEMORY_PRINT_VARIANCE", false);
        bool use_stddev = get_env<bool>("TIMEMORY_PRINT_STDDEV", true);

        if(use_mean)
            utility::write_entry(_os, "MEAN", _stats.get_mean());
        if(use_min)
            utility::write_entry(_os, "MIN", _stats.get_min());
        if(use_max)
            utility::write_entry(_os, "MAX", _stats.get_max());
        if(use_var)
            utility::write_entry(_os, "VAR", _stats.get_variance());
        if(use_stddev)
            utility::write_entry(_os, "STDDEV", _stats.get_stddev());
    }

    template <typename _Self, typename _Vp, typename _Up = _Tp,
              enable_if_t<!(stats_enabled<_Up, _Vp>::value), int> = 0>
    print_statistics(const Type&, utility::stream&, const _Self&, const _Vp&, uint64_t)
    {}

    template <typename _Self>
    print_statistics(const Type&, utility::stream&, const _Self&,
                     const statistics<std::tuple<>>&, uint64_t)
    {}

public:
    template <template <typename> class _Sp, typename _Vp, typename _Up = _Tp,
              enable_if_t<(stats_enabled<_Up, _Vp>::value), int> = 0>
    static void get_header(utility::stream& _os, const _Sp<_Vp>&)
    {
        bool use_mean   = get_env<bool>("TIMEMORY_PRINT_MEAN", true);
        bool use_min    = get_env<bool>("TIMEMORY_PRINT_MIN", true);
        bool use_max    = get_env<bool>("TIMEMORY_PRINT_MIN", true);
        bool use_var    = get_env<bool>("TIMEMORY_PRINT_VARIANCE", false);
        bool use_stddev = get_env<bool>("TIMEMORY_PRINT_STDDEV", true);

        auto _flags = _Tp::get_format_flags();
        auto _width = _Tp::get_width();
        auto _prec  = _Tp::get_precision();

        if(use_mean)
            utility::write_header(_os, "MEAN", _flags, _width, _prec);
        if(use_min)
            utility::write_header(_os, "MIN", _flags, _width, _prec);
        if(use_max)
            utility::write_header(_os, "MAX", _flags, _width, _prec);
        if(use_var)
            utility::write_header(_os, "VAR", _flags, _width, _prec);
        if(use_stddev)
            utility::write_header(_os, "STDDEV", _flags, _width, _prec);
    }

    template <typename _Vp, typename _Up = _Tp,
              enable_if_t<!(stats_enabled<_Up, _Vp>::value), int> = 0>
    static void get_header(utility::stream&, _Vp&)
    {}

    static void get_header(utility::stream&, const statistics<std::tuple<>>&) {}
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct print_header : public common_utils
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;
    using widths_t   = std::vector<int64_t>;

    //----------------------------------------------------------------------------------//
    // only if components are available
    //
    template <typename _Stats, typename _Up = _Tp,
              enable_if_t<(is_enabled<_Up>::value), char> = 0>
    print_header(const Type& _obj, utility::stream& _os, const _Stats& _stats)
    {
        auto _labels  = get_labels(_obj);
        auto _display = get_display_units(_obj);

        utility::write_header(_os, "LABEL");
        utility::write_header(_os, "COUNT");
        utility::write_header(_os, "DEPTH");

        auto _opzip = [](const std::string& _lhs, const std::string& _rhs) {
            return tim::apply<std::string>::join("", _lhs, " [", _rhs, "]");
        };

        auto ios_fixed = std::ios_base::fixed;
        auto ios_dec   = std::ios_base::dec;
        auto ios_showp = std::ios_base::showpoint;
        auto f_self    = ios_fixed | ios_dec | ios_showp;
        int  w_self    = 8;
        int  p_self    = 1;
        auto f_value   = _Tp::get_format_flags();
        auto w_value   = _Tp::get_width();
        auto p_value   = _Tp::get_precision();

        for(size_t i = 0; i < _labels.size(); ++i)
        {
            auto _label = _opzip(_labels.at(i), _display.at(i));
            utility::write_header(_os, _label, f_value, w_value, p_value);
            utility::write_header(_os, "% SELF", f_self, w_self, p_self);
            print_statistics<_Tp>::get_header(_os, _stats);
        }
    }

    template <typename... _Args, typename _Up = _Tp,
              enable_if_t<!(is_enabled<_Up>::value), char> = 0>
    print_header(const Type&, utility::stream&, _Args&&...)
    {}
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
            ss << '\n';
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
            ss << '\n';
        _os << ss.str();
    }

    template <typename _Vp, typename _Stats, typename _Up = _Tp,
              enable_if_t<(is_enabled<_Up>::value), char> = 0>
    print(const Type& _obj, utility::stream& _os, const string_t& _prefix, int64_t _laps,
          int64_t _depth, const _Vp& _self, const _Stats& _stats)
    {
        auto _opzip = [](const std::string& _lhs, const std::string& _rhs) {
            return tim::apply<std::string>::join("", _lhs, " [", _rhs, "]");
        };

        auto _labels = mpl::zip(_opzip, common_utils::get_labels(_obj),
                                common_utils::get_display_units(_obj));

        utility::write_entry(_os, "LABEL", _prefix);
        utility::write_entry(_os, "COUNT", _laps);
        utility::write_entry(_os, "DEPTH", _depth);

        utility::write_entry(_os, _labels, _obj.get());
        utility::write_entry(_os, "% SELF", _self);
        print_statistics<_Tp>(_obj, _os, _self, _stats, _laps);
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
struct echo_measurement<_Tp, false> : public common_utils
{
    template <typename... _Args>
    echo_measurement(_Args&&...)
    {}
};

template <typename _Tp>
struct echo_measurement<_Tp, true> : public common_utils
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    //----------------------------------------------------------------------------------//
    /// generate a name attribute
    ///
    template <typename... _Args>
    static string_t generate_name(const string_t& _prefix, string_t _unit,
                                  _Args&&... _args)
    {
        if(settings::dart_label())
        {
            return (_unit.length() > 0 && _unit != "%")
                       ? join("//", Type::get_label(), _unit)
                       : Type::get_label();
        }

        auto _extra = join("/", std::forward<_Args>(_args)...);
        auto _label = uppercase(Type::get_label());
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

    //------------------------------------------------------------------------------//
    //
    struct impl
    {
        template <typename _Tuple, size_t _Idx, size_t... _Nt, typename... _Args,
                  enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
        static std::string name_generator(const string_t& _prefix, _Tuple _units,
                                          _Args&&... _args)
        {
            return generate_name(_prefix, std::get<_Idx>(_units),
                                 std::forward<_Args>(_args)...);
        }

        template <typename _Tuple, size_t _Idx, size_t... _Nt, typename... _Args,
                  enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
        static std::string name_generator(const string_t& _prefix, _Tuple _units,
                                          _Args&&... _args)
        {
            return join(",",
                        name_generator<_Tuple, _Idx>(_prefix, _units,
                                                     std::forward<_Args>(_args)...),
                        name_generator<_Tuple, _Nt...>(_prefix, _units,
                                                       std::forward<_Args>(_args)...));
        }

        template <typename _Tuple, size_t... _Idx, typename... _Args>
        static std::string name_generator(const string_t& _prefix, _Tuple _units,
                                          _Args&&... _args, index_sequence<_Idx...>)
        {
            return name_generator<_Tuple, _Idx...>(_prefix, _units,
                                                   std::forward<_Args>(_args)...);
        }
    };

    //----------------------------------------------------------------------------------//
    /// generate a name attribute
    ///
    template <typename _Tuple, typename... _Args>
    static string_t generate_name(const string_t& _prefix, _Tuple _unit, _Args&&... _args)
    {
        constexpr size_t _N = std::tuple_size<_Tuple>::value;
        return impl::template name_generator<_Tuple>(
            _prefix, _unit, std::forward<_Args>(_args)..., make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//
    /// generate a name attribute
    ///
    template <typename _Type, typename... _Alloc, typename... _Args>
    static string_t generate_name(const string_t&               _prefix,
                                  std::vector<_Type, _Alloc...> _unit, _Args&&... _args)
    {
        string_t ret;
        for(auto& itr : _unit)
            return join(",", ret,
                        generate_name(_prefix, itr, std::forward<_Args>(_args)...));
        return ret;
    }

    //----------------------------------------------------------------------------------//
    /// generate a name attribute
    ///
    template <typename _Type, size_t _N, typename... _Args>
    static string_t generate_name(const string_t& _prefix, std::array<_Type, _N> _unit,
                                  _Args&&... _args)
    {
        string_t ret;
        for(auto& itr : _unit)
            return join(",", ret,
                        generate_name(_prefix, itr, std::forward<_Args>(_args)...));
        return ret;
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
