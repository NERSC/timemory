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
 * \file timemory/operations/declaration.hpp
 * \brief The declaration for the types for operations without definitions
 */

#pragma once

#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/utility.hpp"

#include <bitset>
#include <cstdio>
#include <iostream>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

//
//--------------------------------------------------------------------------------------//
//
#if !defined(SFINAE_WARNING)
#    if defined(DEBUG)
#        define SFINAE_WARNING(TYPE)                                                     \
            if(::tim::trait::is_available<TYPE>::value)                                  \
            {                                                                            \
                fprintf(stderr, "[%s@%s:%i]> Warning! SFINAE disabled for %s\n",         \
                        __FUNCTION__, __FILE__, __LINE__,                                \
                        ::tim::demangle<TYPE>().c_str());                                \
            }
#    else
#        define SFINAE_WARNING(...)
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct storage_initializer
/// \brief This provides an object that can initialize the storage opaquely, e.g.
/// \code
/// namespace
/// {
///     tim::storage_initializer storage = tim::storage_initalizer::get<T>();
/// }
///
struct TIMEMORY_OPERATIONS_DLL storage_initializer
{
    TIMEMORY_DEFAULT_OBJECT(storage_initializer)

    template <typename T>
    static storage_initializer get() TIMEMORY_VISIBILITY("default");
};
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
struct non_vexing
{};
//
//--------------------------------------------------------------------------------------//
//
/// \struct common_utils
/// \brief common string manipulation utilities
///
//
//--------------------------------------------------------------------------------------//
//
struct common_utils
{
    using attributes_t   = std::map<string_t, string_t>;
    using strset_t       = std::set<string_t>;
    using stringstream_t = std::stringstream;
    using strvec_t       = std::vector<string_t>;

public:
    template <typename Tp>
    static size_t get_distance(const Tp& _data)
    {
        return get_distance_sfinae(_data);
    }

    template <typename Tp>
    static auto get_distance_sfinae(const Tp& _data, int)
        -> decltype(std::distance(_data.begin(), _data.end()), size_t())
    {
        return std::distance(_data.begin(), _data.end());
    }

    template <typename Tp>
    static auto get_distance_sfinae(const Tp&, long) -> size_t
    {
        return size_t(1);
    }

    template <typename Tp>
    static auto get_distance_sfinae(const Tp& _data)
        -> decltype(get_distance_sfinae(_data, 0))
    {
        return get_distance_sfinae(_data, 0);
    }

public:
    template <typename Tp, enable_if_t<(std::is_arithmetic<Tp>::value), int> = 0>
    static Tp get_entry(const Tp& _data, size_t)
    {
        return _data;
    }

    template <typename Tp, enable_if_t<!(std::is_arithmetic<Tp>::value), int> = 0>
    static auto get_entry(const Tp& _data, size_t _idx)
        -> decltype(get_entry_sfinae_(_data, _idx))
    {
        return get_entry_sfinae_<Tp>(_data, _idx);
    }

    template <typename Tp, size_t Idx>
    static Tp get_entry(const Tp& _data, size_t)
    {
        return _data;
    }

    template <typename Tp>
    static auto get_entry_sfinae(const Tp& _data, int, size_t _idx)
        -> decltype(_data.begin(), typename Tp::value_type())
    {
        auto sz  = std::distance(_data.begin(), _data.end());
        auto n   = _idx % sz;
        auto itr = _data.begin();
        std::advance(itr, n);
        return *itr;
    }

    template <typename Tp>
    static Tp get_entry_sfinae(const Tp& _data, long, size_t)
    {
        return _data;
    }

    template <typename Tp>
    static auto get_entry_sfinae_(const Tp& _data, size_t _idx)
        -> decltype(get_entry_sfinae(_data, 0, _idx))
    {
        return get_entry_sfinae<Tp>(_data, 0, _idx);
    }

public:
    template <typename Tp, typename Wp, typename Pp>
    static void write(std::vector<std::stringstream*>& _os,
                      std::ios_base::fmtflags _format, const Tp& _data, const Wp& _width,
                      const Pp& _prec)
    {
        size_t num_data = get_distance(_data);

        for(size_t i = 0; i < num_data; ++i)
        {
            auto  _idata  = get_entry<Tp>(_data, i);
            auto  _iwidth = get_entry<Wp>(_width, i);
            auto  _iprec  = get_entry<Pp>(_prec, i);
            auto* ss      = new std::stringstream;
            ss->setf(_format);
            (*ss) << std::setw(_iwidth) << std::setprecision(_iprec) << _idata;
            _os.emplace_back(ss);
        }
    }

    template <typename... Tp, size_t... Idx, typename Wp, typename Pp>
    static void write(std::vector<std::stringstream*>& _os,
                      std::ios_base::fmtflags _format, const std::tuple<Tp...>& _data,
                      const Wp& _width, const Pp& _prec, index_sequence<Idx...>)
    {
        TIMEMORY_FOLD_EXPRESSION(
            write(_os, _format, std::get<Idx>(_data), _width, _prec));
    }

    template <typename... Tp, typename Wp, typename Pp>
    static void write(std::vector<std::stringstream*>& _os,
                      std::ios_base::fmtflags _format, const std::tuple<Tp...>& _data,
                      const Wp& _width, const Pp& _prec)
    {
        constexpr size_t N = sizeof...(Tp);
        write(_os, _format, _data, _width, _prec, make_index_sequence<N>{});
    }

public:
    template <typename Tp>
    static int64_t get_labels_size(const Tp& _data)
    {
        return get_labels_size_sfinae(_data, 0);
    }

    template <typename Tp>
    static auto get_labels_size_sfinae(const Tp& _data, int)
        -> decltype(_data.label_array(), int64_t())
    {
        return _data.label_array().size();
    }

    template <typename Tp>
    static auto get_labels_size_sfinae(const Tp&, long) -> int64_t
    {
        return 1;
    }

public:
    template <typename Tp>
    static strvec_t get_labels(const Tp& _data)
    {
        return get_labels_sfinae(_data, 0);
    }

    template <typename Tp>
    static auto get_labels_sfinae(const Tp& _data, int)
        -> decltype(_data.label_array(), strvec_t())
    {
        strvec_t _ret;
        for(const auto& itr : _data.label_array())
            _ret.push_back(itr);
        return _ret;
    }

    template <typename Tp>
    static auto get_labels_sfinae(const Tp&, long) -> strvec_t
    {
        return strvec_t{ Tp::get_label() };
    }

public:
    template <typename T>
    static strvec_t as_string_vec(const T& _data)
    {
        return strvec_t{ _data };
    }

    template <typename Tp>
    static std::string as_string(const Tp& _obj)
    {
        std::stringstream ss;
        ss << _obj;
        return ss.str();
    }

    template <typename... T, size_t... Idx>
    static strvec_t as_string_vec(const std::tuple<T...>& _obj, index_sequence<Idx...>)
    {
        using init_list_type = std::initializer_list<std::string>;
        auto&& ret           = init_list_type{ (as_string(std::get<Idx>(_obj)))... };
        return strvec_t(ret);
    }

    template <typename... T>
    static strvec_t as_string_vec(const std::tuple<T...>& _obj)
    {
        constexpr size_t N = sizeof...(T);
        return as_string_vec(_obj, make_index_sequence<N>{});
    }

public:
    template <typename Tp>
    static auto get_display_units_sfinae(const Tp& _data, int)
        -> decltype(_data.display_unit_array(), strvec_t())
    {
        strvec_t _ret;
        for(const auto& itr : _data.display_unit_array())
            _ret.push_back(itr);
        return _ret;
    }

    template <typename Tp>
    static auto get_display_units_sfinae(const Tp&, long) -> strvec_t
    {
        return as_string_vec(Tp::get_display_unit());
    }

    template <typename Tp>
    static strvec_t get_display_units(const Tp& _data)
    {
        return get_display_units_sfinae(_data, 0);
    }

public:
    using sizevector_t = std::vector<size_t>;

    template <typename Tp>
    static sizevector_t get_widths(const Tp& _data)
    {
        return get_widths_sfinae(_data, 0);
    }

    template <typename Tp>
    static auto get_widths_sfinae(const Tp& _data, int)
        -> decltype(_data.width_array(), sizevector_t())
    {
        return _data.width_array();
    }

    template <typename Tp>
    static auto get_widths_sfinae(const Tp&, long) -> sizevector_t
    {
        return sizevector_t{ Tp::get_width() };
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
    template <typename... Args>
    static string_t join(const std::string& _delim, Args&&... _args)
    {
        return apply<string_t>::join(_delim, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    static bool is_empty(const std::string& obj) { return obj.empty(); }

    //----------------------------------------------------------------------------------//

    template <typename Tp, typename... _Extra>
    static bool is_empty(const std::vector<Tp, _Extra...>& obj)
    {
        for(const auto& itr : obj)
            if(!itr.empty())
                return false;
        return true;
    }

    //----------------------------------------------------------------------------------//

    template <template <typename...> class Tuple, typename... Tp>
    static bool is_empty(const Tuple<Tp...>& obj)
    {
        using input_type   = Tuple<Tp...>;
        constexpr size_t N = sizeof...(Tp);
        std::bitset<N>   _bits;
        TIMEMORY_FOLD_EXPRESSION(
            _bits[index_of<Tp, input_type>::value] =
                (std::get<index_of<Tp, input_type>::value>(obj).empty()));
        return _bits.all();
    }

    //----------------------------------------------------------------------------------//

    template <bool _Enabled, typename Arg, enable_if_t<(_Enabled == true), int> = 0>
    static void print_tag(std::ostream& os, const Arg& _arg)
    {
        if(!is_empty(_arg))
            os << " " << _arg;
    }

    //----------------------------------------------------------------------------------//

    template <bool _Enabled, typename Arg, enable_if_t<(_Enabled == false), int> = 0>
    static void print_tag(std::ostream&, const Arg&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct init_storage
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using string_t   = std::string;
    using this_type  = init_storage<Tp>;
    using pointer_t  = tim::base::storage*;
    using get_type   = std::tuple<pointer_t, bool, bool, bool>;

    template <typename Up = Tp, enable_if_t<(trait::is_available<Up>::value), char> = 0>
    init_storage();

    template <typename Up = Tp, enable_if_t<!(trait::is_available<Up>::value), char> = 0>
    init_storage();

    template <typename U = Tp, typename V = typename U::value_type,
              enable_if_t<(implements_storage<U, V>::value), int> = 0>
    static get_type get();

    template <typename U = Tp, typename V = typename U::value_type,
              enable_if_t<!(implements_storage<U, V>::value), int> = 0>
    static get_type get();

    static void init();
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
