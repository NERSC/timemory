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

/** \file utility/stream.hpp
 * \headerfile utility/stream.hpp "timemory/utility/stream.hpp"
 * Provides a simple stream type that generates a vector of strings for column alignment
 *
 */

#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

namespace tim
{
namespace utility
{
namespace base
{
//--------------------------------------------------------------------------------------//
//
struct stream_entry
{
    using string_t       = std::string;
    using stringstream_t = std::stringstream;
    using format_flags   = std::ios_base::fmtflags;

    stream_entry(int _row = -1, int _col = -1, format_flags _fmt = {}, int _width = 0,
                 int _prec = 0, bool _center = false)
    : m_center(_center)
    , m_row(_row)
    , m_column(_col)
    , m_width(_width)
    , m_precision(_prec)
    , m_format(_fmt)
    , m_value("")
    {}

    ~stream_entry()                   = default;
    stream_entry(const stream_entry&) = default;
    stream_entry(stream_entry&&)      = default;
    stream_entry& operator=(const stream_entry&) = default;
    stream_entry& operator=(stream_entry&&) = default;

    string_t get() const { return m_value; }

    bool         center() const { return m_center; }
    int          row() const { return m_row; }
    int          width() const { return m_width; }
    int          column() const { return m_column; }
    int          precision() const { return m_precision; }
    format_flags flags() const { return m_format; }

    void center(bool v) { m_center = v; }
    void row(int v) { m_row = v; }
    void width(int v) { m_width = v; }
    void column(int v) { m_column = v; }
    void precision(int v) { m_precision = v; }
    void setf(format_flags v) { m_format = v; }

    void operator()(const string_t& val) { m_value = val; }

    template <typename _Tp>
    void operator()(const _Tp& val)
    {
        stringstream_t ss;
        ss.setf(m_format);
        ss << std::setprecision(m_precision) << val;
        m_value = ss.str();
    }

    friend bool operator<(const stream_entry& lhs, const stream_entry& rhs)
    {
        return (lhs.row() == rhs.row()) ? (lhs.column() < rhs.column())
                                        : (lhs.row() < rhs.row());
    }

protected:
    bool         m_center    = false;
    int          m_row       = 0;
    int          m_column    = 0;
    int          m_width     = 0;
    int          m_precision = 0;
    format_flags m_format    = {};
    string_t     m_value     = "";
};

//--------------------------------------------------------------------------------------//

template <typename _Stream, typename _Tp>
static void
write_entry(_Stream& ss, const _Tp& obj)
{
    using stringstream_t = std::stringstream;

    auto itr = obj.get();

    if(obj.row() == 0 || obj.center())
    {
        int _w     = obj.width();
        int _i     = itr.length();
        int _whalf = _w / 2;
        int _ihalf = (_i + 1) / 2;
        int _wrem  = (_whalf - _ihalf);
        _wrem      = std::max<int>(_wrem, 0);
        if(_i + _wrem > _w - 3)
            _wrem = _w - 3 - _i;
        stringstream_t ssbeg;
        ssbeg << std::setw(_wrem) << "" << itr;
        ss << std::left << std::setw(_w - 2) << ssbeg.str();
    }
    else
    {
        if(obj.column() == 0)
        {
            stringstream_t _ss;
            _ss << std::left << itr;
            int remain = obj.width() - _ss.str().length() - 2;
            ss << _ss.str() << std::setw(remain) << "";
        }
        else
        {
            ss << std::setw(obj.width() - 2) << std::right << itr;
        }
    }
}

//--------------------------------------------------------------------------------------//

}  // namespace base

//======================================================================================//

struct header : base::stream_entry
{
    header()              = default;
    ~header()             = default;
    header(const header&) = default;
    header(header&&)      = default;
    header& operator=(const header&) = default;
    header& operator=(header&&) = default;

    explicit header(const std::string& _val, format_flags _fmt = {}, int _width = 0,
                    int _prec = 0, bool _center = true)
    : base::stream_entry(0, -1, _fmt, _width, _prec, _center)
    {
        base::stream_entry::operator()(_val);
    }

    template <typename _Tp>
    explicit header(_Tp&& _val, format_flags _fmt = {}, int _width = 0, int _prec = 0,
                    bool _center = true)
    : base::stream_entry(0, -1, _fmt, _width, _prec, _center)
    {
        base::stream_entry::operator()(std::forward<_Tp>(_val));
    }
};

//--------------------------------------------------------------------------------------//

struct entry : base::stream_entry
{
    template <typename _Tp>
    explicit entry(_Tp&& _val, header& _hdr, bool _center = false)
    : base::stream_entry(_hdr)
    , m_hdr(&_hdr)
    {
        m_center = _center;
        base::stream_entry::operator()(std::forward<_Tp>(_val));
    }

    entry(const entry& _rhs)
    : base::stream_entry(_rhs)
    , m_hdr(_rhs.m_hdr)
    {}

    ~entry()       = default;
    entry(entry&&) = default;
    entry& operator=(const entry&) = default;
    entry& operator=(entry&&) = default;

    int          width() const { return m_hdr->width(); }
    int          precision() const { return m_hdr->precision(); }
    format_flags flags() const { return m_hdr->flags(); }

    void width(int v) { m_hdr->width(v); }
    void precision(int v) { m_hdr->precision(v); }
    void setf(format_flags v) { m_hdr->setf(v); }

    const header& get_header() const { return *m_hdr; }
    header&       get_header() { return *m_hdr; }

private:
    header* m_hdr = nullptr;
};

//--------------------------------------------------------------------------------------//

struct stream
{
    template <typename K, typename M>
    using map_t = std::map<K, M>;

    template <typename K, typename M>
    using pair_t = std::pair<K, M>;

    template <typename T>
    using vector_t = std::vector<T>;

    using string_t       = std::string;
    using stringstream_t = std::stringstream;
    using format_flags   = std::ios_base::fmtflags;
    using order_map_t    = vector_t<string_t>;

    using header_col_t = vector_t<header>;
    using entry_col_t  = vector_t<entry>;

    using header_pair_t = pair_t<string_t, header_col_t>;
    using entry_pair_t  = pair_t<string_t, entry_col_t>;

    using header_map_t = vector_t<header_pair_t>;
    using entry_map_t  = vector_t<entry_pair_t>;

public:
    explicit stream(char _delim = '|', char _fill = '-', format_flags _fmt = {},
                    int _width = 0, int _prec = 0, bool _center = false)
    : m_center(_center)
    , m_fill(_fill)
    , m_delim(_delim)
    , m_width(_width)
    , m_precision(_prec)
    , m_rows(0)
    , m_format(_fmt)
    {}

    bool         center() const { return m_center; }
    int          precision() const { return m_precision; }
    int          width() const { return m_width; }
    char         delim() const { return m_delim; }
    format_flags flags() const { return m_format; }

    void center(bool v) { m_center = v; }
    void precision(int v) { m_precision = v; }
    void width(int v) { m_width = v; }
    void delim(char v) { m_delim = v; }
    void setf(format_flags v) { m_format = v; }

    void set_name(string_t v) { m_name = v; }

    static int64_t index(const string_t& _val, const vector_t<string_t>& _obj)
    {
        for(size_t i = 0; i < _obj.size(); ++i)
            if(_obj.at(i) == _val)
                return static_cast<int64_t>(i);
        return -1;
    }

    static int64_t insert(const string_t& _val, vector_t<string_t>& _obj)
    {
        auto idx = index(_val, _obj);
        if(idx < 0)
        {
            idx = _obj.size();
            _obj.push_back(_val);
            if(settings::debug())
                printf("> inserted '%s'...\n", _val.c_str());
        }
        return idx;
    }

    template <typename _Tp>
    static int64_t index(const string_t&                                  _val,
                         const vector_t<pair_t<string_t, vector_t<_Tp>>>& _obj)
    {
        for(size_t i = 0; i < _obj.size(); ++i)
            if(_obj.at(i).first == _val)
                return static_cast<int64_t>(i);
        return -1;
    }

    template <typename _Tp>
    static int64_t insert(const string_t&                            _val,
                          vector_t<pair_t<string_t, vector_t<_Tp>>>& _obj)
    {
        auto idx = index(_val, _obj);
        if(idx < 0)
        {
            idx = _obj.size();
            _obj.resize(_obj.size() + 1);
            _obj[idx].first = _val;
            if(settings::debug())
                printf("[%s]> inserted '%s'...\n", demangle<_Tp>().c_str(), _val.c_str());
        }
        return idx;
    }

    void operator()(header _hdr)
    {
        if(_hdr.get().empty())
            throw std::runtime_error("Header has no value");

        auto _w = std::max<int>(m_width, _hdr.get().length() + 2);
        _hdr.width(_w);

        m_order.push_back(m_name);
        auto _h = insert(m_name, m_headers);
        auto _n = m_headers[_h].second.size();
        _hdr.center(true);
        _hdr.row(0);
        _hdr.column(_n);
        m_headers[_h].second.push_back(_hdr);
    }

    void operator()(entry _obj)
    {
        if(_obj.get().empty())
            throw std::runtime_error("Entry has no value");

        auto _w = std::max<int>(m_width, _obj.get().length() + 2);
        _w      = std::max<int>(_w, _obj.get_header().width());

        _obj.width(_w);
        _obj.get_header().width(_w);

        auto _o = index(m_name, m_order);
        if(_o < 0)
            throw std::runtime_error(string_t("Missing entry for ") + m_name);

        auto _r = insert(m_name, m_entries);
        _obj.center(false);
        _obj.row(m_rows + 1);
        _obj.column(m_cols);
        m_entries[_r].second.push_back(_obj);
        ++m_cols;
    }

    friend std::ostream& operator<<(std::ostream& os, const stream& obj)
    {
        // return if completely empty
        if(obj.m_headers.empty())
            return os;

        // return if not entries
        if(obj.m_entries.empty())
            return os;

        stringstream_t       ss;
        map_t<string_t, int> offset;

        obj.write_separator(ss);

        for(const auto& itr : obj.m_order)
        {
            stringstream_t _ss;
            auto           _key    = itr;
            auto           _offset = offset[_key]++;
            auto           _idx    = index(_key, obj.m_headers);
            if(_idx < 0 ||
               (_idx >= 0 && !(_offset < (int) obj.m_headers[_idx].second.size())))
            {
                throw std::runtime_error("Error! indexing issue!");
            }
            else
            {
                const auto& hitr = obj.m_headers[_idx].second.at(_offset);
                base::write_entry(_ss, hitr);
            }
            ss << obj.delim() << ' ' << _ss.str() << ' ';
        }

        // end the line
        ss << obj.delim() << '\n';

        obj.write_separator(ss);

        offset.clear();

        for(int i = 0; i < obj.m_rows; ++i)
        {
            for(const auto& itr : obj.m_order)
            {
                stringstream_t _ss;
                auto           _key    = itr;
                auto           _offset = offset[_key]++;

                auto _hidx = index(_key, obj.m_headers);
                auto _eidx = index(_key, obj.m_entries);

                if(_eidx < 0 && _hidx >= 0)
                {
                    const auto& _hitr  = obj.m_headers[_hidx].second;
                    auto        _hsize = _hitr.size();
                    const auto& _hdr   = _hitr.at(_offset % _hsize);
                    ss << obj.delim() << ' ' << std::setw(_hdr.width() - 2) << "" << ' ';
                    continue;
                }

                assert(_hidx >= 0);
                assert(_eidx >= 0);

                const auto& _eitr  = obj.m_entries[_eidx].second;
                auto        _esize = _eitr.size();
                const auto& _itr   = _eitr.at(_offset % _esize);

                base::write_entry(_ss, _itr);
                ss << obj.delim() << ' ' << _ss.str() << ' ';
            }
            ss << obj.m_delim << '\n';

            if((i + 1) < obj.m_rows && (i % 10) == 9)
                obj.write_separator(ss);
        }

        obj.write_separator(ss);

        os << ss.str();
        return os;
    }

    template <typename _Stream>
    void write_separator(_Stream& os) const
    {
        map_t<string_t, int> offset;
        stringstream_t       ss;
        ss.fill(m_fill);

        for(const auto& _key : m_order)
        {
            stringstream_t _ss;
            auto           _offset = offset[_key]++;
            auto           _hidx   = index(_key, m_headers);
            assert(_hidx >= 0);
            const auto& _hitr  = m_headers[_hidx].second;
            auto        _hsize = _hitr.size();
            const auto& _hdr   = _hitr.at(_offset % _hsize);
            auto        _w     = _hdr.width();
            ss << m_delim << std::setw(_w) << "";
        }

        ss << m_delim << '\n';
        os << ss.str();
    }

    template <typename... _Tp, template <typename...> class _Tuple, size_t... _Idx>
    static void write(stream&, const _Tuple<_Tp...>&, index_sequence<_Idx...>);

    void clear()
    {
        m_name = "";
        m_headers.clear();
        m_entries.clear();
    }

    header& get_header(const string_t& _key, int64_t _n)
    {
        auto idx = index(_key, m_headers);
        if(idx < 0)
        {
            stringstream_t ss;
            ss << "Missing header '" << _key << "'";
            throw std::runtime_error(ss.str());
        }

        if(!(_n < (int64_t) m_headers[idx].second.size()))
        {
            auto _size = m_headers[idx].second.size();
            return m_headers[idx].second[_n % _size];
        }

        return m_headers[idx].second[_n];
    }

    int add_row()
    {
        m_cols = 0;
        return ++m_rows;
    }

private:
    bool         m_center    = false;
    char         m_fill      = '-';
    char         m_delim     = '|';
    int          m_width     = 0;
    int          m_precision = 0;
    int          m_rows      = 0;
    int          m_cols      = 0;
    format_flags m_format    = {};
    string_t     m_name      = "";
    header_map_t m_headers   = {};
    entry_map_t  m_entries   = {};
    order_map_t  m_order     = {};
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct header_stream
{
    using format_flags = std::ios_base::fmtflags;

    header_stream(format_flags _fmt, int _width, int _prec, bool _center)
    : m_center(_center)
    , m_width(_width)
    , m_precision(_prec)
    , m_format(_fmt)
    {}

    template <typename _Stream>
    _Stream& operator()(_Stream& _os, const _Tp& _obj)
    {
        _os << header(_obj, m_format, m_width, m_precision, m_center);
        return _os;
    }

    bool         m_center;
    int          m_width;
    int          m_precision;
    format_flags m_format;
};

//--------------------------------------------------------------------------------------//

template <typename... _Args>
void
write_header(stream& _os, const std::string& _label, _Args&&... _args)
{
    _os.set_name(_label);
    _os(header(_label, std::forward<_Args>(_args)...));
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
write_entry(stream& _os, const std::string& _label, const _Tp& _value)
{
    _os.set_name(_label);
    _os(entry(_value, _os.get_header(_label, 0)));
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
write_entry(stream& _os, const std::vector<std::string>& _label, const _Tp& _value)
{
    write_entry(_os, _label.front(), _value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Up>
void
write_entry(stream& _os, const std::string& _label, const std::pair<_Tp, _Up>& _value)
{
    write_entry(_os, _label, _value.first);
    write_entry(_os, _label, _value.second);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Up>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::pair<_Tp, _Up>& _value)
{
    size_t _L = _labels.size();
    write_entry(_os, _labels.at(0), _value.first);
    write_entry(_os, _labels.at(1 % _L), _value.second);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Alloc>
void
write_entry(stream& _os, const std::string& _label,
            const std::vector<_Tp, _Alloc...>& _values)
{
    for(const auto& itr : _values)
        write_entry(_os, _label, itr);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Alloc>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::vector<_Tp, _Alloc...>& _values)
{
    size_t _L = _labels.size();
    size_t _N = _values.size();
    for(size_t i = 0; i < _N; ++i)
        write_entry(_os, _labels.at(i % _L), _values.at(i));
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
void
write_entry(stream& _os, const std::string& _label, const std::array<_Tp, _N>& _values)
{
    for(const auto& itr : _values)
        write_entry(_os, _label, itr);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::array<_Tp, _N>& _values)
{
    size_t _L = _labels.size();
    for(size_t i = 0; i < _N; ++i)
        write_entry(_os, _labels.at(i % _L), _values.at(i));
}

//--------------------------------------------------------------------------------------//

template <typename... _Types, size_t... _Idx>
void
write_entry(stream& _os, const std::string& _label, const std::tuple<_Types...>& _values,
            index_sequence<_Idx...>)
{
    using init_list_type = std::initializer_list<int>;
    auto&& ret =
        init_list_type{ (write_entry(_os, _label, std::get<_Idx>(_values)), 0)... };
    consume_parameters(ret);
}

//--------------------------------------------------------------------------------------//

template <typename... _Types, size_t... _Idx>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::tuple<_Types...>& _values, index_sequence<_Idx...>)
{
    using init_list_type = std::initializer_list<int>;
    size_t _L            = _labels.size();
    auto&& ret           = init_list_type{ (
        write_entry(_os, _labels.at(_Idx % _L), std::get<_Idx>(_values)), 0)... };
    consume_parameters(ret);
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
void
write_entry(stream& _os, const std::string& _labels, const std::tuple<_Types...>& _values)
{
    constexpr size_t _N = sizeof...(_Types);
    write_entry(_os, _labels, _values, make_index_sequence<_N>{});
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::tuple<_Types...>& _values)
{
    constexpr size_t _N = sizeof...(_Types);
    write_entry(_os, _labels, _values, make_index_sequence<_N>{});
}

//--------------------------------------------------------------------------------------//

}  // namespace utility

//--------------------------------------------------------------------------------------//

}  // namespace tim
