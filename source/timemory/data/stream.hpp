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

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
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

    explicit stream_entry(int _row = -1, int _col = -1, format_flags _fmt = {},
                          int _width = 0, int _prec = 0, bool _center = false)
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
    bool         left() const { return m_left; }
    int          row() const { return m_row; }
    int          width() const { return m_width; }
    int          column() const { return m_column; }
    int          precision() const { return m_precision; }
    format_flags flags() const { return m_format; }

    void center(bool v) { m_center = v; }
    void left(bool v) { m_left = v; }
    void row(int v) { m_row = v; }
    void width(int v) { m_width = v; }
    void column(int v) { m_column = v; }
    void precision(int v) { m_precision = v; }
    void setf(format_flags v) { m_format = v; }

    void operator()(const string_t& val) { m_value = val; }

    template <typename Tp>
    void construct(const Tp& val)
    {
        stringstream_t ss;
        ss.setf(m_format);
        ss << std::setprecision(m_precision) << val;
        m_value = ss.str();
        if(settings::max_width() > 0 && m_value.length() > (size_t) settings::max_width())
        {
            //
            //  don't truncate and add ellipsis if max width is really small
            //
            if(settings::max_width() > 20)
            {
                m_value = m_value.substr(0, settings::max_width() - 3);
                m_value += "...";
            }
            else
            {
                m_value = m_value.substr(0, settings::max_width());
            }
        }
    }

    friend bool operator<(const stream_entry& lhs, const stream_entry& rhs)
    {
        return (lhs.row() == rhs.row()) ? (lhs.column() < rhs.column())
                                        : (lhs.row() < rhs.row());
    }

protected:
    bool         m_center    = false;
    bool         m_left      = false;
    int          m_row       = 0;
    int          m_column    = 0;
    int          m_width     = 0;
    int          m_precision = 0;
    format_flags m_format    = {};
    string_t     m_value     = "";
};

//--------------------------------------------------------------------------------------//

template <typename StreamT, typename Tp>
static void
write_entry(StreamT& ss, const Tp& obj)
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
        // e.g. 4 leading spaces, 2 spaces at end
        if(((_wrem - itr.length()) - (_w - 2)) > 1)
            _wrem -= 1;
        stringstream_t ssbeg;
        ssbeg << std::setw(_wrem) << "" << itr;
        ss << std::left << std::setw(_w - 2) << ssbeg.str();
    }
    else
    {
        if(obj.column() == 0 || obj.left())
        {
            stringstream_t _ss;
            _ss << std::left << itr;
            int remain = obj.width() - _ss.str().length() - 2;
            ss << _ss.str() << std::setw(remain) << "";
        }
        else
        {
            ss << std::right << std::setw(obj.width() - 2) << itr;
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
        base::stream_entry::construct(_val);
    }

    template <typename Tp>
    explicit header(const Tp& _val, format_flags _fmt, int _width, int _prec,
                    bool _center = true)
    : base::stream_entry(0, -1, _fmt, _width, _prec, _center)
    {
        base::stream_entry::construct(std::forward<Tp>(_val));
    }
};

//--------------------------------------------------------------------------------------//

struct entry : base::stream_entry
{
    template <typename Tp>
    explicit entry(Tp&& _val, header& _hdr, bool _center = false, bool _left = false)
    : base::stream_entry(_hdr)
    , m_hdr(&_hdr)
    , m_permit_empty(false)
    {
        m_center = _center;
        m_left   = _left;
        base::stream_entry::construct(std::forward<Tp>(_val));
    }

    explicit entry(const std::string& _val, header& _hdr, bool _center = false,
                   bool _left = true)
    : base::stream_entry(_hdr)
    , m_hdr(&_hdr)
    , m_permit_empty(true)
    {
        m_center = _center;
        m_left   = _left;
        base::stream_entry::construct(_val);
    }

    entry(const entry& _rhs)
    : base::stream_entry(_rhs)
    , m_hdr(_rhs.m_hdr)
    {}

    ~entry()       = default;
    entry(entry&&) = default;
    entry& operator=(const entry&) = default;
    entry& operator=(entry&&) = default;

    bool         permit_empty() const { return m_permit_empty; }
    int          width() const { return m_hdr->width(); }
    int          precision() const { return m_hdr->precision(); }
    format_flags flags() const { return m_hdr->flags(); }

    void permit_empty(bool v) { m_permit_empty = v; }
    void width(int v) { m_hdr->width(v); }
    void precision(int v) { m_hdr->precision(v); }
    void setf(format_flags v) { m_hdr->setf(v); }

    const header& get_header() const { return *m_hdr; }
    header&       get_header() { return *m_hdr; }

private:
    header* m_hdr          = nullptr;
    bool    m_permit_empty = false;
};

//--------------------------------------------------------------------------------------//

struct stream
{
    template <typename T>
    using set_t = std::set<T>;

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
    using break_set_t  = set_t<int>;

public:
    explicit stream(char _delim = '|', char _fill = '-', format_flags _fmt = {},
                    int _width = 0, int _prec = 0, bool _center = false)
    : m_center(_center)
    , m_fill(_fill)
    , m_delim(_delim)
    , m_width(_width)
    , m_precision(_prec)
    , m_rows(0)
    , m_cols(0)
    , m_prefix_begin(0)
    , m_prefix_end(0)
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

    // NOLINTNEXTLINE
    void set_name(string_t v) { m_name = v; }
    // NOLINTNEXTLINE
    void set_banner(string_t v) { m_banner = v; }

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

    template <typename Tp>
    static int64_t index(const string_t&                                 _val,
                         const vector_t<pair_t<string_t, vector_t<Tp>>>& _obj)
    {
        for(size_t i = 0; i < _obj.size(); ++i)
            if(_obj.at(i).first == _val)
                return static_cast<int64_t>(i);
        return -1;
    }

    template <typename Tp>
    static int64_t insert(const string_t&                           _val,
                          vector_t<pair_t<string_t, vector_t<Tp>>>& _obj)
    {
        auto idx = index(_val, _obj);
        if(idx < 0)
        {
            idx = _obj.size();
            _obj.resize(_obj.size() + 1);
            _obj[idx].first = _val;
            if(settings::debug())
                printf("[%s]> inserted '%s'...\n", demangle<Tp>().c_str(), _val.c_str());
        }
        return idx;
    }

    void set_prefix_begin(int val = -1)
    {
        m_prefix_begin = (val < 0) ? ((int) m_order.size()) : val;
    }

    void set_prefix_end(int val = -1)
    {
        m_prefix_end = (val < 0) ? ((int) m_order.size()) : val;
    }

    void insert_break(int val = -1)
    {
        m_break.insert((val < 0) ? ((int) m_order.size()) : val);
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
        if(_obj.get().empty() && !_obj.permit_empty())
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

        obj.write_banner(ss);

        obj.write_separator(ss, '-');

        int64_t norder_col = 0;
        for(const auto& itr : obj.m_order)
        {
            int64_t col = ++norder_col;

            stringstream_t _ss;
            // NOLINTNEXTLINE
            auto _key    = itr;
            auto _offset = offset[_key]++;
            auto _idx    = index(_key, obj.m_headers);
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

            if(obj.m_break.count(col) > 0)
                break;
        }

        // end the line
        ss << obj.delim() << '\n';

        obj.write_separator(ss, obj.m_delim);

        auto write_empty = [&](stringstream_t& _ss, int64_t _hidx, int64_t _offset) {
            const auto& _hitr  = obj.m_headers[_hidx].second;
            auto        _hsize = _hitr.size();
            const auto& _hdr   = _hitr.at(_offset % _hsize);
            _ss << obj.delim() << ' ' << std::setw(_hdr.width() - 2) << "" << ' ';
        };

        offset.clear();

        for(int i = 0; i < obj.m_rows; ++i)
        {
            bool just_broke = false;
            norder_col      = 0;
            for(const auto& itr : obj.m_order)
            {
                just_broke  = false;
                int64_t col = ++norder_col;

                stringstream_t _ss;
                auto           _key    = itr;  // NOLINT
                auto           _offset = offset[_key]++;

                auto _hidx = index(_key, obj.m_headers);
                auto _eidx = index(_key, obj.m_entries);

                if(_eidx < 0 && _hidx >= 0)
                {
                    write_empty(ss, _hidx, _offset);
                }
                else
                {
                    assert(_hidx >= 0);
                    assert(_eidx >= 0);

                    const auto& _eitr  = obj.m_entries[_eidx].second;
                    auto        _esize = _eitr.size();
                    const auto& _itr   = _eitr.at(_offset % _esize);

                    base::write_entry(_ss, _itr);
                    ss << obj.delim() << ' ' << _ss.str() << ' ';
                }

                // printf("column: %i, order size: %i, count: %i\n", col,
                // obj.m_order.size(),
                //       obj.m_break.count(col));
                if(col < (int64_t) obj.m_order.size() && obj.m_break.count(col) > 0)
                {
                    ss << obj.m_delim << '\n';
                    just_broke = true;
                    for(auto j = obj.m_prefix_begin; j < obj.m_prefix_end; ++j)
                        write_empty(ss, j, 0);
                }
            }
            if(!just_broke)
                ss << obj.m_delim << '\n';

            if((i + 1) < obj.m_rows && (i % 10) == 9)
                obj.write_separator(ss, obj.m_delim);
        }

        obj.write_separator(ss, '-');

        os << ss.str();
        return os;
    }

    template <typename StreamT>
    void write_separator(StreamT& os, char _delim) const
    {
        map_t<string_t, int> offset;
        stringstream_t       ss;
        ss.fill(m_fill);

        int64_t norder_col = 0;
        for(const auto& _key : m_order)
        {
            int64_t        col = ++norder_col;
            stringstream_t _ss;
            auto           _offset = offset[_key]++;
            auto           _hidx   = index(_key, m_headers);
            assert(_hidx >= 0);
            const auto& _hitr  = m_headers[_hidx].second;
            auto        _hsize = _hitr.size();
            const auto& _hdr   = _hitr.at(_offset % _hsize);
            auto        _w     = _hdr.width();
            if(col == 1)
                ss << m_delim << std::setw(_w) << "";
            else
                ss << _delim << std::setw(_w) << "";
            if(m_break.count(col) > 0)
                break;
        }

        ss << m_delim << '\n';
        os << ss.str();
    }

    template <typename StreamT>
    void write_banner(StreamT& os) const
    {
        if(m_banner.length() == 0)
            return;

        write_separator(os, '-');

        map_t<string_t, int> offset;
        stringstream_t       ss;

        int64_t tot_w      = 0;
        int64_t norder_col = 0;
        for(const auto& _key : m_order)
        {
            int64_t col     = ++norder_col;
            auto    _offset = offset[_key]++;
            auto    _hidx   = index(_key, m_headers);
            assert(_hidx >= 0);
            const auto& _hitr  = m_headers[_hidx].second;
            auto        _hsize = _hitr.size();
            const auto& _hdr   = _hitr.at(_offset % _hsize);
            tot_w += _hdr.width() + 1;
            if(m_break.count(col) > 0)
                break;
        }

        auto obeg = tot_w / 2;
        obeg -= m_banner.length() / 2;
        obeg += m_banner.length();
        auto oend = tot_w - obeg;

        ss << m_delim << std::setw(obeg) << std::right << m_banner << std::setw(oend)
           << std::right << m_delim << '\n';
        os << ss.str();
    }

    template <typename... Tp, template <typename...> class _Tuple, size_t... Idx>
    static void write(stream&, const _Tuple<Tp...>&, index_sequence<Idx...>);

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

    /// \fn stream::add_row()
    /// \brief indicate that a row of data has been finished
    int add_row()
    {
        m_cols = 0;
        return ++m_rows;
    }

    /// \fn stream::sort(sorter, keys, exclude)
    /// \brief Provide a \param sorter functor that operates on all or a specific
    /// set of header keys. If \param keys is empty, all header keys are sorted.
    /// If \param keys is non-empty, the sorted keys are placed at the front of the
    /// container and any remaining keys not list in \param exclude will be added to the
    /// end of the container in the order consistent with the origial construction
    void sort(const std::function<bool(const std::string&, const std::string&)>& sorter,
              std::vector<std::string>     keys    = {},
              const std::set<std::string>& exclude = {})
    {
        // if no keys were provided, add all of them
        if(keys.empty())
            for(const auto& itr : m_order)
                keys.push_back(itr);

        // the new headers
        order_map_t _order{};

        // sort the keys
        std::sort(keys.begin(), keys.end(), sorter);

        // generate the new layout in the order specified
        for(const auto& itr : keys)
        {
            bool found = false;
            for(auto hitr = m_order.begin(); hitr != m_order.end(); ++hitr)
            {
                if(*hitr == itr)
                {
                    _order.push_back(*hitr);
                    // remove entry
                    m_order.erase(hitr);
                    found = true;
                    break;
                }
            }
            if(!found)
                PRINT_HERE("Warning! Expected header tag '%s' not found when sorting",
                           itr.c_str());
        }

        // insert any remaining not excluded
        for(const auto& itr : m_order)
        {
            if(exclude.count(itr) == 0)
                _order.push_back(itr);
        }

        // set the new headers
        m_order = _order;
    }

private:
    bool         m_center       = false;
    char         m_fill         = '-';
    char         m_delim        = '|';
    int          m_width        = 0;
    int          m_precision    = 0;
    int          m_rows         = 0;
    int          m_cols         = 0;
    int64_t      m_prefix_begin = 0;
    int64_t      m_prefix_end   = 0;
    format_flags m_format       = {};
    string_t     m_name         = "";
    string_t     m_banner       = "";
    header_map_t m_headers      = {};
    entry_map_t  m_entries      = {};
    order_map_t  m_order        = {};
    break_set_t  m_break        = {};
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct header_stream
{
    using format_flags = std::ios_base::fmtflags;

    header_stream(format_flags _fmt, int _width, int _prec, bool _center)
    : m_center(_center)
    , m_width(_width)
    , m_precision(_prec)
    , m_format(_fmt)
    {}

    template <typename StreamT>
    StreamT& operator()(StreamT& _os, const Tp& _obj)
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

template <typename... ArgsT>
void
write_header(stream& _os, const std::string& _label, std::ios_base::fmtflags _fmt = {},
             int _width = 0, int _prec = 0, bool _center = true)
{
    _os.set_name(_label);
    _os(header(_label, _fmt, _width, _prec, _center));
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
write_entry(stream& _os, const std::string& _label, const Tp& _value, bool c = false,
            bool l = false)
{
    _os.set_name(_label);
    _os(entry(_value, _os.get_header(_label, 0), c, l));
}

//--------------------------------------------------------------------------------------//

inline void
write_entry(stream& _os, const std::string& _label, const std::string& _value,
            bool c = false, bool = false)
{
    _os.set_name(_label);
    _os(entry(_value, _os.get_header(_label, 0), c, true));
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
write_entry(stream& _os, const std::vector<std::string>& _label, const Tp& _value,
            bool c = false, bool l = false)
{
    write_entry(_os, _label.front(), _value, c, l);
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
write_entry(stream& _os, const std::string& _label,
            const std::vector<std::string>& _value, bool c = false, bool l = false)
{
    for(const auto& itr : _value)
        write_entry(_os, _label, itr, c, l);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
void
write_entry(stream& _os, const std::string& _label, const std::pair<Tp, Up>& _value,
            bool c = false, bool l = false)
{
    write_entry(_os, _label, _value.first, c, l);
    write_entry(_os, _label, _value.second, c, l);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::pair<Tp, Up>& _value, bool c = false, bool l = false)
{
    size_t _L = _labels.size();
    write_entry(_os, _labels.at(0), _value.first, c, l);
    write_entry(_os, _labels.at(1 % _L), _value.second, c, l);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Alloc>
void
write_entry(stream& _os, const std::string& _label,
            const std::vector<Tp, Alloc...>& _values, bool c = false, bool l = false)
{
    for(const auto& itr : _values)
        write_entry(_os, _label, itr, c, l);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Alloc>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::vector<Tp, Alloc...>& _values, bool c = false, bool l = false)
{
    size_t _L = _labels.size();
    size_t N  = _values.size();
    for(size_t i = 0; i < N; ++i)
        write_entry(_os, _labels.at(i % _L), _values.at(i), c, l);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
void
write_entry(stream& _os, const std::string& _label, const std::array<Tp, N>& _values,
            bool c = false, bool l = false)
{
    for(const auto& itr : _values)
        write_entry(_os, _label, itr, c, l);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::array<Tp, N>& _values, bool c = false, bool l = false)
{
    size_t _L = _labels.size();
    for(size_t i = 0; i < N; ++i)
        write_entry(_os, _labels.at(i % _L), _values.at(i), c, l);
}

//--------------------------------------------------------------------------------------//

template <typename... Types, size_t... Idx>
void
write_entry(stream& _os, const std::string& _label, const std::tuple<Types...>& _values,
            index_sequence<Idx...>, bool c = false, bool l = false)
{
    TIMEMORY_FOLD_EXPRESSION(write_entry(_os, _label, std::get<Idx>(_values), c, l));
}

//--------------------------------------------------------------------------------------//

template <typename... Types, size_t... Idx>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::tuple<Types...>& _values, index_sequence<Idx...>, bool c = false,
            bool l = false)
{
    size_t _L = _labels.size();
    TIMEMORY_FOLD_EXPRESSION(
        write_entry(_os, _labels.at(Idx % _L), std::get<Idx>(_values), c, l));
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
void
write_entry(stream& _os, const std::string& _labels, const std::tuple<Types...>& _values,
            bool c = false, bool l = false)
{
    constexpr size_t N = sizeof...(Types);
    write_entry(_os, _labels, _values, make_index_sequence<N>{}, c, l);
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
void
write_entry(stream& _os, const std::vector<std::string>& _labels,
            const std::tuple<Types...>& _values, bool c = false, bool l = false)
{
    constexpr size_t N = sizeof...(Types);
    write_entry(_os, _labels, _values, make_index_sequence<N>{}, c, l);
}

//--------------------------------------------------------------------------------------//

}  // namespace utility

//--------------------------------------------------------------------------------------//

}  // namespace tim
