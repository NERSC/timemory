// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file storage.hpp
 * \headerfile storage.hpp "timemory/storage.hpp"
 * Storage class for manager
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/apply.hpp"
#include "timemory/formatters.hpp"
#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpi.hpp"
#include "timemory/serializer.hpp"
#include "timemory/singleton.hpp"
#include "timemory/string.hpp"
#include "timemory/timer.hpp"
#include "timemory/utility.hpp"

//--------------------------------------------------------------------------------------//

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct tim_api data_tuple
: public std::tuple<uintmax_t, uintmax_t, uintmax_t, tim::string, std::shared_ptr<_Tp>>
{
    typedef data_tuple<_Tp>            this_type;
    typedef tim::string                string_t;
    typedef _Tp                        data_type;
    typedef std::shared_ptr<data_type> pointer_type;
    typedef std::tuple<uintmax_t, uintmax_t, uintmax_t, tim::string, std::shared_ptr<_Tp>>
        base_type;

    //------------------------------------------------------------------------//
    //      constructors
    //
    // default and full initialization
    data_tuple(uintmax_t _a = 0, uintmax_t _b = 0, uintmax_t _c = 0, string_t _d = "",
               pointer_type _e = pointer_type(nullptr))
    : base_type(_a, _b, _c, _d, _e)
    {
    }
    // copy constructors
    data_tuple(const data_tuple&) = default;
    data_tuple(data_tuple&&)      = default;
    data_tuple(const base_type& _data)
    : base_type(_data)
    {
    }
    // assignment operator
    data_tuple& operator=(const data_tuple& rhs)
    {
        if(this == &rhs)
            return *this;
        base_type::operator=(rhs);
        return *this;
    }
    // move operator
    data_tuple& operator=(data_tuple&& rhs)
    {
        if(this == &rhs)
            return *this;
        base_type::operator=(std::move(rhs));
        return *this;
    }

    //------------------------------------------------------------------------//
    //
    //
    uintmax_t&       key() { return std::get<0>(*this); }
    const uintmax_t& key() const { return std::get<0>(*this); }

    uintmax_t&       level() { return std::get<1>(*this); }
    const uintmax_t& level() const { return std::get<1>(*this); }

    uintmax_t&       offset() { return std::get<2>(*this); }
    const uintmax_t& offset() const { return std::get<2>(*this); }

    string_t        tag() { return std::get<3>(*this); }
    const string_t& tag() const { return std::get<3>(*this); }

    data_type&       data() { return *(std::get<4>(*this).get()); }
    const data_type& data() const { return *(std::get<4>(*this).get()); }

    //------------------------------------------------------------------------//
    //
    //
    data_tuple& operator=(const base_type& rhs)
    {
        if(this == &rhs)
            return *this;
        base_type::operator=(rhs);
        return *this;
    }

    //------------------------------------------------------------------------//
    //
    //
    friend bool operator==(const this_type& lhs, const this_type& rhs)
    {
        return (lhs.key() == rhs.key() && lhs.level() == rhs.level() &&
                lhs.tag() == rhs.tag());
    }

    //------------------------------------------------------------------------//
    //
    //
    friend bool operator!=(const this_type& lhs, const this_type& rhs)
    {
        return !(lhs == rhs);
    }

    //------------------------------------------------------------------------//
    //
    //
    this_type& operator+=(const this_type& rhs)
    {
        data() += rhs.data();
        return *this;
    }

    //------------------------------------------------------------------------//
    //
    //
    const this_type operator+(const this_type& rhs) const
    {
        return this_type(*this) += rhs;
    }

    //------------------------------------------------------------------------//
    // serialization function
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(serializer::make_nvp("key", key()), serializer::make_nvp("level", level()),
           serializer::make_nvp("offset", offset()),
           serializer::make_nvp("tag", std::string(tag().c_str())),
           serializer::make_nvp("data", data()));
    }

    //------------------------------------------------------------------------//
    //
    //
    friend std::ostream& operator<<(std::ostream& os, const data_tuple& t)
    {
        std::stringstream ss;
        ss << std::setw(2 * t.level()) << ""
           << "[" << t.tag() << "]";
        os << ss.str();
        return os;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
tim_api class data_storage
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef _Tp                              value_t;
    typedef std::shared_ptr<_Tp>             pointer_t;
    typedef data_tuple<_Tp>                  tuple_t;
    typedef uomap<uintmax_t, pointer_t>      map_t;
    typedef tim::graph<tuple_t>              graph_t;
    typedef typename graph_t::iterator       iterator;
    typedef typename graph_t::const_iterator const_iterator;

public:
    template <typename _Up                                           = _Tp,
              enable_if_t<std::is_same<_Up, tim::timer>::value, int> = 0>
    explicit data_storage(int32_t instance_count)
    : m_missing(tim::format::timer(tim::string("TiMemory total unrecorded time"),
                                   tim::format::timer::default_format(),
                                   tim::format::timer::default_unit()))
    , m_total(pointer_t(new value_t(tim::format::timer(
          tim::string("> [exe] total"), tim::format::timer::default_format(),
          tim::format::timer::default_unit(), true))))
    {
        std::stringstream ss;
        ss << "TiMemory total unrecorded time (manager " << (instance_count) << ")";
        m_missing.format()->prefix(ss.str());
        m_missing.start();
    }

    template <typename _Up                                           = _Tp,
              enable_if_t<std::is_same<_Up, tim::usage>::value, int> = 0>
    explicit data_storage(int32_t instance_count)
    : m_missing(tim::format::rss(tim::string("TiMemory total unrecorded time"),
                                 tim::format::timer::default_format(),
                                 tim::format::timer::default_unit()))
    , m_total(pointer_t(new value_t(tim::format::rss(
          tim::string("> [exe] total"), tim::format::timer::default_format(),
          tim::format::timer::default_unit(), true))))
    {
        std::stringstream ss;
        ss << "TiMemory total unrecorded time (manager " << (instance_count) << ")";
        m_missing.format()->prefix(ss.str());
        m_missing.record();
    }

    ~data_storage() = default;

    data_storage(const data_storage&) = default;
    data_storage(data_storage&&)      = default;
    data_storage& operator=(const data_storage&) = default;
    data_storage& operator=(data_storage&&) = default;

public:
    void set_head(iterator itr) { m_head = itr; }
    void set_head(tuple_t& itr) { m_current = m_graph.set_head(itr); }

    const_iterator   begin() const { return m_graph.cbegin(); }
    const_iterator   cbegin() const { return m_graph.cbegin(); }
    iterator         end() { return m_graph.end(); }
    const_iterator   end() const { return m_graph.cend(); }
    const_iterator   cend() const { return m_graph.cend(); }
    map_t&           map() { return m_map; }
    const map_t&     map() const { return m_map; }
    iterator         current() const { return m_current; }
    graph_t          graph() const { return m_graph; }
    iterator&        current() { return m_current; }
    graph_t&         graph() { return m_graph; }
    const value_t&   missing() const { return m_missing; }
    const pointer_t& total() const { return m_total; }
    value_t&         missing() { return m_missing; }
    pointer_t&       total() { return m_total; }
    iterator         head() { return m_head; }

    void      pop_graph() { m_current = graph_t::parent(m_current); }
    uintmax_t total_laps() const
    {
        uintmax_t _n = 0;
        for(const auto& itr : *this)
            _n += itr.data().laps();
        return _n;
    }

    //--------------------------------------------------------------------------------------//
    inline void start_total() { m_total->stop(); }
    //--------------------------------------------------------------------------------------//
    inline void stop_total() { m_total->stop(); }
    //--------------------------------------------------------------------------------------//
    inline void reset_total()
    {
        bool _restart = m_total->is_running();
        if(_restart)
            m_total->stop();
        m_total->reset();
        if(_restart)
            m_total->start();
    }

private:
    /// missing timer
    value_t m_missing;
    /// global timer
    pointer_t m_total;
    /// hashed string map for fast lookup
    map_t m_map;
    /// graph for storage
    graph_t m_graph;
    /// current graph iterator
    iterator m_current;
    /// head node
    iterator m_head = nullptr;
};

//--------------------------------------------------------------------------------------//

}  // namespace tim
