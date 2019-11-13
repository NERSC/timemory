// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/bits/settings.hpp"
#include "timemory/utility/graph.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//  graph instance + current node + head node
//
//--------------------------------------------------------------------------------------//

template <typename _Node>
class graph_data
{
public:
    using this_type      = graph_data<_Node>;
    using graph_t        = tim::graph<_Node>;
    using iterator       = typename graph_t::iterator;
    using const_iterator = typename graph_t::const_iterator;

public:
    // graph_data() = default;

    explicit graph_data(const _Node& rhs)
    : m_has_head(true)
    , m_depth(0)
    {
        m_head    = m_graph.set_head(rhs);
        m_depth   = 0;
        m_current = m_head;
    }

    ~graph_data() { m_graph.clear(); }

    // allow move and copy construct
    graph_data(this_type&&) = delete;
    graph_data& operator=(this_type&&) = delete;

    // delete copy-assignment
    graph_data(const this_type&) = delete;
    graph_data& operator=(const this_type&) = delete;

    bool has_head() const { return m_has_head; }

    const int64_t& depth() const { return m_depth; }
    const graph_t& graph() const { return m_graph; }

    int64_t&  depth() { return m_depth; }
    graph_t&  graph() { return m_graph; }
    iterator& current() { return m_current; }
    iterator& head() { return m_head; }

    iterator       begin() { return m_graph.begin(); }
    iterator       end() { return m_graph.end(); }
    const_iterator begin() const { return m_graph.begin(); }
    const_iterator end() const { return m_graph.end(); }

    inline void clear()
    {
        m_graph.clear();
        m_has_head = false;
        m_depth    = 0;
        m_current  = nullptr;
    }

    inline void reset()
    {
        m_graph.erase_children(m_head);
        m_depth   = 0;
        m_current = m_head;
    }

    inline iterator pop_graph()
    {
        if(m_depth > 0 && !m_graph.is_head(m_current))
        {
            --m_depth;
            m_current = graph_t::parent(m_current);
        }
        else if(m_depth == 0)
        {
            m_current = m_head;
        }
        return m_current;
    }

    inline iterator append_child(_Node& node)
    {
        ++m_depth;
        return (m_current = m_graph.append_child(m_current, node));
    }

    inline iterator append_head(_Node& node)
    {
        return m_graph.append_child(m_head, node);
    }

    inline iterator emplace_child(iterator _itr, _Node& node)
    {
        return m_graph.append_child(_itr, node);
    }

private:
    bool     m_has_head = false;
    int64_t  m_depth    = 0;
    graph_t  m_graph;
    iterator m_current = nullptr;
    iterator m_head    = nullptr;
};
}  // namespace tim
