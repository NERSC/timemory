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

#include "timemory/details/settings.hpp"
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

using graph_hash_map_t       = std::unordered_map<int64_t, std::shared_ptr<std::string>>;
using graph_hash_alias_t     = std::unordered_map<int64_t, int64_t>;
using graph_hash_map_ptr_t   = std::shared_ptr<graph_hash_map_t>;
using graph_hash_alias_ptr_t = std::shared_ptr<graph_hash_alias_t>;

//--------------------------------------------------------------------------------------//

inline graph_hash_map_ptr_t
get_hash_ids()
{
    static thread_local auto _pointer = std::make_shared<graph_hash_map_t>();
    return _pointer;
}

//--------------------------------------------------------------------------------------//

inline graph_hash_alias_ptr_t
get_hash_aliases()
{
    static thread_local auto _pointer = std::make_shared<graph_hash_alias_t>();
    return _pointer;
}

//--------------------------------------------------------------------------------------//

inline int64_t
add_hash_id(graph_hash_map_ptr_t& _hash_map, const std::string& prefix)
{
    int64_t _hash_id = std::hash<std::string>()(prefix.c_str());
    if(_hash_map && _hash_map->find(_hash_id) == _hash_map->end())
    {
        if(settings::debug())
            printf("[%s@'%s':%i]> adding hash id: %s...\n", __FUNCTION__, __FILE__,
                   __LINE__, prefix.c_str());

        (*_hash_map)[_hash_id] = std::make_shared<std::string>(prefix);
        if(_hash_map->bucket_count() < _hash_map->size())
            _hash_map->rehash(_hash_map->size() + 10);
    }
    return _hash_id;
}

//--------------------------------------------------------------------------------------//

inline int64_t
add_hash_id(const std::string& prefix)
{
    static thread_local auto _hash_map = get_hash_ids();
    return add_hash_id(_hash_map, prefix);
}

//--------------------------------------------------------------------------------------//

inline void
add_hash_id(graph_hash_map_ptr_t _hash_map, graph_hash_alias_ptr_t _hash_alias,
            int64_t _hash_id, int64_t _alias_hash_id)
{
    if(_hash_alias->find(_alias_hash_id) == _hash_alias->end() &&
       _hash_map->find(_hash_id) != _hash_map->end())
    {
        (*_hash_alias)[_alias_hash_id] = _hash_id;
        if(_hash_alias->bucket_count() < _hash_alias->size())
            _hash_alias->rehash(_hash_alias->size() + 10);
    }
}

//--------------------------------------------------------------------------------------//

inline void
add_hash_id(int64_t _hash_id, int64_t _alias_hash_id)
{
    add_hash_id(get_hash_ids(), get_hash_aliases(), _hash_id, _alias_hash_id);
}

//--------------------------------------------------------------------------------------//

inline std::string
get_hash_identifier(graph_hash_map_ptr_t _hash_map, graph_hash_alias_ptr_t _hash_alias,
                    int64_t _hash_id)
{
    using string_ptr_t = std::shared_ptr<std::string>;
    auto _get_string   = [](string_ptr_t _str) {
        if(_str.get())
            return *_str;
        else
            return std::string("");
        /*
        const auto len = _str.length() + 1;
        char* _cstr = new char[len];
        for(uintmax_t i = 0; i < len; ++i)
            _cstr[i] = _str[i];
        _cstr[len-1] = '\0';
        std::string tmp(_cstr, _cstr + len);
        delete [] _cstr;
        return tmp;
        */
    };

    auto _map_itr   = _hash_map->find(_hash_id);
    auto _alias_itr = _hash_alias->find(_hash_id);

    if(_map_itr != _hash_map->end())
        return _get_string(_map_itr->second);
    else if(_alias_itr != _hash_alias->end())
    {
        _map_itr = _hash_map->find(_alias_itr->second);
        if(_map_itr != _hash_map->end())
            return _get_string(_map_itr->second);
    }

    if(settings::verbose() > 0 || settings::debug())
    {
        std::stringstream ss;
        ss << "Error! node with hash " << _hash_id
           << " did not have an associated prefix!\n";
        ss << "Hash map:\n";
        auto _w = 30;
        for(const auto& itr : *_hash_map)
            ss << "    " << std::setw(_w) << itr.first << " : " << (*itr.second) << "\n";
        if(_hash_alias->size() > 0)
        {
            ss << "Alias hash map:\n";
            for(const auto& itr : *_hash_alias)
                ss << "    " << std::setw(_w) << itr.first << " : " << itr.second << "\n";
        }
        fprintf(stderr, "%s\n", ss.str().c_str());
    }
    return std::string("unknown-hash=") + std::to_string(_hash_id);
}

//--------------------------------------------------------------------------------------//

inline std::string
get_hash_identifier(int64_t _hash_id)
{
    return get_hash_identifier(get_hash_ids(), get_hash_aliases(), _hash_id);
}

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
    graph_data() = default;

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
    graph_data(const this_type&) = default;
    graph_data& operator=(this_type&&) = default;

    // delete copy-assignment
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
        m_has_head = false;
        m_depth    = 0;
        m_graph.clear();
        m_current = nullptr;
        m_head    = nullptr;
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

    inline iterator emplace_child(iterator _itr, _Node&& node)
    {
        return m_graph.append_child(_itr, std::move(node));
    }

private:
    bool     m_has_head = false;
    int64_t  m_depth    = 0;
    graph_t  m_graph;
    iterator m_current = nullptr;
    iterator m_head    = nullptr;
};
}  // namespace tim
