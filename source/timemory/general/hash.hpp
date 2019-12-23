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

/** \file general/hash.hpp
 * \headerfile general/hash.hpp "timemory/general/hash.hpp"
 * Provides correlation between the hashes and the the prefix for components
 *
 */

#pragma once

#include "timemory/general/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/settings.hpp"

#include <array>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//  hash storage
//
//--------------------------------------------------------------------------------------//

using hash_result_type          = std::size_t;
using graph_hash_map_t          = std::unordered_map<hash_result_type, std::string>;
using graph_hash_alias_t        = std::unordered_map<hash_result_type, hash_result_type>;
using graph_hash_map_ptr_t      = std::shared_ptr<graph_hash_map_t>;
using graph_hash_map_ptr_pair_t = std::pair<graph_hash_map_ptr_t, graph_hash_map_ptr_t>;
using graph_hash_alias_ptr_t    = std::shared_ptr<graph_hash_alias_t>;

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_EXTERN_INIT)

extern graph_hash_map_ptr_t
get_hash_ids();

extern graph_hash_alias_ptr_t
get_hash_aliases();

#else

//--------------------------------------------------------------------------------------//

inline graph_hash_map_ptr_t
get_hash_ids()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<graph_hash_map_t>();
    return _inst;
}

//--------------------------------------------------------------------------------------//

inline graph_hash_alias_ptr_t
get_hash_aliases()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<graph_hash_alias_t>();
    return _inst;
}

#endif

//--------------------------------------------------------------------------------------//

inline hash_result_type
add_hash_id(graph_hash_map_ptr_t& _hash_map, const std::string& prefix)
{
    hash_result_type _hash_id = std::hash<std::string>()(prefix.c_str());
    if(_hash_map && _hash_map->find(_hash_id) == _hash_map->end())
    {
        if(settings::debug())
            printf("[%s@'%s':%i]> adding hash id: %s = %llu...\n", __FUNCTION__, __FILE__,
                   __LINE__, prefix.c_str(), (long long unsigned) _hash_id);

        (*_hash_map)[_hash_id] = prefix;
        if(_hash_map->bucket_count() < _hash_map->size())
            _hash_map->rehash(_hash_map->size() + 10);
    }
    return _hash_id;
}

//--------------------------------------------------------------------------------------//

inline hash_result_type
add_hash_id(const std::string& prefix)
{
    static thread_local auto _hash_map = get_hash_ids();
    return add_hash_id(_hash_map, prefix);
}

//--------------------------------------------------------------------------------------//

inline void
add_hash_id(graph_hash_map_ptr_t _hash_map, graph_hash_alias_ptr_t _hash_alias,
            hash_result_type _hash_id, hash_result_type _alias_hash_id)
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
add_hash_id(hash_result_type _hash_id, hash_result_type _alias_hash_id)
{
    add_hash_id(get_hash_ids(), get_hash_aliases(), _hash_id, _alias_hash_id);
}

//--------------------------------------------------------------------------------------//

inline std::string
get_hash_identifier(graph_hash_map_ptr_t _hash_map, graph_hash_alias_ptr_t _hash_alias,
                    hash_result_type _hash_id)
{
    auto _map_itr   = _hash_map->find(_hash_id);
    auto _alias_itr = _hash_alias->find(_hash_id);

    if(_map_itr != _hash_map->end())
        return _map_itr->second;
    else if(_alias_itr != _hash_alias->end())
    {
        _map_itr = _hash_map->find(_alias_itr->second);
        if(_map_itr != _hash_map->end())
            return _map_itr->second;
    }

    if(settings::verbose() > 0 || settings::debug())
    {
        std::stringstream ss;
        ss << "Error! node with hash " << _hash_id
           << " did not have an associated prefix!\n";
        ss << "Hash map:\n";
        auto _w = 30;
        for(const auto& itr : *_hash_map)
            ss << "    " << std::setw(_w) << itr.first << " : " << (itr.second) << "\n";
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
get_hash_identifier(hash_result_type _hash_id)
{
    return get_hash_identifier(get_hash_ids(), get_hash_aliases(), _hash_id);
}

//======================================================================================//

}  // namespace tim
