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

#pragma once

#include "timemory/api.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/hash/macros.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/utility/utility.hpp"

#include <cstdint>
#include <iomanip>
#include <iosfwd>
#include <sstream>
#include <string>
#include <unordered_map>

#if defined(TIMEMORY_HASH_SOURCE) || !defined(TIMEMORY_USE_HASH_EXTERN)

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(hash_map_ptr_t&)
get_hash_ids()
{
    static thread_local auto _inst =
        get_shared_ptr_pair_instance<hash_map_t, TIMEMORY_API>();
    return _inst;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(hash_alias_ptr_t&)
get_hash_aliases()
{
    static thread_local auto _inst =
        get_shared_ptr_pair_instance<hash_alias_map_t, TIMEMORY_API>();
    return _inst;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(std::shared_ptr<hash_resolver_vec_t>&)
get_hash_resolvers()
{
    static auto _inst = []() {
        auto _subinst = get_shared_ptr_lone_instance<hash_resolver_vec_t, TIMEMORY_API>();
        if(_subinst && _subinst->empty())
            _subinst->reserve(10);
        return _subinst;
    }();
    return _inst;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(hash_value_t)
get_hash_id(const hash_alias_ptr_t& _hash_alias, hash_value_t _hash_id)
{
    auto _alias_itr = _hash_alias->find(_hash_id);
    if(_alias_itr != _hash_alias->end())
        return _alias_itr->second;
    return _hash_id;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(void)
add_hash_id(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
            hash_value_t _hash_id, hash_value_t _alias_hash_id)
{
    if(_hash_alias->find(_alias_hash_id) == _hash_alias->end() &&
       _hash_map->find(_hash_id) != _hash_map->end())
    {
        (*_hash_alias)[_alias_hash_id] = _hash_id;
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(void)
add_hash_id(hash_value_t _hash_id, hash_value_t _alias_hash_id)
{
    add_hash_id(get_hash_ids(), get_hash_aliases(), _hash_id, _alias_hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(std::string)
get_hash_identifier(const hash_map_ptr_t& _hash_map, const hash_alias_ptr_t& _hash_alias,
                    hash_value_t _hash_id)
{
    auto _map_itr   = _hash_map->find(_hash_id);
    auto _alias_itr = _hash_alias->find(_hash_id);

    if(_map_itr != _hash_map->end())
    {
        return _map_itr->second;
    }

    if(_alias_itr != _hash_alias->end())
    {
        _map_itr = _hash_map->find(_alias_itr->second);
        if(_map_itr != _hash_map->end())
            return _map_itr->second;
    }

    if(_hash_id > 0)
    {
        std::stringstream ss;
        ss << "Error! node with hash " << _hash_id
           << " does not have an associated string!";
#    if defined(DEBUG)
        ss << "\nHash map:\n";
        auto _w = 30;
        for(const auto& itr : *_hash_map)
            ss << "    " << std::setw(_w) << itr.first << " : " << (itr.second) << "\n";
        if(_hash_alias->size() > 0)
        {
            ss << "Alias hash map:\n";
            for(const auto& itr : *_hash_alias)
                ss << "    " << std::setw(_w) << itr.first << " : " << itr.second << "\n";
        }
#    endif
        fprintf(stderr, "%s\n", ss.str().c_str());
    }

    return std::string("unknown-hash=") + std::to_string(_hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(std::string)
get_hash_identifier(hash_value_t _hash_id)
{
    return get_hash_identifier(get_hash_ids(), get_hash_aliases(), _hash_id);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_HASH_LINKAGE(std::string)
demangle_hash_identifier(std::string inp, char bdelim, char edelim)
{
    inp         = demangle(inp);
    size_t _beg = inp.find_first_of(bdelim);
    while(_beg != std::string::npos)
    {
        size_t _end = inp.find_first_of(edelim, _beg);
        if(_end == std::string::npos)
            break;
        auto _sz = _end - _beg - 1;
        inp      = inp.replace(_beg + 1, _sz, demangle(inp.substr(_beg + 1, _sz)));
        _beg     = inp.find_first_of(bdelim, _end + 1);
    }
    return demangle(inp);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

#endif
