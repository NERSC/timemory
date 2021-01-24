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
#include "timemory/hash/macros.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/macros/language.hpp"

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              GENERAL PURPOSE TLS DATA
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Tag = TIMEMORY_API, typename PtrT = std::shared_ptr<Tp>,
          typename PairT = std::pair<PtrT, PtrT>>
PairT&
get_shared_ptr_pair();
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Tag = TIMEMORY_API, typename PtrT = std::shared_ptr<Tp>,
          typename PairT = std::pair<PtrT, PtrT>>
PtrT
get_shared_ptr_pair_instance();
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Tag = TIMEMORY_API, typename PtrT = std::shared_ptr<Tp>,
          typename PairT = std::pair<PtrT, PtrT>>
PtrT
get_shared_ptr_pair_master_instance();
//
//--------------------------------------------------------------------------------------//
//
//                                  HASH ALIASES
//
//--------------------------------------------------------------------------------------//
//
using hash_type = std::hash<string_view_t>;
using hash_value_type =
    std::decay_t<decltype(hash_type{}(std::declval<string_view_t>()))>;
using graph_hash_map_t          = std::unordered_map<hash_value_type, std::string>;
using graph_hash_alias_t        = std::unordered_map<hash_value_type, hash_value_type>;
using graph_hash_map_ptr_t      = std::shared_ptr<graph_hash_map_t>;
using graph_hash_map_ptr_pair_t = std::pair<graph_hash_map_ptr_t, graph_hash_map_ptr_t>;
using graph_hash_alias_ptr_t    = std::shared_ptr<graph_hash_alias_t>;
//
//--------------------------------------------------------------------------------------//
//
//                              HASH DATA STRUCTURES
//
//--------------------------------------------------------------------------------------//
//
graph_hash_map_ptr_t&
get_hash_ids() TIMEMORY_HOT;
//
//--------------------------------------------------------------------------------------//
//
graph_hash_alias_ptr_t&
get_hash_aliases() TIMEMORY_HOT;
//
//--------------------------------------------------------------------------------------//
//
//                              STRING -> HASH CONVERSION
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
TIMEMORY_INLINE hash_value_type
                get_hash_id(Tp&& _prefix) TIMEMORY_HOT;
//
template <typename Tp>
hash_value_type
get_hash_id(Tp&& _prefix)
{
    return std::hash<string_view_t>{}(std::forward<Tp>(_prefix));
}
//
//--------------------------------------------------------------------------------------//
//
hash_value_type
get_hash_id(const graph_hash_alias_ptr_t& _hash_alias,
            hash_value_type               _hash_id) TIMEMORY_HOT;
//
//--------------------------------------------------------------------------------------//
//
/// \fn hash_value_type add_hash_id(graph_hash_map_ptr_t&, const string_view_t&)
/// \brief add an string to the given hash-map (if it doesn't already exist) and return
/// the hash
///
hash_value_type
add_hash_id(graph_hash_map_ptr_t& _hash_map, const string_view_t& _prefix) TIMEMORY_HOT;
//
inline hash_value_type
add_hash_id(graph_hash_map_ptr_t& _hash_map, const string_view_t& _prefix)
{
    hash_value_type _hash_id = get_hash_id(_prefix);
    if(_hash_map && _hash_map->find(_hash_id) == _hash_map->end())
    {
        (*_hash_map)[_hash_id] = std::string{ _prefix };
    }
    return _hash_id;
}
//
//--------------------------------------------------------------------------------------//
//
/// \fn hash_value_type add_hash_id(const string_view_t&)
/// \brief add an string to the default hash-map (if it doesn't already exist) and return
/// the hash
///
hash_value_type
add_hash_id(const string_view_t& _prefix) TIMEMORY_HOT;
//
inline hash_value_type
add_hash_id(const string_view_t& _prefix)
{
    return add_hash_id(get_hash_ids(), _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
void
add_hash_id(const graph_hash_map_ptr_t&   _hash_map,
            const graph_hash_alias_ptr_t& _hash_alias, hash_value_type _hash_id,
            hash_value_type _alias_hash_id) TIMEMORY_HOT;
//
//--------------------------------------------------------------------------------------//
//
void
add_hash_id(hash_value_type _hash_id, hash_value_type _alias_hash_id) TIMEMORY_HOT;
//
//--------------------------------------------------------------------------------------//
//
//                              HASH -> STRING CONVERSION
//
//--------------------------------------------------------------------------------------//
//
/// \fn string_view_t get_hash_identifier_fast(hash_value_type)
/// \brief this does not check other threads or aliases. Only call this function when
/// you know that the hash exists on the thread and is not an alias
//
string_view_t
get_hash_identifier_fast(hash_value_type _hash) TIMEMORY_HOT;
//
inline string_view_t
get_hash_identifier_fast(hash_value_type _hash)
{
    auto& _hash_ids = get_hash_ids();
    auto  itr       = _hash_ids->find(_hash);
    if(itr != _hash_ids->end())
        return itr->second;
    return "";
}
//
//--------------------------------------------------------------------------------------//
//
std::string
get_hash_identifier(const graph_hash_map_ptr_t&   _hash_map,
                    const graph_hash_alias_ptr_t& _hash_alias, hash_value_type _hash_id);
//
//--------------------------------------------------------------------------------------//
//
std::string
get_hash_identifier(hash_value_type _hash_id);
//
//--------------------------------------------------------------------------------------//
//
//                              HASH STRING DEMANGLING
//
//--------------------------------------------------------------------------------------//
//
std::string
demangle_hash_identifier(std::string, char bdelim = '[', char edelim = ']');
//
//--------------------------------------------------------------------------------------//
//
template <typename... Args>
auto
get_demangled_hash_identifier(Args&&... _args)
{
    return demangle_hash_identifier(get_hash_identifier(std::forward<Args>(_args)...));
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
