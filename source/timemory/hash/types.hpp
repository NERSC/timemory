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
 * \file timemory/hash/types.hpp
 * \brief Declare the hash types
 */

#pragma once

#include "timemory/hash/macros.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//  hash storage
//
//--------------------------------------------------------------------------------------//
//
using hash_result_type          = size_t;
using graph_hash_map_t          = std::unordered_map<hash_result_type, std::string>;
using graph_hash_alias_t        = std::unordered_map<hash_result_type, hash_result_type>;
using graph_hash_map_ptr_t      = std::shared_ptr<graph_hash_map_t>;
using graph_hash_map_ptr_pair_t = std::pair<graph_hash_map_ptr_t, graph_hash_map_ptr_t>;
using graph_hash_alias_ptr_t    = std::shared_ptr<graph_hash_alias_t>;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
