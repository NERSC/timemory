//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file serializer.hpp
 * \headerfile serializer.hpp "timemory/utility/serializer.hpp"
 * Headers for serialization
 */

#pragma once

// define this so avoid warnings about noexcept functions throwing
#define CEREAL_RAPIDJSON_ASSERT(x)                                                       \
    {                                                                                    \
    }

// general
#include <cereal/cereal.hpp>
#include <cereal/macros.hpp>
// #include <cereal/access.hpp>

// types
#include <cereal/types/array.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

// #include <cereal/types/functional.hpp>
// #include <cereal/types/deque.hpp>
// #include <cereal/types/atomic.hpp>
// #include <cereal/types/bitset.hpp>
// #include <cereal/types/chrono.hpp>
// #include <cereal/types/complex.hpp>
// #include <cereal/types/forward_list.hpp>
// #include <cereal/types/list.hpp>
// #include <cereal/types/polymorphic.hpp>
// #include <cereal/types/queue.hpp>
// #include <cereal/types/set.hpp>
// #include <cereal/types/stack.hpp>
// #include <cereal/types/unordered_map.hpp>
// #include <cereal/types/unordered_set.hpp>

// archives
#include <cereal/archives/json.hpp>
// #include <cereal/archives/adapters.hpp>
// #include <cereal/archives/binary.hpp>

//======================================================================================//

namespace serializer
{
//--------------------------------------------------------------------------------------//

using cereal::make_nvp;

//--------------------------------------------------------------------------------------//

}  // namespace serializer

//======================================================================================//
