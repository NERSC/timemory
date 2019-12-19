//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

/** \file init.cpp
 * This file defined the extern init
 *
 */

#include "timemory/components.hpp"
#include "timemory/general/hash.hpp"
#include "timemory/manager.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/utility.hpp"

using namespace tim::component;

#if defined(TIMEMORY_EXTERN_INIT)

//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//
env_settings*
env_settings::instance()
{
    static env_settings* _instance = new env_settings();
    return _instance;
}

//--------------------------------------------------------------------------------------//
//
//
graph_hash_map_ptr_t
get_hash_ids()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<graph_hash_map_t>();
    return _inst;
}

//--------------------------------------------------------------------------------------//
//
//
graph_hash_alias_ptr_t
get_hash_aliases()
{
    static thread_local auto _inst = get_shared_ptr_pair_instance<graph_hash_alias_t>();
    return _inst;
}

}  // namespace tim

#endif  // defined(TIMEMORY_EXTERN_INIT)
