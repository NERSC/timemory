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
 * \file timemory/components/papi/extern.hpp
 * \brief Include the extern declarations for papi components
 */

#pragma once

#if defined(TIMEMORY_USE_PAPI)

//======================================================================================//
//
#    include "timemory/components/base.hpp"
#    include "timemory/components/macros.hpp"
//
#    include "timemory/components/papi/components.hpp"
#    include "timemory/components/papi/types.hpp"
//
#    if defined(TIMEMORY_COMPONENT_SOURCE) ||                                            \
        (!defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_COMPONENT_EXTERN))
// source/header-only requirements
#        include "timemory/environment/declaration.hpp"
#        include "timemory/operations/definition.hpp"
#        include "timemory/plotting/definition.hpp"
#        include "timemory/settings/declaration.hpp"
#        include "timemory/storage/definition.hpp"
#    else
// extern requirements
#        include "timemory/environment/declaration.hpp"
#        include "timemory/operations/definition.hpp"
#        include "timemory/plotting/declaration.hpp"
#        include "timemory/settings/declaration.hpp"
#        include "timemory/storage/declaration.hpp"
#    endif
//
//======================================================================================//
//
namespace tim
{
namespace component
{
//
TIMEMORY_EXTERN_TEMPLATE(struct base<papi_vector, std::vector<long long>>)
TIMEMORY_EXTERN_TEMPLATE(struct base<papi_array<8>, std::array<long long, 8>>)
TIMEMORY_EXTERN_TEMPLATE(struct base<papi_array<16>, std::array<long long, 16>>)
TIMEMORY_EXTERN_TEMPLATE(struct base<papi_array<32>, std::array<long long, 32>>)
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
//
TIMEMORY_EXTERN_OPERATIONS(component::papi_vector, true)
TIMEMORY_EXTERN_OPERATIONS(component::papi_array8_t, true)
TIMEMORY_EXTERN_OPERATIONS(component::papi_array16_t, true)
TIMEMORY_EXTERN_OPERATIONS(component::papi_array32_t, true)
//
//======================================================================================//
//
TIMEMORY_EXTERN_STORAGE(component::papi_vector, papi_vector)
TIMEMORY_EXTERN_STORAGE(component::papi_array8_t, papi_array8)
TIMEMORY_EXTERN_STORAGE(component::papi_array16_t, papi_array16)
TIMEMORY_EXTERN_STORAGE(component::papi_array32_t, papi_array32)
//
//======================================================================================//

#endif  // TIMEMORY_USE_PAPI
