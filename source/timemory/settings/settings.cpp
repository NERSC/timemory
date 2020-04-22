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

#include "timemory/settings/macros.hpp"
//
#include "timemory/settings/types.hpp"
//
#include "timemory/settings/declaration.hpp"
#include "timemory/settings/definition.hpp"
//

TIMEMORY_SETTINGS_EXTERN_TEMPLATE(api::native_tag)

#if defined(TIMEMORY_SETTINGS_SOURCE) ||                                                 \
    !(defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_SETTINGS_EXTERN))

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(setting_callback_map_t&)
get_parse_callbacks() { return get_parse_callback_map<settings, api::native_tag>(); }
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(setting_description_map_t&)
get_setting_descriptions() { return get_descriptions<settings, api::native_tag>(); }
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

#endif
