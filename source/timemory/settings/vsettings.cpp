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

#include "timemory/settings/vsettings.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/types.hpp"

namespace tim
{
//
TIMEMORY_SETTINGS_LINKAGE(bool)
vsettings::matches(const std::string& inp) const
{
    if(inp == m_env_name || inp == m_name)
        return true;
    return std::any_of(m_cmdline.begin(), m_cmdline.end(),
                       [&](auto itr) { return (itr == inp); });
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(void)
vsettings::clone(std::shared_ptr<vsettings> rhs)
{
    m_name        = rhs->m_name;
    m_env_name    = rhs->m_env_name;
    m_description = rhs->m_description;
    m_cmdline     = rhs->m_cmdline;
    m_count       = rhs->m_count;
    m_max_count   = rhs->m_max_count;
}
//
}  // namespace tim
