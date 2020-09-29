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

/** \file "timemory/trace.hpp"
 * Header file for library tracing operations
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trace
{
//
//--------------------------------------------------------------------------------------//
//
struct trace
{};
struct region
{};
struct library
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct lock
{
    lock()
    : m_value(!get_global())
    {
        if(m_value)
            get_global() = true;
    }

    ~lock()
    {
        if(m_value)
            get_global() = false;
    }

    lock(lock&&) noexcept = default;
    lock& operator=(lock&&) noexcept = default;

    lock(const lock&) = delete;
    lock& operator=(const lock&) = delete;

    operator bool() const { return m_value; }

    bool& get_local() { return m_value; }

    void release()
    {
        if(m_value)
        {
            get_global() = false;
            m_value      = false;
        }
    }

public:
    static bool& get_global()
    {
        static thread_local bool _instance = false;
        return _instance;
    }

private:
    bool m_value;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace trace
}  // namespace tim

//--------------------------------------------------------------------------------------//
