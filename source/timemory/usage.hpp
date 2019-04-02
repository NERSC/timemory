// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file usage.hpp
 * \headerfile usage.hpp "timemory/usage.hpp"
 * Resident set size handler
 *
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <string>

#include "timemory/base_usage.hpp"
#include "timemory/data_types.hpp"
#include "timemory/formatters.hpp"
#include "timemory/macros.hpp"
#include "timemory/serializer.hpp"
#include "timemory/string.hpp"

//======================================================================================//

// RSS - Resident set size (physical memory use, not in swap)

namespace tim
{
//======================================================================================//

tim_api class usage : public base_usage<rusage::peak_rss, rusage::current_rss>
{
public:
    typedef usage                                             this_type;
    typedef intmax_t                                          size_type;
    typedef base_usage<rusage::peak_rss, rusage::current_rss> base_type;
    //

public:
    //------------------------------------------------------------------------//
    //      Default constructor variants
    //------------------------------------------------------------------------//
    usage()
    : m_sum_usage(nullptr)
    {
    }

    //------------------------------------------------------------------------//
    //      Copy construct and assignment
    //------------------------------------------------------------------------//
    usage(const base_type& rhs)
    : base_type(rhs)
    , m_sum_usage(nullptr)
    {
    }

    usage(const usage& rhs)
    : base_type(rhs)
    , m_sum_usage(nullptr)
    {
    }

    usage& operator=(const usage& rhs)
    {
        if(this != &rhs)
        {
            base_usage::operator=(rhs);
        }
        return *this;
    }

public:
    //------------------------------------------------------------------------//
    //                          FRIEND operators
    //------------------------------------------------------------------------//
    //      operator - usage
    //
    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        return this_type(lhs) -= rhs;
    }

    //------------------------------------------------------------------------//
    //      operator + usage
    //
    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        return this_type(lhs) += rhs;
    }

protected:
    usage* m_sum_usage;
};

//--------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
