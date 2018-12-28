//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California,
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

#ifndef test_interface_hpp_
#define test_interface_hpp_

// C headers
#include <cassert>
#include <cstdint>
#include <cstdio>

// C++ headers
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// TiMemory headers
#include "timemory/string.hpp"
#include <timemory/mpi.hpp>

//----------------------------------------------------------------------------//

// ASSERT_NEAR
// EXPECT_EQ
// EXPECT_FLOAT_EQ
// EXPECT_DOUBLE_EQ

#define EXPECT_EQ(lhs, rhs)                                                    \
    if(lhs != rhs)                                                             \
    {                                                                          \
        std::stringstream ss;                                                  \
        ss << #lhs << " != " << #rhs << " @ line " << __LINE__ << " of "       \
           << __FILE__;                                                        \
        std::cerr << ss.str() << std::endl;                                    \
        throw std::runtime_error(ss.str());                                    \
    }

#define ASSERT_FALSE(expr)                                                     \
    if(expr)                                                                   \
    {                                                                          \
        std::stringstream ss;                                                  \
        ss << "Expression: ( " << #expr << " ) "                               \
           << "failed @ line " << __LINE__ << " of " << __FILE__;              \
        std::cerr << ss.str() << std::endl;                                    \
        throw std::runtime_error(ss.str());                                    \
    }

#define ASSERT_TRUE(expr)                                                      \
    if(!(expr))                                                                \
    {                                                                          \
        std::stringstream ss;                                                  \
        ss << "Expression: !( " << #expr << " ) "                              \
           << "failed @ line " << __LINE__ << " of " << __FILE__;              \
        std::cerr << ss.str() << std::endl;                                    \
        throw std::runtime_error(ss.str());                                    \
    }

#define PRINT_HERE printf(" [%s@'%s':%i]\n", __FUNCTION__, __FILE__, __LINE__)

inline tim::string
rank_prefix()
{
    std::stringstream ss;
    if(tim::mpi_is_initialized())
        ss << "[" << tim::mpi_rank() << "] ";
    return ss.str();
}

#define rank_cout std::cout << rank_prefix()

//----------------------------------------------------------------------------//
#define TEST_SUMMARY(argv_0, ntest_counter, nfail_counter)                     \
    {                                                                          \
        std::stringstream rank_sout;                                           \
        rank_sout << "\nDone.\n" << std::endl;                                 \
        rank_sout << "[" << argv_0 << "] ";                                    \
        if(num_fail > 0)                                                       \
            rank_sout << "Tests failed: " << nfail_counter << "/"              \
                      << ntest_counter << std::endl;                           \
        else                                                                   \
            rank_sout << "Tests passed: " << (ntest_counter - nfail_counter)   \
                      << "/" << ntest_counter << std::endl;                    \
        rank_cout << rank_sout.str();                                          \
    }

//----------------------------------------------------------------------------//
// Usage:
//  try
//  {
//      RUN_TEST(test_serialize, num_test, num_fail);
//  }
//  catch(std::exception& e)
//  {
//      std::cerr << e.what() << std::endl;
//  }
//
#define RUN_TEST(func, ntest_counter, nfail_counter)                           \
    {                                                                          \
        try                                                                    \
        {                                                                      \
            ntest_counter += 1;                                                \
            func();                                                            \
        }                                                                      \
        catch(std::exception & e)                                              \
        {                                                                      \
            std::cerr << e.what() << std::endl;                                \
            nfail_counter += 1;                                                \
        }                                                                      \
    }

//----------------------------------------------------------------------------//

#endif
