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

/** \file timer.hpp
 * \headerfile timer.hpp "timemory/timer.hpp"
 * Primary timer class
 * Inherits from base_timer
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include <cstdint>
#include <memory>
#include <string>

#include "timemory/base_timer.hpp"
#include "timemory/formatters.hpp"
#include "timemory/macros.hpp"
#include "timemory/string.hpp"

namespace tim
{
//======================================================================================//
// Main timer class
//======================================================================================//

tim_api class timer : public internal::base_timer
{
public:
    typedef base_timer                 base_type;
    typedef timer                      this_type;
    typedef tim::string                string_t;
    typedef std::unique_ptr<this_type> unique_ptr_type;
    typedef std::shared_ptr<this_type> shared_ptr_type;

public:
    explicit timer(bool _auto_start, timer* _sum_timer);
    timer(const string_t& _prefix = "");
    timer(const timer* rhs, const string_t& _prefix, bool _align_width = false);

    virtual ~timer();

public:
    // copy and assign
    timer(const this_type&);
    this_type& operator=(const this_type&);
    // parent timer accumulating sum
    timer* summation_timer() const { return m_sum_timer; }
    void   summation_timer(timer* _ref) { m_sum_timer = _ref; }

public:
    // public static functions
    static void default_record_memory(bool _val) { f_record_memory() = _val; }
    static bool default_record_memory() { return f_record_memory(); }

public:
    // public member functions
    timer& stop_and_return()
    {
        this->stop();
        return *this;
    }

    tim::string as_string(bool ign_cutoff = true) const
    {
        std::stringstream ss;
        this->report(ss, false, ign_cutoff);
        return ss.str();
    }

    void print(bool ign_cutoff = true) const
    {
        std::cout << this->as_string(ign_cutoff) << std::endl;
    }

    void grab_metadata(const this_type& rhs);

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        internal::base_timer::serialize(ar, version);
    }

    timer* summation_object() const { return m_sum_timer; }

public:
    //------------------------------------------------------------------------//
    //      operator += timer
    //
    this_type& operator+=(const this_type& rhs)
    {
        // auto_lock_t l(m_mutex);
        m_accum += rhs.get_accum();
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator -= timer
    //
    this_type& operator-=(const this_type& rhs)
    {
        // auto_lock_t l(m_mutex);
        m_accum -= rhs.get_accum();
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator *= integer
    //
    this_type& operator*=(const uintmax_t& rhs)
    {
        auto_lock_t l(m_mutex);
        m_accum *= rhs;
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator /= integer
    //
    this_type& operator/=(const uintmax_t& rhs)
    {
        auto_lock_t l(m_mutex);
        m_accum /= rhs;
        return *this;
    }

public:
    //------------------------------------------------------------------------//
    //                          FRIEND operators
    //------------------------------------------------------------------------//
    //      operator - timer
    //
    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        return this_type(lhs) -= rhs;
    }

    //------------------------------------------------------------------------//
    //      operator + timer
    //
    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        return this_type(lhs) += rhs;
    }

public:
    // public member functions

protected:
    timer* m_sum_timer;

private:
    static bool& f_record_memory();
};

//--------------------------------------------------------------------------------------//

inline timer::timer(const string_t& _prefix)
//: base_type(timer_format_t(new format_type(_prefix, _format)))
//, m_sum_timer(nullptr)
{
}

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
