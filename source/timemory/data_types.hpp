//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

//--------------------------------------------------------------------------------------//

#include <atomic>
#include <fstream>
#include <string>

#include "timemory/base_clock.hpp"
#include "timemory/macros.hpp"
#include "timemory/serializer.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/string.hpp"
#include "timemory/utility.hpp"

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace internal
{
//======================================================================================//
//
//  Class for handling the start/stop data
//
//======================================================================================//

template <typename _Tp>
tim_api class base_data
{
public:
    typedef base_data<_Tp>       this_type;
    typedef std::tuple<_Tp, _Tp> data_type;

public:
    base_data()
    : m_running(false)
    , m_data(data_type())
    {
    }

    _Tp& start() { return std::get<0>(m_data); }
    _Tp& stop() { return std::get<1>(m_data); }

    const _Tp& start() const { return std::get<0>(m_data); }
    const _Tp& stop() const { return std::get<1>(m_data); }

    void pause() { m_running = false; }
    void resume() { m_running = true; }

    const bool& running() const { return m_running; }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(serializer::make_nvp("start", std::get<0>(m_data)),
           serializer::make_nvp("stop", std::get<1>(m_data)));
    }

    this_type& operator=(const this_type& rhs)
    {
        if(this != &rhs)
        {
            m_running = rhs.m_running;
            m_data    = rhs.m_data;
        }
        return *this;
    }

    data_type&       data() { return m_data; }
    const data_type& data() const { return m_data; }

protected:
    mutable bool m_running;
    data_type    m_data;
};

//======================================================================================//
//
//  Class for handling the data differences
//
//======================================================================================//

template <typename _Tp>
tim_api class base_delta
{
public:
    typedef base_delta<_Tp> this_type;
    typedef _Tp             data_type;
    typedef _Tp             incr_type;
    typedef base_data<_Tp>  op_type;

public:
    base_delta()
    : m_lap(0)
    , m_sum(data_type())
    {
    }

public:
    const uintmax_t& size() const { return m_lap; }
    uintmax_t&       size() { return m_lap; }
    const _Tp&       sum() const { return m_sum; }

    void reset()
    {
        m_lap = 0;
        // m_sum = data_type();
    }

public:
    //------------------------------------------------------------------------//
    //      operator = this
    //
    virtual this_type& operator=(const this_type& rhs)
    {
        if(this != &rhs)
        {
            m_lap = rhs.m_lap;
            m_sum = rhs.m_sum;
        }
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator += data
    //
    virtual this_type& operator+=(const op_type& data)
    {
        m_sum += data;
        m_lap += 1;

        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator += this
    //
    virtual this_type& operator+=(const this_type& rhs)
    {
        m_lap += rhs.m_lap;
        m_sum += rhs.m_sum;
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator -= this
    //
    virtual this_type& operator-=(const this_type& rhs)
    {
        m_sum -= rhs.m_sum;
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator *= integer
    //
    virtual this_type& operator*=(const uintmax_t& rhs)
    {
        if(rhs > 0)
            m_sum *= rhs;
        return *this;
    }

    //------------------------------------------------------------------------//
    //      operator /= integer
    //
    virtual this_type& operator/=(const uintmax_t& rhs)
    {
        if(rhs > 0)
            m_sum /= rhs;
        return *this;
    }

protected:
    uintmax_t m_lap;
    data_type m_sum;
};

//======================================================================================//
//
//  class for timer data
//
//======================================================================================//

tim_api class timer_data
{
public:
    typedef timer_data                                  this_type;
    typedef std::tuple<uintmax_t, uintmax_t, uintmax_t> data_type;

    this_type& operator=(const data_type& data)
    {
        m_real = std::get<0>(data);
        m_user = std::get<1>(data);
        m_syst = std::get<2>(data);
        return *this;
    }

    this_type& operator+=(const this_type& rhs)
    {
        m_real += rhs.real();
        m_user += rhs.user();
        m_syst += rhs.sys();
        return *this;
    }

    this_type& operator+=(const base_data<this_type>& rhs)
    {
        m_real += (rhs.stop().real() - rhs.start().real());
        m_user += (rhs.stop().user() - rhs.start().user());
        m_syst += (rhs.stop().sys() - rhs.start().sys());
        return *this;
    }

    this_type& operator+=(const base_delta<this_type>& rhs)
    {
        m_real += rhs.sum().real();
        m_user += rhs.sum().user();
        m_syst += rhs.sum().sys();
        return *this;
    }

    this_type& operator-=(const this_type& rhs)
    {
        m_real -= rhs.real();
        m_user -= rhs.user();
        m_syst -= rhs.sys();
        return *this;
    }

    this_type& operator-=(const base_data<this_type>& rhs)
    {
        m_real -= (rhs.stop().real() - rhs.start().real());
        m_user -= (rhs.stop().user() - rhs.start().user());
        m_syst -= (rhs.stop().sys() - rhs.start().sys());
        return *this;
    }

    this_type& operator-=(const base_delta<this_type>& rhs)
    {
        m_real -= rhs.sum().real();
        m_user -= rhs.sum().user();
        m_syst -= rhs.sum().sys();
        return *this;
    }

    this_type& operator*=(const uintmax_t& rhs)
    {
        m_real *= rhs;
        m_user *= rhs;
        m_syst *= rhs;
        return *this;
    }

    this_type& operator/=(const uintmax_t& rhs)
    {
        m_real /= rhs;
        m_user /= rhs;
        m_syst /= rhs;
        return *this;
    }

    const uintmax_t& real() const { return m_real; }
    const uintmax_t& user() const { return m_user; }
    const uintmax_t& sys() const { return m_syst; }

    void reset()
    {
        m_real = 0.0;
        m_user = 0.0;
        m_syst = 0.0;
    }

protected:
    uintmax_t m_real;
    uintmax_t m_user;
    uintmax_t m_syst;
};

//======================================================================================//

}  // namespace internal

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
