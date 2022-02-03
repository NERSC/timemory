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

#pragma once

#include <cstring>

namespace tim
{
namespace numeric
{
struct half_raw
{
    unsigned short x;
};

struct half2_raw
{
    unsigned short x;
    unsigned short y;
};

struct half;

half
float2half(float);

float half2float(half);

// BEGIN STRUCT half
struct half
{
    // CREATORS
    half() = default;

    half(const half_raw& x)
    : m_x{ x.x }
    {}

    half(float x)
    : m_x{ float2half(x).m_x }
    {}
    half(double x)
    : m_x{ float2half(x).m_x }
    {}

    half(const half&) = default;
    half(half&&)      = default;
    ~half()           = default;

    // MANIPULATORS
    half& operator=(const half&) = default;
    half& operator=(half&&) = default;
    half& operator          =(const half_raw& x)
    {
        m_x = x.x;
        return *this;
    }
    half& operator=(float x)
    {
        m_x = float2half(x).m_x;
        return *this;
    }
    half& operator=(double x) { return *this = static_cast<float>(x); }

    // ACCESSORS
    operator float() const { return half2float(*this); }
    operator half_raw() const { return half_raw{ m_x }; }

protected:
    unsigned short m_x;
};

//======================================================================================//

struct half2
{
    half x;
    half y;

    // CREATORS
    half2() = default;
    half2(const half2_raw& ix)
    : x{ reinterpret_cast<const half&>(ix.x) }
    , y{ reinterpret_cast<const half&>(ix.y) }
    {}
    half2(const half& ix, const half& iy)
    : x{ ix }
    , y{ iy }
    {}
    half2(const half2&) = default;
    half2(half2&&)      = default;
    ~half2()            = default;

    // MANIPULATORS
    half2& operator=(const half2&) = default;
    half2& operator=(half2&&) = default;
    half2& operator           =(const half2_raw& ix)
    {
        x = reinterpret_cast<const half_raw&>(ix.x);
        y = reinterpret_cast<const half_raw&>(ix.y);
        return *this;
    }

    // ACCESSORS
    operator half2_raw() const
    {
        return half2_raw{ reinterpret_cast<const unsigned short&>(x),
                          reinterpret_cast<const unsigned short&>(y) };
    }
};
// END STRUCT half2

inline unsigned short
internal_float2half(float flt, unsigned int& sgn, unsigned int& rem)
{
    unsigned int x{};
    std::memcpy(&x, &flt, sizeof(flt));

    unsigned int u = (x & 0x7fffffffU);
    sgn            = ((x >> 16) & 0x8000U);

    // NaN/+Inf/-Inf
    if(u >= 0x7f800000U)
    {
        rem = 0;
        return static_cast<unsigned short>((u == 0x7f800000U) ? (sgn | 0x7c00U)
                                                              : 0x7fffU);
    }
    // Overflows
    if(u > 0x477fefffU)
    {
        rem = 0x80000000U;
        return static_cast<unsigned short>(sgn | 0x7bffU);
    }
    // Normal numbers
    if(u >= 0x38800000U)
    {
        rem = u << 19;
        u -= 0x38000000U;
        return static_cast<unsigned short>(sgn | (u >> 13));
    }
    // +0/-0
    if(u < 0x33000001U)
    {
        rem = u;
        return static_cast<unsigned short>(sgn);
    }
    // Denormal numbers
    unsigned int exponent = u >> 23;
    unsigned int mantissa = (u & 0x7fffffU);
    unsigned int shift    = 0x7eU - exponent;
    mantissa |= 0x800000U;
    rem = mantissa << (32 - shift);
    return static_cast<unsigned short>(sgn | (mantissa >> shift));
}

inline half
float2half(float x)
{
    half_raw     r;
    unsigned int sgn{};
    unsigned int rem{};
    r.x = internal_float2half(x, sgn, rem);
    if(rem > 0x80000000U || (rem == 0x80000000U && ((r.x & 0x1) != 0)))
        ++r.x;

    return r;
}

inline half
float2half_rn(float x)
{
    return float2half(x);
}

inline half
float2half_rz(float x)
{
    half_raw     r;
    unsigned int sgn{};
    unsigned int rem{};
    r.x = internal_float2half(x, sgn, rem);

    return r;
}

inline half
float2half_rd(float x)
{
    half_raw     r;
    unsigned int sgn{};
    unsigned int rem{};
    r.x = internal_float2half(x, sgn, rem);
    if((rem != 0u) && (sgn != 0u))
        ++r.x;

    return r;
}

inline half
float2half_ru(float x)
{
    half_raw     r;
    unsigned int sgn{};
    unsigned int rem{};
    r.x = internal_float2half(x, sgn, rem);
    if((rem != 0u) && (sgn == 0u))
        ++r.x;

    return r;
}

inline half2
float2half2_rn(float x)
{
    return half2{ float2half_rn(x), float2half_rn(x) };
}

inline half2
floats2half2_rn(float x, float y)
{
    return half2{ float2half_rn(x), float2half_rn(y) };
}

inline float
internal_half2float(unsigned short x)
{
    unsigned int sign     = ((x >> 15) & 1);
    unsigned int exponent = ((x >> 10) & 0x1f);
    unsigned int mantissa = ((x & 0x3ff) << 13);

    if(exponent == 0x1fU)
    { /* NaN or Inf */
        mantissa = (mantissa != 0u ? (sign = 0, 0x7fffffU) : 0);
        exponent = 0xffU;
    }
    else if(exponent == 0u)
    { /* Denorm or Zero */
        if(mantissa != 0u)
        {
            unsigned int msb;
            exponent = 0x71U;
            do
            {
                msb = (mantissa & 0x400000U);
                mantissa <<= 1; /* normalize */
                --exponent;
            } while(msb == 0u);
            mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
        }
    }
    else
    {
        exponent += 0x70U;
    }
    unsigned int u = ((sign << 31) | (exponent << 23) | mantissa);
    float        f;
    memcpy(&f, &u, sizeof(u));

    return f;
}

inline float
half2float(half x)
{
    return internal_half2float(static_cast<half_raw>(x).x);
}

inline float
low2float(half2 x)
{
    return internal_half2float(static_cast<half2_raw>(x).x);
}

inline float
high2float(half2 x)
{
    return internal_half2float(static_cast<half2_raw>(x).y);
}
}  // namespace numeric
}  // namespace tim
