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

/** \file serializer.hpp
 * \headerfile serializer.hpp "timemory/serializer.hpp"
 * Serialization not using Cereal
 * Not currently finished
 */

#ifndef serializer_hpp_
#define serializer_hpp_

#include "timemory/macros.hpp"

#include <type_traits>
#include <string>
#include <ostream>
#include <istream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <ios>

//============================================================================//

namespace serializer
{

//----------------------------------------------------------------------------//

template <bool B, typename T = void>
using _enable_if_t = typename enable_if<B, T>::type;

//----------------------------------------------------------------------------//

inline std::string quote()
{
    return "\"";
}

//----------------------------------------------------------------------------//

template <typename _Tp>
std::string as_string(_Tp val)
{
    std::stringstream ss;
    ss << val;
    return ss.str();
}

//----------------------------------------------------------------------------//

class tag
{
public:
    tag(const std::string& _str) : m_tag(_str) { }
    std::string& operator()() { return m_tag; }
    const std::string& operator()() const { return m_tag; }

    friend std::ostream& operator<<(std::ostream& os, const tag& obj)
    {
        os << quote() << obj() << quote() << " : ";
        return os;
    }

protected:
    std::string m_tag;
};

//----------------------------------------------------------------------------//

template <typename _Os, typename _Tp,
          typename = _enable_if_t<std::is_integral<_Tp>::value>>
void serialize_object(_Os& os, const _Tp& obj)
{
    std::stringstream ss;
    ss << obj;
    os << ss.str();
}

//----------------------------------------------------------------------------//

template <typename _Os, typename _Tp,
          typename = _enable_if_t<std::is_floating_point<_Tp>::value>>
void serialize_object(_Os& os, const _Tp& obj)
{
    std::stringstream ss;
    ss << std::hexfloat << obj;
    os << ss.str();
}

//----------------------------------------------------------------------------//

template <typename _Os, typename _Tp,
          typename = _enable_if_t<std::is_same<_Tp, std::string>::value>>
void serialize_object(_Os& os, const _Tp& obj)
{
    std::stringstream ss;
    ss << quote() << obj << quote();
    os << ss.str();
}

//----------------------------------------------------------------------------//

template <typename _Os, typename _Key, typename _Tp>
void serialize_object(_Os& os, const std::map<_Key, _Tp>& obj)
{
    std::stringstream ss;
    ss << "[ ";
    for(const auto& itr : obj)
    {
        tag _tag(as_string(itr.first));
        ss << _tag;
        object<_Tp> _obj(itr.second);
        ss << _obj;
    }
    ss << " ]";
    os << ss.str();
}

//----------------------------------------------------------------------------//

template <typename _Tp>
class object
{
public:
    typedef typename std::remove_cv<_Tp>::type  value_type;

public:
    object(value_type _val) : m_obj(_val) { }
    value_type& operator()() { return m_obj; }
    const value_type& operator()() const { return m_obj; }

    template <typename _Up>
    friend std::ostream& operator<<(std::ostream& os, const object<_Up>& obj)
    {
        serialize_object<std::ostream, _Up>(os, obj());
        os << ", ";
        return os;
    }

protected:
    value_type m_obj;
};



}

//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//

#endif

