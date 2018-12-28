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
//
//----------------------------------------------------------------------------//
//  class header file
//
//  string
//
//  Class description:
//
//  Definition of a string.

//----------------------------------------------------------------------------//

#ifndef string_hpp_
#define string_hpp_

#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdio.h>
#include <string>

#ifdef WIN32
#    define strcasecmp _stricmp
#endif

//----------------------------------------------------------------------------//

namespace tim
{
//----------------------------------------------------------------------------//

class string : public std::basic_string<char>
{
public:
    typedef std::basic_string<char> stl_string;
    typedef uintmax_t               size_type;

public:
    enum caseCompare
    {
        exact,
        ignoreCase
    };
    enum stripType
    {
        leading,
        trailing,
        both
    };

    inline string();
    inline string(char);
    inline string(const char*);
    inline string(const char*, size_type);
    inline string(const string&);
    inline string(const std::string&);
    ~string() {}

    inline string& operator=(const string&);
    inline string& operator=(const std::string&);
    inline string& operator=(const char*);

    inline char  operator()(size_type) const;
    inline char& operator()(size_type);

    inline string&    operator+=(const char*);
    inline string&    operator+=(const std::string&);
    inline string&    operator+=(const char&);
    inline string&    operator+=(const tim::string&);
    friend stl_string operator+(const tim::string& lhs, const tim::string& rhs)
    {
        return tim::string(lhs) += rhs;
    }
    friend stl_string operator+(const char* lhs, const tim::string& rhs)
    {
        return tim::string(lhs) += rhs;
    }
    // friend tim::string operator+(const char* lhs, const stl_string& rhs)
    //{ return tim::string(lhs) += tim::string(rhs); }
    friend stl_string operator+(const tim::string& lhs, const char* rhs)
    {
        return tim::string(lhs) += rhs;
    }

    inline bool operator==(const string&) const;
    inline bool operator==(const char*) const;
    inline bool operator==(const stl_string&) const;
    inline bool operator!=(const string&) const;
    inline bool operator!=(const char*) const;
    inline bool operator!=(const stl_string&) const;

    // inline operator const char*() const;
    // inline operator stl_string() const { return
    // static_cast<stl_string>(*this);
    // }
    inline string operator()(size_type, size_type);

    inline intmax_t compareTo(const char*, caseCompare mode = exact) const;
    inline intmax_t compareTo(const string&, caseCompare mode = exact) const;

    inline string& prepend(const char*);
    inline string& append(const string&);

    inline std::istream& readLine(std::istream&, bool skipWhite = true);

    inline string& replace(uintmax_t, uintmax_t, const char*, uintmax_t);
    inline string& replace(size_type, size_type, const char*);

    inline string& remove(size_type);
    inline string& remove(size_type, size_type);

    inline intmax_t first(char) const;
    inline intmax_t last(char) const;

    inline bool contains(const std::string&) const;
    inline bool contains(char) const;

    // stripType = 0 beginning
    // stripType = 1 end
    // stripType = 2 both
    //
    inline string strip(intmax_t strip_Type = trailing, char c = ' ');

    inline void toLower();
    inline void toUpper();

    inline bool isNull() const;

    inline size_type index(const char*, intmax_t pos = 0) const;
    inline size_type index(char, intmax_t pos = 0) const;
    inline size_type index(const string&, size_type, size_type,
                           caseCompare) const;

    inline const char* data() const;

    inline intmax_t strcasecompare(const char*, const char*) const;

    inline uintmax_t hash(caseCompare cmp = exact) const;
    inline uintmax_t stlhash() const;
};

}  // namespace tim

//----------------------------------------------------------------------------//

inline tim::string::string() {}

//----------------------------------------------------------------------------//

inline tim::string::string(const char* astring)
: stl_string(astring)
{
}

//----------------------------------------------------------------------------//

inline tim::string::string(const char* astring, uintmax_t len)
: stl_string(astring, len)
{
}

//----------------------------------------------------------------------------//

inline tim::string::string(char ch)
{
    char str[2];
    str[0]              = ch;
    str[1]              = '\0';
    stl_string::operator=(str);
}

//----------------------------------------------------------------------------//

inline tim::string::string(const tim::string& str)
: stl_string(str)
{
}

//----------------------------------------------------------------------------//

inline tim::string::string(const std::string& str)
: stl_string(str)
{
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::operator=(const tim::string& str)
{
    if(&str == this)
        return *this;
    stl_string::operator=(str);
    return *this;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::operator=(const std::string& str)
{
    stl_string::operator=(str);
    return *this;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::operator=(const char* str)
{
    stl_string::operator=(str);
    return *this;
}

//----------------------------------------------------------------------------//
//
inline char
tim::string::operator()(uintmax_t i) const
{
    return operator[](i);
}

//----------------------------------------------------------------------------//

inline char&
tim::string::operator()(uintmax_t i)
{
    return stl_string::operator[](i);
}

//----------------------------------------------------------------------------//

inline tim::string
tim::string::operator()(uintmax_t start, uintmax_t extent)
{
    return tim::string(substr(start, extent));
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::operator+=(const char* str)
{
    stl_string::operator+=(str);
    return *this;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::operator+=(const std::string& str)
{
    stl_string::operator+=(str);
    return *this;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::operator+=(const char& ch)
{
    stl_string::operator+=(ch);
    return *this;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::operator+=(const tim::string& str)
{
    tim::string::operator+=(str.c_str());
    return *this;
}

//----------------------------------------------------------------------------//

inline bool
tim::string::operator==(const tim::string& str) const
{
    if(length() != str.length())
        return false;
    return (stl_string::compare(str) == 0);
}
//----------------------------------------------------------------------------//

inline bool
tim::string::operator==(const stl_string& str) const
{
    if(length() != str.length())
        return false;
    return (stl_string::compare(str) == 0);
}

//----------------------------------------------------------------------------//

inline bool
tim::string::operator==(const char* str) const
{
    return (stl_string::compare(str) == 0);
}

//----------------------------------------------------------------------------//

inline bool
tim::string::operator!=(const tim::string& str) const
{
    return !(*this == str);
}

//----------------------------------------------------------------------------//

inline bool
tim::string::operator!=(const stl_string& str) const
{
    return !(*this == str);
}

//----------------------------------------------------------------------------//

inline bool
tim::string::operator!=(const char* str) const
{
    return !(*this == str);
}

//----------------------------------------------------------------------------//
/*
inline tim::string::operator const char*() const
{
    return c_str();
}
*/
//----------------------------------------------------------------------------//

inline intmax_t
tim::string::strcasecompare(const char* s1, const char* s2) const
{
    char* buf1 = new char[strlen(s1) + 1];
    char* buf2 = new char[strlen(s2) + 1];

    for(uintmax_t i = 0; i <= strlen(s1); ++i)
        buf1[i] = tolower(char(s1[i]));
    for(uintmax_t j = 0; j <= strlen(s2); ++j)
        buf2[j] = tolower(char(s2[j]));

    intmax_t res = strcmp(buf1, buf2);
    delete[] buf1;
    delete[] buf2;
    return res;
}

//----------------------------------------------------------------------------//

inline intmax_t
tim::string::compareTo(const char* str, caseCompare mode) const
{
    return (mode == exact) ? strcmp(c_str(), str)
                           : strcasecompare(c_str(), str);
}

//----------------------------------------------------------------------------//

inline intmax_t
tim::string::compareTo(const tim::string& str, caseCompare mode) const
{
    return compareTo(str.c_str(), mode);
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::prepend(const char* str)
{
    insert(0, str);
    return *this;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::append(const tim::string& str)
{
    stl_string::operator+=(str);
    return *this;
}

//----------------------------------------------------------------------------//

inline std::istream&
tim::string::readLine(std::istream& strm, bool skipWhite)
{
    char tmp[1024];
    if(skipWhite)
    {
        strm >> std::ws;
        strm.getline(tmp, 1024);
        *this = tmp;
    }
    else
    {
        strm.getline(tmp, 1024);
        *this = tmp;
    }
    return strm;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::replace(uintmax_t start, uintmax_t nbytes, const char* buff,
                     uintmax_t n2)
{
    stl_string::replace(start, nbytes, buff, n2);
    return *this;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::replace(uintmax_t pos, uintmax_t n, const char* str)
{
    stl_string::replace(pos, n, str);
    return *this;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::remove(uintmax_t n)
{
    if(n < size())
    {
        erase(n, size() - n);
    }
    return *this;
}

//----------------------------------------------------------------------------//

inline tim::string&
tim::string::remove(uintmax_t pos, uintmax_t N)
{
    erase(pos, N + pos);
    return *this;
}

//----------------------------------------------------------------------------//

inline intmax_t
tim::string::first(char ch) const
{
    return find(ch);
}

//----------------------------------------------------------------------------//

inline intmax_t
tim::string::last(char ch) const
{
    return rfind(ch);
}

//----------------------------------------------------------------------------//

inline bool
tim::string::contains(const std::string& str) const
{
    return (stl_string::find(str) != stl_string::npos);
}

//----------------------------------------------------------------------------//

inline bool
tim::string::contains(char ch) const
{
    return (stl_string::find(ch) != stl_string::npos);
}

//----------------------------------------------------------------------------//

inline tim::string
tim::string::strip(intmax_t strip_Type, char ch)
{
    string retVal = *this;
    if(length() == 0)
        return retVal;

    uintmax_t i = 0;
    switch(strip_Type)
    {
        case leading:
        {
            for(i = 0; i < length(); i++)
                if(stl_string::operator[](i) != ch)
                    break;
            retVal = substr(i, length() - i);
        }
        break;
        case trailing:
        {
            intmax_t j = 0;
            for(j = length() - 1; j >= 0; --j)
                if(stl_string::operator[](j) != ch)
                    break;
            retVal = substr(0, j + 1);
        }
        break;
        case both:
        {
            for(i = 0; i < length(); ++i)
                if(stl_string::operator[](i) != ch)
                    break;
            string   tmp(substr(i, length() - i));
            intmax_t k = 0;
            for(k = tmp.length() - 1; k >= 0; --k)
                if(tmp.stl_string::operator[](k) != ch)
                    break;
            retVal = tmp.substr(0, k + 1);
        }
        break;
        default: break;
    }
    return retVal;
}

//----------------------------------------------------------------------------//

inline void
tim::string::toLower()
{
    for(uintmax_t i = 0; i < size(); i++)
    {
        stl_string::operator[](i) = tolower(char(stl_string::operator[](i)));
    }
}

//----------------------------------------------------------------------------//

inline void
tim::string::toUpper()
{
    for(uintmax_t i = 0; i < size(); ++i)
    {
        stl_string::operator[](i) = toupper(char(stl_string::operator[](i)));
    }
}

//----------------------------------------------------------------------------//

inline bool
tim::string::isNull() const
{
    return empty();
}

//----------------------------------------------------------------------------//
//
inline uintmax_t
tim::string::index(const string& str, uintmax_t ln, uintmax_t st,
                   tim::string::caseCompare) const
{
    return stl_string::find(str.c_str(), st, ln);
}

//----------------------------------------------------------------------------//

inline uintmax_t
tim::string::index(const char* str, intmax_t pos) const
{
    return stl_string::find(str, pos);
}

//----------------------------------------------------------------------------//

inline uintmax_t
tim::string::index(char ch, intmax_t pos) const
{
    return stl_string::find(ch, pos);
}

//----------------------------------------------------------------------------//

inline const char*
tim::string::data() const
{
    return c_str();
}

//----------------------------------------------------------------------------//

inline uintmax_t tim::string::hash(caseCompare) const
{
    const char* str = c_str();
    uintmax_t   h   = 0;
    for(; *str; ++str)
        h = 5 * h + *str;

    return uintmax_t(h);
}

//----------------------------------------------------------------------------//

inline uintmax_t
tim::string::stlhash() const
{
    const char*   str = c_str();
    unsigned long h   = 0;
    for(; *str; ++str)
        h = 5 * h + *str;

    return uintmax_t(h);
}

//----------------------------------------------------------------------------//

namespace std
{
//----------------------------------------------------------------------------//

template <> struct hash<tim::string>
{
public:
    hash()
    : m_hash(0)
    {
    }
    hash(const tim::string& obj)
    : m_hash((*this)(obj))
    {
    }

    std::size_t operator()(const tim::string& obj)
    {
        return std::hash<std::string>()(std::string(obj.c_str()));
    }

private:
    std::size_t m_hash;
};

//----------------------------------------------------------------------------//

}  // namespace std

//----------------------------------------------------------------------------//

#endif  // string_hpp_
