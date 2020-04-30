//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

/** \file timemory/tools/available.hpp
 * \headerfile tools/available.hpp "tools/available.hpp"
 * Handles serializing the settings
 *
 */

#pragma once

#define TIMEMORY_DISABLE_BANNER

#include "cereal/external/base64.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/utility.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <vector>

#if !defined(CEREAL_EPILOGUE_FUNCTION_NAME)
#    define CEREAL_EPILOGUE_FUNCTION_NAME epilogue
#endif

#if !defined(CEREAL_PROLOGUE_FUNCTION_NAME)
#    define CEREAL_PROLOGUE_FUNCTION_NAME prologue
#endif

//======================================================================================//

namespace cereal
{
class SettingsTextArchive
: public OutputArchive<SettingsTextArchive>
, public traits::TextArchive
{
public:
    using width_type   = std::array<uint64_t, 5>;
    using value_type   = std::array<std::string, 5>;
    using array_type   = std::vector<value_type>;
    using exclude_type = std::set<std::string>;

public:
    //! Construct, outputting to the provided stream
    /// @param stream The array of output data
    SettingsTextArchive(array_type& stream, exclude_type exclude)
    : OutputArchive<SettingsTextArchive>(this)
    , output_stream(&stream)
    , exclude_stream(exclude)
    {
        name_counter.push(0);
    }

    ~SettingsTextArchive() {}

    void saveBinaryValue(const void* data, size_t size, const char* name = nullptr)
    {
        setNextName(name);
        writeName();

        auto base64string =
            base64::encode(reinterpret_cast<const unsigned char*>(data), size);
        saveValue(base64string);
    }

    void startNode() { name_counter.push(0); }

    void finishNode()
    {
        name_counter.pop();
        current_stream = nullptr;
    }

    //! Sets the name for the next node created with startNode
    void setNextName(const char* name)
    {
        if(exclude_stream.count(name) == 0)
        {
            output_stream->push_back(value_type{});
            current_stream           = &output_stream->back();
            current_stream->at(0)    = name;
            std::string       func   = name;
            const std::string prefix = "TIMEMORY_";
            func                     = func.erase(0, prefix.length());
            std::transform(func.begin(), func.end(), func.begin(),
                           [](char& c) { return tolower(c); });
            std::stringstream ss;
            ss << "settings::" << func << "()";
            current_stream->at(3) = ss.str();
        }
        else
        {
            current_stream = nullptr;
        }
    }

    void setNextType(const char* name)
    {
        if(current_stream)
        {
            std::stringstream ss;
            ss << std::boolalpha << name;
            current_stream->at(2) = ss.str();
        }
    }

public:
    template <typename Tp>
    inline void saveValue(Tp _val)
    {
        if(current_stream)
        {
            std::stringstream ssval;
            ssval << std::boolalpha << _val;
            current_stream->at(1) = ssval.str();
        }
        current_stream = nullptr;
    }

    void writeName() {}

    void makeArray() { current_stream = nullptr; }

private:
    value_type*          current_stream = nullptr;
    array_type*          output_stream;
    exclude_type         exclude_stream;
    std::stack<uint32_t> name_counter;  //!< Counter for creating unique names
};

//======================================================================================//
//
//      prologue and epilogue functions
//
//======================================================================================//

//--------------------------------------------------------------------------------------//
//! Prologue for NVPs for settings archive
/*! NVPs do not start or finish nodes - they just set up the names */
template <typename T>
inline void
CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive&, const NameValuePair<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Epilogue for NVPs for settings archive
/*! NVPs do not start or finish nodes - they just set up the names */
template <typename T>
inline void
CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive&, const NameValuePair<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Prologue for deferred data for settings archive
/*! Do nothing for the defer wrapper */
template <typename T>
inline void
CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive&, const DeferredData<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Epilogue for deferred for settings archive
/*! NVPs do not start or finish nodes - they just set up the names */
template <typename T>
inline void
CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive&, const DeferredData<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Prologue for SizeTags for settings archive
/*! SizeTags are ignored */
template <typename T>
inline void
CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive& ar, const SizeTag<T>&)
{
    ar.makeArray();
}

//--------------------------------------------------------------------------------------//
//! Epilogue for SizeTags for settings archive
/*! SizeTags are ignored */
template <typename T>
inline void
CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive&, const SizeTag<T>&)
{}

//--------------------------------------------------------------------------------------//
//! Prologue for all other types for settings archive
/*! Starts a new node, named either automatically or by some NVP,
    that may be given data by the type about to be archived*/
template <typename T>
inline void
CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive& ar, const T&)
{
    ar.startNode();
}

//--------------------------------------------------------------------------------------//
//! Epilogue for all other types other for settings archive
/*! Finishes the node created in the prologue*/
template <typename T>
inline void
CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive& ar, const T&)
{
    ar.finishNode();
}

//--------------------------------------------------------------------------------------//
//! Prologue for arithmetic types for settings archive
inline void
CEREAL_PROLOGUE_FUNCTION_NAME(SettingsTextArchive& ar, const std::nullptr_t&)
{}

//--------------------------------------------------------------------------------------//
//! Epilogue for arithmetic types for settings archive
inline void
CEREAL_EPILOGUE_FUNCTION_NAME(SettingsTextArchive&, const std::nullptr_t&)
{}

//======================================================================================//
//
//  Common serialization functions
//
//======================================================================================//

//! Serializing NVP types
template <typename T>
inline void
CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive& ar, const NameValuePair<T>& t)
{
    ar.setNextName(t.name);
    if(std::is_same<T, std::string>::value)
        ar.setNextType("string");
    else
        ar.setNextType(tim::demangle<T>().c_str());
    ar(t.value);
}

template <typename CharT, typename Traits, typename Alloc>
inline void
CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive& ar,
                          const NameValuePair<std::basic_string<CharT, Traits, Alloc>>& t)
{
    ar.setNextName(t.name);
    ar.setNextType("string");
    ar(t.value);
}

//! Saving for nullptr
inline void
CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive&, const std::nullptr_t&)
{}

//! Saving for arithmetic
template <typename T, traits::EnableIf<std::is_arithmetic<T>::value> = traits::sfinae>
inline void
CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive& ar, const T& t)
{
    if(std::is_same<T, std::string>::value)
        ar.setNextType("string");
    ar.saveValue(t);
}

//! saving string
template <typename CharT, typename Traits, typename Alloc>
inline void
CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive&                           ar,
                          const std::basic_string<CharT, Traits, Alloc>& str)
{
    ar.setNextType("string");
    ar.saveValue(str);
}

//--------------------------------------------------------------------------------------//
//! Saving SizeTags
template <typename T>
inline void
CEREAL_SAVE_FUNCTION_NAME(SettingsTextArchive&, const SizeTag<T>&)
{
    // nothing to do here, we don't explicitly save the size
}

}  // namespace cereal

// register archives for polymorphic support
CEREAL_REGISTER_ARCHIVE(SettingsTextArchive)
