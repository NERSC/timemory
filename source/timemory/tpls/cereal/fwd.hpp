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

#pragma once

#include <string>
#include <type_traits>

// forward decls
namespace tim
{
namespace cereal
{
class BinaryInputArchive;
class BinaryOutputArchive;
class PortableBinaryInputArchive;
class PortableBinaryOutputArchive;
class XMLInputArchive;
class XMLOutputArchive;
//
struct MinimalJsonWriter;
struct PrettyJsonWriter;
//
template <typename WriterType = PrettyJsonWriter>
class BaseJSONOutputArchive;
//
using JSONOutputArchive        = BaseJSONOutputArchive<PrettyJsonWriter>;
using PrettyJSONOutputArchive  = BaseJSONOutputArchive<PrettyJsonWriter>;
using MinimalJSONOutputArchive = BaseJSONOutputArchive<MinimalJsonWriter>;
//
class JSONInputArchive;
//
template <class T>
class NameValuePair;
//
template <class T>
NameValuePair<T>
make_nvp(const std::string& name, T&& value);
//
template <class T>
NameValuePair<T>
make_nvp(const char* name, T&& value);
// json are always included so no forward decl necessary
}  // namespace cereal
}  // namespace tim

#include "timemory/tpls/cereal/cereal/details/helpers.hpp"
