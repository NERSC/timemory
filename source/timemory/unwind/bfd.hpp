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

#include "timemory/defines.h"
#include "timemory/log/macros.hpp"
#include "timemory/unwind/types.hpp"

#include <string>
#include <utility>

namespace tim
{
namespace unwind
{
int
bfd_error(const char* string);

int
bfd_message(int _lvl, std::string_view);

void
set_bfd_verbose(int);

#if defined(TIMEMORY_USE_BFD)

struct bfd_file
{
    explicit bfd_file(std::string);
    ~bfd_file();

    static void* open(const std::string&);

    int  read_symtab();
    bool is_good() const { return (data != nullptr) && !name.empty(); }

    operator bool() const { return is_good(); }

    std::string name = {};
    void*       data = nullptr;
    void**      syms = nullptr;
};

#else

struct bfd_file
{
    explicit bfd_file(std::string) {}
    ~bfd_file() = default;

    static void* open(const std::string&) { return nullptr; }
    static int   read_symtab() { return -1; }
    static bool  is_good() { return false; }

    operator bool() const { return false; }

    std::string_view name = {};
    void*            data = nullptr;
    void**           syms = nullptr;
};

#endif
}  // namespace unwind
}  // namespace tim
