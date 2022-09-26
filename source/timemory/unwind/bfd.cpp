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

#include "timemory/unwind/bfd.hpp"

#include "timemory/unwind/addr2line.hpp"

#include <mutex>

#if defined(TIMEMORY_USE_BFD)
#    define PACKAGE "timemory-bfd"
#    include <bfd.h>
#endif

namespace tim
{
namespace unwind
{
#if defined(TIMEMORY_USE_BFD)

namespace
{
void
initialize_bfd()
{
    static std::once_flag _once{};
    std::call_once(_once, bfd_init);
}
}  // namespace

int
bfd_error(const char* string)
{
    const char* errmsg = bfd_errmsg(bfd_get_error());
    if(string)
        TIMEMORY_PRINTF_WARNING(stderr, "%s: %s\n", string, errmsg);
    else
        TIMEMORY_PRINTF_WARNING(stderr, "%s\n", errmsg);

    return -1;
}

bfd_file::bfd_file(std::string _inp)
: name{ std::move(_inp) }
, data{ open(name) }
{
    read_symtab();
}

bfd_file::~bfd_file()
{
    delete[] reinterpret_cast<asymbol**>(syms);
    if(data)
        bfd_close(static_cast<bfd*>(data));
}

void*
bfd_file::open(const std::string& _v)
{
    initialize_bfd();

    auto* _data = bfd_openr(_v.c_str(), nullptr);
    _data->flags |= BFD_DECOMPRESS;
    if(_data && bfd_check_format(_data, bfd_object) == 0)
        return nullptr;
    return _data;
}

int
bfd_file::read_symtab()
{
    bfd_boolean dynamic = FALSE;

    bfd* _data = static_cast<bfd*>(data);
    if((bfd_get_file_flags(_data) & HAS_SYMS) == 0)
        return bfd_error(bfd_get_filename(_data));

    auto storage = bfd_get_symtab_upper_bound(_data);
    if(storage < 0L)
        return bfd_error(bfd_get_filename(_data));
    else if(storage == 0L)
    {
        storage = bfd_get_dynamic_symtab_upper_bound(_data);
        dynamic = TRUE;
    }

    auto* _syms    = new asymbol*[storage];
    auto  symcount = (dynamic != 0) ? bfd_canonicalize_dynamic_symtab(_data, _syms)
                                   : bfd_canonicalize_symtab(_data, _syms);

    if(symcount < 0)
    {
        delete[] _syms;
        return bfd_error(bfd_get_filename(_data));
    }

    syms = reinterpret_cast<void**>(_syms);

    return 0;
}

#else

int
bfd_error(const char*)
{
    return -1;
}

#endif

}  // namespace unwind
}  // namespace tim
