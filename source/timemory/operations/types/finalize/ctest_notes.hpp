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

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"

#include <set>
#include <string>

namespace tim
{
//
class manager;
//
namespace operation
{
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
struct ctest_notes_deleter : public std::default_delete<std::set<std::string>>
{
    using strset_t = std::set<std::string>;

    ctest_notes_deleter()  = default;
    ~ctest_notes_deleter() = default;

    void operator()(strset_t* data)
    {
        std::stringstream ss;
        // loop over ASCII report filenames
        for(auto&& itr : *data)
        {
#if defined(_WIN32) || defined(_WIN64)
            while(itr.find("\\") != std::string::npos)
                itr = itr.replace(itr.find("\\"), 1, "/");
#endif
            ss << "LIST(APPEND CTEST_NOTES_FILES \"" << itr << "\")\n";
        }

        if(data->size() > 0)
            ss << "LIST(REMOVE_DUPLICATES CTEST_NOTES_FILES)\n";

        auto fname = settings::compose_output_filename("CTestNotes", "txt", false, -1,
                                                       false, settings::output_prefix());
        std::ofstream ofs(fname, std::ios::out | std::ios::app);
        if(ofs)
        {
            if(settings::debug() || settings::verbose() > 1)
                std::cout << "[ctest_notes]> Outputting '" << fname << "'..."
                          << std::endl;
            ofs << ss.str() << std::endl;
        }
    }
};

//
//--------------------------------------------------------------------------------------//
//
template <>
struct ctest_notes<manager>
{
    using strset_t    = std::set<std::string>;
    using notes_ptr_t = std::unique_ptr<strset_t, ctest_notes_deleter>;

    static notes_ptr_t& get_notes()
    {
        static auto _instance = notes_ptr_t(new strset_t);
        return _instance;
    }

    ctest_notes(std::string&& fname)
    {
        get_notes()->insert(std::forward<std::string>(fname));
    }

    ctest_notes(strset_t&& fnames)
    {
        for(auto&& itr : fnames)
            get_notes()->insert(itr);
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct ctest_notes : public ctest_notes<manager>
{
    using strset_t    = std::set<std::string>;
    using base_type   = ctest_notes<manager>;
    using notes_ptr_t = typename base_type::notes_ptr_t;

    static notes_ptr_t& get_notes() { return base_type::get_notes(); }

    template <typename Tp>
    ctest_notes(Tp&& fnames)
    : base_type(std::forward<Tp>(fnames))
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
