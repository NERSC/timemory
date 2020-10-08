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

#include "timemory/timemory.hpp"

#include "gtest/gtest.h"

#include <chrono>
#include <thread>

static int    _argc = 0;
static char** _argv = nullptr;

//--------------------------------------------------------------------------------------//

class warning_tests : public ::testing::Test
{};

//--------------------------------------------------------------------------------------//

TEST_F(warning_tests, enabled)
{
    std::stringstream ss;
    ss << "debug = True\n";
    ss << "verbose = 1\n";
    ss << "enabled = OFF\n";
    ss << "fake_test = dummy\n";

    tim::settings::debug()           = false;
    tim::settings::verbose()         = 0;
    tim::settings::enabled()         = true;
    tim::settings::config_file()     = "";
    tim::settings::suppress_config() = true;

    tim::set_env("TIMEMORY_SUPPRESS_PARSING", "ON", 1);

    tim::timemory_init(_argc, _argv);

    auto _settings = tim::settings::instance();
    auto _rsuccess = _settings->read(ss);

    _settings->parse();
    tim::set_env("TIMEMORY_GLOBAL_COMPONENTS", "wall_clock", 1);

    EXPECT_FALSE(_rsuccess) << " Unsuccessful read";
    EXPECT_TRUE(tim::settings::suppress_parsing());
    EXPECT_TRUE(tim::settings::debug());
    EXPECT_FALSE(tim::settings::enabled());
    EXPECT_EQ(tim::settings::verbose(), 1);
    EXPECT_EQ(tim::settings::global_components(), std::string{});

    EXPECT_EQ(tim::settings::suppress_parsing(), _settings->get_suppress_parsing());
    EXPECT_EQ(tim::settings::debug(), _settings->get_debug());
    EXPECT_EQ(tim::settings::enabled(), _settings->get_enabled());
    EXPECT_EQ(tim::settings::verbose(), _settings->get_verbose());
    EXPECT_EQ(tim::settings::global_components(), _settings->get_global_components());

    {
        std::ofstream ofs(".settings-test.json");
        if(ofs)
        {
            cereal::JSONOutputArchive oa(ofs);
            tim::settings::serialize_settings(oa);
        }
        if(ofs)
            ofs << '\n';
        ofs.close();
    }

    // reverse some variables
    auto orig_proc                      = tim::settings::collapse_processes();
    auto orig_thrd                      = tim::settings::collapse_threads();
    tim::settings::collapse_processes() = !orig_proc;
    tim::settings::collapse_threads()   = !orig_thrd;

    {
        std::ifstream ifs(".settings-test.json");
        if(ifs)
        {
            cereal::JSONInputArchive ia(ifs);
            tim::settings::serialize_settings(ia);
        }
        ifs.close();
    }

    EXPECT_EQ(tim::settings::collapse_processes(), orig_proc);
    EXPECT_EQ(tim::settings::collapse_threads(), orig_thrd);

    tim::timemory_finalize();
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    _argc = argc;
    _argv = argv;
    return RUN_ALL_TESTS();
}

//--------------------------------------------------------------------------------------//

// TIMEMORY_INITIALIZE_STORAGE(TIMEMORY_COMPONENTS_END)

//--------------------------------------------------------------------------------------//
