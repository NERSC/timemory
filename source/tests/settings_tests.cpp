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

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/timemory.hpp"

//--------------------------------------------------------------------------------------//

static uint64_t reference_value = 10;
static uint64_t copied_value    = 10;

//--------------------------------------------------------------------------------------//

namespace details
{
//  Get the current tests name
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
consume(long n)
{
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}

// get a random entry from vector
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class settings_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

TEST_F(settings_tests, insert_via_args)
{
    using settings_t = tim::settings;
    using strvec_t   = std::vector<std::string>;

    reference_value = 10;
    copied_value    = 10;

    settings_t* _settings = settings_t::instance();

    std::string _ref_env  = "SETTINGS_TEST_REFERENCE";
    std::string _ref_name = "test_reference";
    std::string _ref_cmd  = "--test-reference";
    std::string _ref_desc = TIMEMORY_JOIN("", "Reference value for settings_tests.",
                                          details::get_test_name());

    auto _ins = _settings->insert<uint64_t, uint64_t&>(
        _ref_env, _ref_name, _ref_desc, reference_value, strvec_t({ _ref_cmd }), 1, -2);

    EXPECT_TRUE(_ins.second);

    auto tidx_check = std::type_index(typeid(uint64_t));
    auto vidx_check = std::type_index(typeid(uint64_t&));
    auto _env_check = tim::get_env<uint64_t>(_ref_env, 0);
    auto _odr_check = _settings->ordering().back();
    auto _itr_check = _settings->find(_ref_env);

    EXPECT_EQ(_env_check, reference_value);
    EXPECT_EQ(_odr_check, _ref_env);
    ASSERT_NE(_itr_check, _settings->end());
    EXPECT_EQ(_itr_check->first, _ref_env);

    auto itr = _itr_check->second;

    EXPECT_EQ(itr->get_type_index(), tidx_check);
    EXPECT_EQ(itr->get_value_index(), vidx_check);

    EXPECT_EQ(itr->get_count(), 1);
    EXPECT_EQ(itr->get_max_count(), -2);

    EXPECT_EQ(itr->get_name(), _ref_name);
    EXPECT_EQ(itr->get_env_name(), _ref_env);
    EXPECT_EQ(itr->get_command_line().front(), _ref_cmd);

    EXPECT_TRUE(itr->matches(_ref_env, true));
    EXPECT_TRUE(itr->matches(_ref_name, true));
    EXPECT_TRUE(itr->matches(_ref_cmd, true));

    EXPECT_TRUE(itr->matches(_ref_env, false));
    EXPECT_TRUE(itr->matches(_ref_name, false));
    EXPECT_TRUE(itr->matches(_ref_cmd, false));

    uint64_t _tmp = 0;
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    auto _disp = itr->get_display();
    EXPECT_EQ(_disp["name"], _ref_name);
    EXPECT_EQ(_disp["count"], "1");
    EXPECT_EQ(_disp["max_count"], "-2");
    EXPECT_EQ(_disp["env_name"], _ref_env);
    EXPECT_EQ(_disp["description"], _ref_desc);
    EXPECT_EQ(_disp["command_line"], _ref_cmd);

    reference_value = 5;
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    itr->set<uint64_t>(4);
    EXPECT_EQ(reference_value, 4);
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    itr->set("3");
    EXPECT_EQ(reference_value, 3);
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    auto citr = itr->clone();
    EXPECT_EQ(citr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(citr->as_string(), std::to_string(reference_value));

    reference_value = 2;
    EXPECT_NE(reference_value, citr->get<uint64_t>().second);
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));
    EXPECT_NE(citr->get<uint64_t>().second, reference_value);
    EXPECT_NE(citr->as_string(), std::to_string(reference_value));

    citr->set<uint64_t>(1);
    copied_value = citr->get<uint64_t>().second;
    EXPECT_TRUE(citr->get(_tmp));
    EXPECT_EQ(reference_value, 2);
    EXPECT_EQ(citr->get<uint64_t>().second, 1);
    EXPECT_EQ(copied_value, 1);
    EXPECT_EQ(_tmp, 1);
    EXPECT_EQ(_tmp, copied_value);
    EXPECT_EQ(citr->as_string(), std::to_string(copied_value));
}

//--------------------------------------------------------------------------------------//

TEST_F(settings_tests, insert_via_pointer)
{
    using settings_t = tim::settings;
    using strvec_t   = std::vector<std::string>;

    reference_value = 10;
    copied_value    = 10;

    settings_t* _settings = settings_t::instance();

    std::string _ref_env  = "SETTINGS_TEST_COPY";
    std::string _ref_name = "test_copy";
    std::string _ref_cmd  = "--test-copy";
    std::string _ref_desc =
        TIMEMORY_JOIN("", "Copied value for settings_tests.", details::get_test_name());

    auto _ptr = std::make_shared<tim::tsettings<uint64_t>>(
        reference_value, _ref_name, _ref_env, _ref_desc, strvec_t({ _ref_cmd }), 1, -2);
    auto _ins = _settings->insert(_ptr);

    EXPECT_TRUE(_ins.second);

    auto tidx_check = std::type_index(typeid(uint64_t));
    auto vidx_check = std::type_index(typeid(uint64_t));
    auto _env_check = tim::get_env<uint64_t>(_ref_env, 0);
    auto _odr_check = _settings->ordering().back();
    auto _itr_check = _settings->find(_ref_env);

    EXPECT_EQ(_env_check, reference_value);
    EXPECT_EQ(_odr_check, _ref_env);
    ASSERT_NE(_itr_check, _settings->end());
    EXPECT_EQ(_itr_check->first, _ref_env);

    auto itr = _itr_check->second;

    EXPECT_EQ(itr->get_type_index(), tidx_check);
    EXPECT_EQ(itr->get_value_index(), vidx_check);

    EXPECT_EQ(itr->get_count(), 1);
    EXPECT_EQ(itr->get_max_count(), -2);

    EXPECT_EQ(itr->get_name(), _ref_name);
    EXPECT_EQ(itr->get_env_name(), _ref_env);
    EXPECT_EQ(itr->get_command_line().front(), _ref_cmd);

    EXPECT_TRUE(itr->matches(_ref_env, true));
    EXPECT_TRUE(itr->matches(_ref_name, true));
    EXPECT_TRUE(itr->matches(_ref_cmd, true));

    EXPECT_TRUE(itr->matches(_ref_env, false));
    EXPECT_TRUE(itr->matches(_ref_name, false));
    EXPECT_TRUE(itr->matches(_ref_cmd, false));

    uint64_t _tmp = 0;
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_EQ(itr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(_tmp, reference_value);
    EXPECT_EQ(itr->as_string(), std::to_string(reference_value));

    auto _disp = itr->get_display();
    EXPECT_EQ(_disp["name"], _ref_name);
    EXPECT_EQ(_disp["count"], "1");
    EXPECT_EQ(_disp["max_count"], "-2");
    EXPECT_EQ(_disp["env_name"], _ref_env);
    EXPECT_EQ(_disp["description"], _ref_desc);
    EXPECT_EQ(_disp["command_line"], _ref_cmd);

    reference_value = 5;
    EXPECT_TRUE(itr->get(_tmp));
    EXPECT_NE(_tmp, reference_value);
    EXPECT_NE(itr->get<uint64_t>().second, reference_value);
    EXPECT_NE(itr->as_string(), std::to_string(reference_value));

    EXPECT_TRUE(itr->set<uint64_t>(2));
    EXPECT_TRUE(itr->get(reference_value));
    EXPECT_EQ(reference_value, 2);

    auto citr = itr->clone();
    EXPECT_EQ(citr->get<uint64_t>().second, reference_value);
    EXPECT_EQ(citr->as_string(), std::to_string(reference_value));

    EXPECT_TRUE(citr->set<uint64_t>(1));
    EXPECT_NE(reference_value, citr->get<uint64_t>().second);

    copied_value = citr->get<uint64_t>().second;
    EXPECT_TRUE(citr->get(_tmp));
    EXPECT_EQ(reference_value, 2);
    EXPECT_EQ(citr->get<uint64_t>().second, 1);
    EXPECT_EQ(copied_value, 1);
    EXPECT_EQ(_tmp, 1);
    EXPECT_EQ(_tmp, copied_value);
    EXPECT_EQ(citr->as_string(), std::to_string(copied_value));
}

//--------------------------------------------------------------------------------------//
