
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>

using auto_lock_t = std::unique_lock<std::mutex>;

extern "C"
{
    void kokkosp_init_library(const int, const uint64_t, const uint32_t, void*);
    void kokkosp_finalize_library();
    void kokkosp_begin_parallel_for(const char*, uint32_t, uint64_t*);
    void kokkosp_end_parallel_for(uint64_t);
    void kokkosp_begin_parallel_scan(const char*, uint32_t, uint64_t*);
    void kokkosp_end_parallel_scan(uint64_t);
    void kokkosp_begin_parallel_reduce(const char*, uint32_t, uint64_t*);
    void kokkosp_end_parallel_reduce(uint64_t);

    void kokkosp_push_profile_region(const char* name);
    void kokkosp_pop_profile_region();
    void kokkosp_create_profile_section(const char* name, uint32_t* sec_id);
    void kokkosp_destroy_profile_section(uint32_t sec_id);
    void kokkosp_start_profile_section(uint32_t sec_id);
    void kokkosp_stop_profile_section(uint32_t sec_id);
}

template <typename Tp>
Tp
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

int
main(int argc, char** argv)
{
    kokkosp_init_library(0, 0, 0, nullptr);

    std::string exe = argv[0];
    if(exe.find('/') != std::string::npos)
        exe = exe.substr(exe.find_last_of('/') + 1);

    // start recording application
    kokkosp_push_profile_region(exe.c_str());

    long nfib = 47;
    if(argc > 1)
        nfib = atol(argv[1]);

    // some data
    std::vector<std::vector<int64_t>> vv(10);
    std::vector<std::thread>          threads;
    std::mutex                        _mutex;

    // vary the time slightly
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // create "reduce" over thread creation
    uint64_t r_id;
    kokkosp_begin_parallel_reduce("thread_creation", 0, &r_id);

    // launch the threads
    for(int j = 0; j < 10; ++j)
    {
        // create a profile for the memory
        uint32_t psec_id;
        kokkosp_create_profile_section("memory_section", &psec_id);

        auto _function = [&](int i) {
            uint64_t id = 0;
            kokkosp_begin_parallel_for("fibonacci", 0, &id);

            // vary the time slightly
            if(i % 3 == 2)
                std::this_thread::sleep_for(std::chrono::seconds((i + 1)));

            // create some memory and use it so it can't be optimized away
            auto                 n    = 500000 * (i + 1);
            long                 cfib = 0;
            std::vector<int64_t> v(0);
            {
                // for consistent memory allocation results
                auto_lock_t lk(_mutex);
                // start profiling memory
                kokkosp_start_profile_section(psec_id);
                v    = std::vector<int64_t>(n, nfib);
                cfib = random_entry(v);
                // stop profile memory
                kokkosp_stop_profile_section(psec_id);
            }

            uint64_t s_id;
            // start the collection
            auto label = std::string("fibonacci_runtime_") + std::to_string(i);
            kokkosp_begin_parallel_scan(label.c_str(), 0, &s_id);
            auto ret = fibonacci(cfib);
            // end the collection
            kokkosp_end_parallel_scan(s_id);

            // use the return value of function so it can't be optimized away
            printf("fibonacci(%li) = %li\n", cfib, ret);

            // for consistent memory allocation results
            {
                auto_lock_t lk(_mutex);
                // clear some memory so the memory fields change
                if(i % 4 == 3)
                    vv[i] = std::vector<int64_t>(0);
                else
                    vv[i] = std::move(v);
            }
            kokkosp_end_parallel_for(id);
        };
        // destroy the memory profile section
        kokkosp_destroy_profile_section(psec_id);

        threads.emplace_back(std::thread(_function, j));
        // threads.back().join();
    }

    // end "reduction" over the thread-creation
    kokkosp_end_parallel_reduce(r_id);

    for(auto& itr : threads)
        itr.join();
    threads.clear();

    // end the profile region
    kokkosp_pop_profile_region();

    kokkosp_finalize_library();
}
