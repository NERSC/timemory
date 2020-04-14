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

/**
 * \file timemory/operations/types/finalize_print.hpp
 * \brief Definition for various functions for finalize_print in operations
 */

#pragma once

//======================================================================================//
//
#include "timemory/operations/macros.hpp"
//
#include "timemory/operations/types.hpp"
//
#include "timemory/operations/declaration.hpp"
//
//======================================================================================//
//
#include "timemory/manager/declaration.hpp"
#include "timemory/plotting/definition.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/stream.hpp"
//
#include "timemory/mpl/math.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace tim
{
namespace operation
{
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
#if !(defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_OPERATIONS_EXTERN)) ||        \
    defined(TIMEMORY_OPERATIONS_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OPERATIONS_LINKAGE(void)
base::print::print_plot(const std::string& outfname, const std::string suffix)
{
    if(node_rank == 0)
    {
        auto plot_label = label;
        if(!suffix.empty())
            plot_label += std::string(" ") + suffix;

        plotting::plot(label, plot_label, settings::output_path(),
                       settings::dart_output(), outfname);
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OPERATIONS_LINKAGE(void)
base::print::write(std::ostream& os, stream_type stream)
{
    if(stream)
        os << *stream << std::flush;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OPERATIONS_LINKAGE(void)
base::print::print_cout(stream_type stream)
{
    printf("\n");
    write(std::cout, stream);
    printf("\n");
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_OPERATIONS_LINKAGE(void)
base::print::print_text(const std::string& outfname, stream_type stream)
{
    if(outfname.length() > 0 && stream)
    {
        std::ofstream fout(outfname.c_str());
        if(fout)
        {
            printf("[%s]|%i> Outputting '%s'...\n", label.c_str(), node_rank,
                   outfname.c_str());
            write(fout, stream);
            manager::instance()->add_text_output(label, outfname);
        }
        else
        {
            fprintf(stderr, "[storage<%s>::%s @ %i]|%i> Error opening '%s'...\n",
                    label.c_str(), __FUNCTION__, __LINE__, node_rank, outfname.c_str());
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
#endif  // !defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_OPERATIONS_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::setup()
{
    settings::indent_width<Tp, 0>(Tp::get_width());
    settings::indent_width<Tp, 1>(4);
    settings::indent_width<Tp, 2>(4);

    description = Tp::get_description();
    for(auto& itr : description)
        itr = toupper(itr);

    // find the max width
    for(const auto mitr : node_results)
    {
        for(const auto& itr : mitr)
        {
            const auto& itr_obj    = itr.data();
            const auto& itr_prefix = itr.prefix();
            const auto& itr_depth  = itr.depth();

            if(itr_depth < 0 || itr_depth > settings::max_depth() ||
               itr_depth > max_call_stack)
                continue;

            max_depth = std::max<int64_t>(max_depth, itr_depth);

            // find global max
            settings::indent_width<Tp, 0>(itr_prefix.length());
            settings::indent_width<Tp, 1>(std::log10(itr_obj.get_laps()) + 1);
            settings::indent_width<Tp, 2>(std::log10(itr_depth) + 1);
        }
    }

    auto file_exists = [](const std::string& fname) {
        std::cout << "Checking for existing input at " << fname << "...\n";
        std::ifstream inpf(fname.c_str());
        auto          success = inpf.is_open();
        inpf.close();
        return success;
    };

    auto is_minimal_json = std::is_same<trait::output_archive_t<Tp>,
                                        cereal::MinimalJSONOutputArchive>::value;
    auto is_pretty_json =
        std::is_same<trait::output_archive_t<Tp>, cereal::PrettyJSONOutputArchive>::value;
    auto fext       = (is_minimal_json || is_pretty_json) ? ".json" : ".xml";
    auto extensions = tim::delimit(settings::input_extensions(), ",; ");

    json_outfname = settings::compose_output_filename(label, fext);
    text_outfname = settings::compose_output_filename(label, ".txt");

    if(settings::diff_output())
    {
        extensions.insert(extensions.begin(), fext);
        for(auto itr : extensions)
        {
            auto inpfname = settings::compose_input_filename(label, itr);
            if(file_exists(inpfname))
            {
                json_inpfname = inpfname;
                break;
            }
        }
    }

    if(!json_inpfname.empty())
    {
        auto dext     = std::string(".diff") + fext;
        json_diffname = settings::compose_output_filename(label, dext);
        text_diffname = settings::compose_output_filename(label, ".diff.txt");
        printf("difference filenames: '%s' and '%s'\n", json_diffname.c_str(),
               text_diffname.c_str());
    }

    if(!(file_output && text_output) && !cout_output)
        return;

    write_stream(data_stream, node_results);
    data_stream->set_banner(description);

    if(node_delta.size() > 0)
    {
        write_stream(diff_stream, node_delta);
        std::stringstream ss;
        ss << description << " vs. " << json_inpfname;
        if(input_concurrency != data_concurrency)
        {
            auto delta_conc = (data_concurrency - input_concurrency);
            ss << " with " << delta_conc << " " << ((delta_conc > 0) ? "more" : "less")
               << "threads";
        }
        diff_stream->set_banner(ss.str());
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::write_stream(stream_type& stream, result_type& result_array)
{
    auto stream_fmt   = Tp::get_format_flags();
    auto stream_width = Tp::get_width();
    auto stream_prec  = Tp::get_precision();

    stream.reset(new utility::stream('|', '-', stream_fmt, stream_width, stream_prec));

    using get_return_type = decltype(std::declval<const Tp>().get());
    using compute_type    = math::compute<get_return_type>;

    auto_lock_t slk(type_mutex<decltype(std::cout)>(), std::defer_lock);
    if(!slk.owns_lock())
        slk.lock();

    auto result = get_flattened(result_array);
    for(auto itr = result.begin(); itr != result.end(); ++itr)
    {
        auto& itr_obj    = (*itr)->data();
        auto& itr_prefix = (*itr)->prefix();
        auto& itr_depth  = (*itr)->depth();
        auto  itr_laps   = itr_obj.get_laps();

        if(itr_depth < 0 || itr_depth > get_max_depth())
            continue;

        // counts the number of non-exclusive values
        int64_t nexclusive = 0;
        // the sum of the exclusive values
        get_return_type exclusive_values{};

        // if we are not at the bottom of the call stack (i.e. completely
        // inclusive)
        if(itr_depth < max_depth)
        {
            // get the next iteration
            auto eitr = itr;
            std::advance(eitr, 1);
            // counts the number of non-exclusive values
            nexclusive = 0;
            // the sum of the exclusive values
            exclusive_values = get_return_type{};
            // continue while not at end of graph until first sibling is
            // encountered
            if(eitr != result.end())
            {
                auto eitr_depth = (*eitr)->depth();
                while(eitr_depth != itr_depth)
                {
                    auto& eitr_obj = (*eitr)->data();

                    // if one level down, this is an exclusive value
                    if(eitr_depth == itr_depth + 1)
                    {
                        // if first exclusive value encountered: assign; else:
                        // combine
                        if(nexclusive == 0)
                            exclusive_values = eitr_obj.get();
                        else
                            compute_type::plus(exclusive_values, eitr_obj.get());
                        // increment. beyond 0 vs. 1, this value plays no role
                        ++nexclusive;
                    }
                    // increment iterator for next while check
                    ++eitr;
                    if(eitr == result.end())
                        break;
                    eitr_depth = (*eitr)->depth();
                }
            }
        }

        auto itr_self  = compute_type::percent_diff(exclusive_values, itr_obj.get());
        auto itr_stats = (*itr)->stats();

        bool _first = std::distance(result.begin(), itr) == 0;
        if(_first)
            operation::print_header<Tp>(itr_obj, *(stream.get()), itr_stats);

        operation::print<Tp>(itr_obj, *(stream.get()), itr_prefix, itr_laps, itr_depth,
                             itr_self, itr_stats);

        stream->add_row();
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::update_data()
{
    dmp::barrier();
    node_init        = dmp::is_initialized();
    node_rank        = dmp::rank();
    node_size        = dmp::size();
    node_results     = data->dmp_get();
    data_concurrency = data->instance_count().load();
    dmp::barrier();

    if(settings::debug())
        printf("[%s]|%i> dmp results size: %i\n", label.c_str(), node_rank,
               (int) node_results.size());

    setup();

    read_json();

    if(node_input.size() > 0 && node_rank == 0)
    {
        using input_type = decay_t<decltype(node_input)>;
        using value_type = typename input_type::value_type;
        node_delta.resize(node_input.size(), value_type{});

        size_t num_ranks = std::min<size_t>(node_input.size(), node_results.size());

        for(size_t i = 0; i < num_ranks; ++i)
        {
            for(auto& iitr : node_input.at(i))
            {
                for(auto& ritr : node_results.at(i))
                {
                    if(iitr == ritr)
                    {
                        node_delta.at(i).push_back(ritr);
                        node_delta.at(i).back() -= iitr;
                        break;
                    }
                }
            }
        }
        write_stream(diff_stream, node_delta);
        std::stringstream ss;
        ss << description << " vs. " << json_inpfname;
        if(input_concurrency != data_concurrency)
        {
            auto delta_conc = (data_concurrency - input_concurrency);
            ss << " with " << delta_conc << " " << ((delta_conc > 0) ? "more" : "less")
               << "threads";
        }
        diff_stream->set_banner(ss.str());
    }

#if defined(DEBUG)
    if(tim::settings::debug() && tim::settings::verbose() > 3)
    {
        auto indiv_results = get_flattened(node_results);
        printf("\n");
        size_t w = 0;
        for(const auto& itr : indiv_results)
            w = std::max<size_t>(w, data->get_prefix(itr->hash()).length());
        for(const auto& itr : indiv_results)
        {
            std::cout << std::setw(w) << std::left << data->get_prefix(itr->hash())
                      << " : " << itr->data();
            auto _hierarchy = itr->hierarchy();
            for(size_t i = 0; i < _hierarchy.size(); ++i)
            {
                if(i == 0)
                    std::cout << " :: ";
                std::cout << data->get_prefix(_hierarchy[i]);
                if(i + 1 < _hierarchy.size())
                    std::cout << "/";
            }
            std::cout << std::endl;
        }
        printf("\n");
    }
#endif

    if(flame_output)
        operation::finalize::flamegraph<Tp>(data, label);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Archive>
void
print<Tp, true>::print_metadata(true_type, Archive& ar, const Tp& obj)
{
    // clang-format off
    ar(cereal::make_nvp("type", obj.label_array()),
       cereal::make_nvp("description", obj.description_array()),
       cereal::make_nvp("unit_value", obj.unit_array()),
       cereal::make_nvp("unit_repr", obj.display_unit_array()));
    // clang-format on
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Archive>
void
print<Tp, true>::print_metadata(false_type, Archive& ar, const Tp& obj)
{
    // clang-format off
    ar(cereal::make_nvp("type", obj.get_label()),
       cereal::make_nvp("description", obj.get_description()),
       cereal::make_nvp("unit_value", obj.get_unit()),
       cereal::make_nvp("unit_repr", obj.get_display_unit()));
    // clang-format on
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::print_json(const std::string& outfname, result_type& results,
                            int64_t concurrency)
{
    using policy_type = policy::output_archive_t<Tp>;
    using bool_type   = typename trait::array_serialization<Tp>::type;

    if(outfname.length() > 0)
    {
        std::ofstream ofs(outfname.c_str());
        if(ofs)
        {
            auto fext = outfname.substr(outfname.find_last_of(".") + 1);
            if(fext.empty())
                fext = "unknown";
            manager::instance()->add_file_output(fext, label, outfname);
            printf("[%s]|%i> Outputting '%s'...\n", label.c_str(), node_rank,
                   outfname.c_str());

            // ensure write final block during destruction before the file is closed
            auto oa = policy_type::get(ofs);

            oa->setNextName("timemory");
            oa->startNode();

            // node
            {
                (*oa)(cereal::make_nvp("num_ranks", results.size()));
                oa->setNextName("ranks");
                oa->startNode();
                oa->makeArray();
                for(uint64_t i = 0; i < results.size(); ++i)
                {
                    if(results.at(i).empty())
                        continue;

                    oa->startNode();

                    (*oa)(cereal::make_nvp("rank", i));
                    (*oa)(cereal::make_nvp("concurrency", concurrency));
                    print_metadata(bool_type{}, *oa, results.at(i).front().data());
                    Tp::extra_serialization(*oa, 1);
                    save(*oa, results.at(i));

                    oa->finishNode();
                }
                oa->finishNode();
            }
            oa->finishNode();
        }
        if(ofs)
            ofs << std::endl;
        ofs.close();
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::print_dart()
{
    using strvector_t = std::vector<std::string>;

    // if only a specific type should be echoed
    if(settings::dart_type().length() > 0)
    {
        auto dtype = settings::dart_type();
        if(operation::echo_measurement<Tp>::lowercase(dtype) !=
           operation::echo_measurement<Tp>::lowercase(label))
        {
            dart_output = false;
            return;
        }
    }

    uint64_t _nitr = 0;

    auto indiv_results = get_flattened(node_results);
    for(auto& itr : indiv_results)
    {
        auto& itr_depth = itr->depth();

        if(itr_depth < 0 || itr_depth > max_depth)
            continue;

        // if only a specific number of measurements should be echoed
        if(settings::dart_count() > 0 && _nitr >= settings::dart_count())
            continue;

        auto&       itr_obj       = itr->data();
        auto&       itr_hierarchy = itr->hierarchy();
        strvector_t str_hierarchy{};
        for(const auto& hitr : itr_hierarchy)
            str_hierarchy.push_back(data->get_prefix(hitr));
        operation::echo_measurement<Tp>(itr_obj, str_hierarchy);
        ++_nitr;
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
print<Tp, true>::read_json()
{
    using policy_type = policy::input_archive_t<Tp>;
    // using bool_type   = typename trait::array_serialization<Tp>::type;

    if(json_inpfname.length() > 0)
    {
        std::ifstream ifs(json_inpfname.c_str());
        if(ifs)
        {
            printf("[%s]|%i> Reading '%s'...\n", label.c_str(), node_rank,
                   json_inpfname.c_str());

            size_t num_ranks = 0;
            // ensure write final block during destruction before the file is closed
            auto ia = policy_type::get(ifs);

            ia->setNextName("timemory");
            ia->startNode();

            // node
            try
            {
                (*ia)(cereal::make_nvp("num_ranks", num_ranks));
                ia->setNextName("ranks");
                ia->startNode();

                node_input.resize(num_ranks);
                for(uint64_t i = 0; i < node_input.size(); ++i)
                {
                    if(node_results.at(i).empty())
                        continue;

                    ia->startNode();

                    (*ia)(cereal::make_nvp("rank", i));
                    (*ia)(cereal::make_nvp("concurrency", input_concurrency));
                    load(*ia, node_input.at(i));

                    ia->finishNode();
                }
                ia->finishNode();
            } catch(std::exception& e)
            {
                fprintf(stderr, "[%s]> Error reading input file '%s': %s\n",
                        label.c_str(), json_inpfname.c_str(), e.what());
            }

            ia->finishNode();
        }
        else
        {
            fprintf(stderr, "[%s]|%i> Failure opening '%s' for input...\n", label.c_str(),
                    node_rank, json_inpfname.c_str());
        }
        ifs.close();
    }
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
