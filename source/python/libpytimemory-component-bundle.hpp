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

#if !defined(TIMEMORY_PYAUTO_TIMER_SOURCE)
#    define TIMEMORY_PYAUTO_TIMER_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/components/extern.hpp"
#include "timemory/containers/definition.hpp"
#include "timemory/containers/extern.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/properties.hpp"
#include "timemory/variadic/definition.hpp"

using namespace tim::component;

//======================================================================================//
//
namespace pycomponent_bundle
{
//
template <typename UserBundleT>
class pycomponent_bundle
{
public:
    using type               = UserBundleT;
    using component_bundle_t = tim::component_bundle<TIMEMORY_API, UserBundleT>;

public:
    pycomponent_bundle(component_bundle_t* _ptr = nullptr)
    : m_ptr(_ptr)
    {}

    ~pycomponent_bundle()
    {
        if(m_ptr)
            m_ptr->stop();
        delete m_ptr;
    }

    pycomponent_bundle& operator=(component_bundle_t* _ptr)
    {
        if(m_ptr)
            delete m_ptr;
        m_ptr = _ptr;
        return *this;
    }

    void start()
    {
        DEBUG_PRINT_HERE("%p, global size = %lu, instance size = %lu", m_ptr, size(),
                         (m_ptr) ? m_ptr->get<type>()->size() : 0);
        if(m_ptr)
            m_ptr->start();
    }

    void stop()
    {
        DEBUG_PRINT_HERE("%p, global size = %lu, instance size = %lu", m_ptr, size(),
                         (m_ptr) ? m_ptr->get<type>()->size() : 0);
        if(m_ptr)
            m_ptr->stop();
    }

    static size_t size() { return type::bundle_size(); }

    static void reset()
    {
        DEBUG_PRINT_HERE("size = %lu", size());
        type::reset();
    }

    template <typename... ArgsT>
    static void configure(ArgsT&&... _args)
    {
        tim::configure<type>(std::forward<ArgsT>(_args)...);
    }

private:
    component_bundle_t* m_ptr = nullptr;
};
//
//--------------------------------------------------------------------------------------//
//
namespace init
{
template <typename UserBundleT>
pycomponent_bundle<UserBundleT>*
component_bundle(const std::string& func, const std::string& file, const int line,
                 py::list args)
{
    using component_bundle_t = tim::component_bundle<TIMEMORY_API, UserBundleT>;
    using settings           = tim::settings;
    using source_loc         = tim::source_location;
    using mode               = source_loc::mode;
    namespace scope          = tim::scope;

    std::string sargs = "";
    for(auto itr : args)
    {
        try
        {
            auto v = itr.cast<std::string>();
            sargs  = (sargs.empty()) ? v : TIMEMORY_JOIN("/", sargs, v);
        } catch(...)
        {}
    }

    auto&& _scope = scope::get_default();
    auto&& _loc   = source_loc(mode::complete, func.c_str(), line, file.c_str(), sargs);
    auto&& _obj   = (settings::enabled())
                      ? new component_bundle_t(_loc.get_captured(sargs), true, _scope)
                      : nullptr;
    return new pycomponent_bundle<UserBundleT>(_obj);
}
}  // namespace init
//
//--------------------------------------------------------------------------------------//
//
template <typename UserBundleT>
void
generate(py::module& _pymod, const char* _name, const char* _doc)
{
    using bundle_t = pycomponent_bundle<UserBundleT>;

    py::class_<bundle_t> comp_bundle(_pymod, _name, _doc);

    auto configure_pybundle = [](py::list _args, bool flat_profile,
                                 bool timeline_profile) {
        std::set<TIMEMORY_COMPONENT> components;
        if(_args.size() == 0)
            components.insert(WALL_CLOCK);

        for(auto itr : _args)
        {
            std::string        _sitr = "";
            TIMEMORY_COMPONENT _citr = TIMEMORY_COMPONENTS_END;

            try
            {
                _sitr = itr.cast<std::string>();
                if(_sitr.length() > 0)
                    _citr = tim::runtime::enumerate(_sitr);
                else
                    continue;
            } catch(...)
            {}

            if(_citr == TIMEMORY_COMPONENTS_END)
            {
                try
                {
                    _citr = itr.cast<TIMEMORY_COMPONENT>();
                } catch(...)
                {}
            }

            if(_citr != TIMEMORY_COMPONENTS_END)
                components.insert(_citr);
            else
            {
                PRINT_HERE("%s", "ignoring argument that failed casting to either "
                                 "'timemory.component' and string");
            }
        }

        size_t isize = bundle_t::size();
        if(tim::settings::debug() || tim::settings::verbose() > 3)
        {
            PRINT_HERE("%s", "configuring pybundle");
        }

        tim::configure<UserBundleT>(components,
                                    tim::scope::config{ flat_profile, timeline_profile });

        if(tim::settings::debug() || tim::settings::verbose() > 3)
        {
            size_t fsize = bundle_t::size();
            if((fsize - isize) < components.size())
            {
                std::stringstream ss;
                ss << "Warning: final size " << fsize << ", input size " << isize
                   << ". Difference is less than the components size: "
                   << components.size();
                PRINT_HERE("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
            PRINT_HERE("final size: %lu, input size: %lu, components size: %lu\n",
                       (unsigned long) fsize, (unsigned long) isize,
                       (unsigned long) components.size());
        }
    };

    UserBundleT::global_init();

    //==================================================================================//
    //
    //                      Component bundle
    //
    //==================================================================================//
    comp_bundle.def(py::init(&init::component_bundle<UserBundleT>), "Initialization",
                    py::arg("func"), py::arg("file"), py::arg("line"),
                    py::arg("extra") = py::list());
    //----------------------------------------------------------------------------------//
    comp_bundle.def("start", &bundle_t::start, "Start the bundle");
    //----------------------------------------------------------------------------------//
    comp_bundle.def("stop", &bundle_t::stop, "Stop the bundle");
    //----------------------------------------------------------------------------------//
    comp_bundle.def_static(
        "configure", configure_pybundle, py::arg("components") = py::list(),
        py::arg("flat_profile") = false, py::arg("timeline_profile") = false,
        "Configure the profiler types (default: 'wall_clock')");
    //----------------------------------------------------------------------------------//
    comp_bundle.def_static("reset", &bundle_t::reset,
                           "Reset the components in the bundle");
}
//
}  // namespace pycomponent_bundle
//
//======================================================================================//
