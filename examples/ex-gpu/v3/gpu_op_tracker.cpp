
#include "gpu_op_tracker.hpp"

#include "timemory/timemory.hpp"

namespace tim
{
namespace component
{
gpu_op_tracker::gpu_op_tracker(const gpu_op_tracker& rhs)
: base_type{ rhs }
, m_copy{ true }
, m_device_num{ rhs.m_device_num }
, m_blocks{ rhs.m_blocks }
, m_prefix{ rhs.m_prefix }
, m_stream{ rhs.m_stream }
, m_data{ rhs.m_data }
, m_device_data{ rhs.m_device_data }
{}

gpu_op_tracker&
gpu_op_tracker::operator=(const gpu_op_tracker& rhs)
{
    if(this != &rhs)
    {
        base_type::operator=(rhs);
        m_copy             = true;
        m_device_num       = rhs.m_device_num;
        m_blocks           = rhs.m_blocks;
        m_prefix           = rhs.m_prefix;
        m_stream           = rhs.m_stream;
        m_data             = rhs.m_data;
        m_device_data      = rhs.m_device_data;
    }
    return *this;
}

void
gpu_op_tracker::preinit()
{
    gpu_device_op_data::label()       = label();
    gpu_device_op_data::description() = description();
}

std::string
gpu_op_tracker::label()
{
    return "gpu_op_tracker";
}

std::string
gpu_op_tracker::description()
{
    return "Counts user specified number of operations (e.g. flops)";
}

void
gpu_op_tracker::stop()
{
    if(m_device_data)
    {
        std::vector<int> _host(m_blocks, 0);

        TIMEMORY_GPU_RUNTIME_API_CALL(
            gpu::memcpy(_host.data(), m_data, m_blocks, gpu::device_to_host_v, m_stream));

        gpu::stream_sync(m_stream);

        tracker_type _child{ "", tim::scope::tree{} };
        for(size_t i = 0; i < _host.size(); ++i)
        {
            auto itr = _host[i];
            if(itr == 0)
                continue;

            m_tracker.store(std::plus<int64_t>{}, static_cast<int64_t>(itr));

            if(settings::add_secondary())
            {
                // secondary data
                _child.rekey(TIMEMORY_JOIN('_', "thread", i));
                _child.push()
                    .start()
                    .store(std::plus<int64_t>{}, static_cast<int64_t>(itr))
                    .stop()
                    .pop()
                    .reset();
            }
        }

        m_tracker.stop();
    }
}

gpu_op_tracker::device_data
gpu_op_tracker::get_device_data() const
{
    return device_data{ static_cast<uint32_t>(m_blocks), m_data };
}

size_t&
gpu_op_tracker::max_blocks()
{
    static size_t _value = 0;
    return _value;
}
}  // namespace component
}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(gpu_op_tracker, false, void)
