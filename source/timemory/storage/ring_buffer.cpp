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

#include "timemory/storage/ring_buffer.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/units.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace tim
{
namespace base
{
//
ring_buffer::~ring_buffer() { destroy(); }
//
void
ring_buffer::init(size_t _size)
{
    if(m_init)
        destroy();

    m_init = true;

    // Round up to multiple of page size.
    _size += units::get_page_size() - ((_size % units::get_page_size())
                                           ? (_size % units::get_page_size())
                                           : units::get_page_size());

    m_size        = _size;
    m_read_count  = 0;
    m_write_count = 0;

    // Set file path depending on whether shared memory is compiled in or not.
#ifdef SHM
    char path[] = "/dev/shm/rb-XXXXXX";
#else
    char path[] = "/tmp/rb-XXXXXX";
#endif /* SHM */

    // Create a temporary file for mmap backing.
    if((m_fd = mkstemp(path)) < 0)
        destroy();

    // Remove file from filesystem. Note the file is still open by the
    // process.
    // XXX there might be a security problem with this, if so, use umaks 0600.
    if(unlink(path))
        destroy();

    // Resize file to buffer size.
    if(ftruncate(m_fd, m_size) < 0)
        destroy();

    // Map twice the buffer size.
    if((m_ptr = mmap(NULL, 2 * m_size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) ==
       MAP_FAILED)
        destroy();

    // Map the temporary file into the first half of the above mapped space.
    if(mmap(m_ptr, m_size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, m_fd, 0) ==
       MAP_FAILED)
        destroy();

    // Map the temporary file into the second half of the mapped space.
    // This creates two consecutive copies of the same physical memory, thus
    // allowing contiues reads and writes of the buffer.
    if(mmap(static_cast<char*>(m_ptr) + m_size, m_size, PROT_READ | PROT_WRITE,
            MAP_FIXED | MAP_SHARED, m_fd, 0) == MAP_FAILED)
        destroy();
}

void
ring_buffer::destroy()
{
    m_init = false;
    // Truncate file to zero, to avoid writing back memory to file, on munmap.
    if(ftruncate(m_fd, 0) < 0)
    {
        bool _cond = settings::verbose() > 0 || settings::debug();
        CONDITIONAL_PRINT_HERE(
            _cond, "Ring buffer failed to truncate the file descriptor %i\n", m_fd);
    }
    // Unmap the mapped virtual memmory.
    auto ret = munmap(m_ptr, m_size * 2);
    // Close the backing file.
    close(m_fd);
    if(ret)
        perror("munmap");
}

}  // namespace base
}  // namespace tim
