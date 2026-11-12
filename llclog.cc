#include "llclog.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

llclog::llclog(CACHE* cache)
    : replacement(cache), NUM_WAY(cache->NUM_WAY), last_used_cycles(static_cast<std::size_t>(cache->NUM_SET * cache->NUM_WAY), 0),
      offset_bits(cache->OFFSET_BITS)
{
  const char* p = std::getenv("LLCLOG_PATH");
  out = std::fopen(p ? p : "llc_access.bin", "wb");
  if (!out)
    std::cerr << "llclog: cannot open output file" << std::endl;
}

llclog::llclog(llclog&& o) noexcept
    : replacement(std::move(o)), NUM_WAY(o.NUM_WAY), last_used_cycles(std::move(o.last_used_cycles)), cycle(o.cycle), offset_bits(o.offset_bits),
      out(o.out), n_logged(o.n_logged)
{
  o.out = nullptr;
}

llclog::~llclog()
{
  if (out)
    std::fclose(out);
}

long llclog::find_victim(uint32_t, uint64_t, long set, const champsim::cache_block*, champsim::address, champsim::address, access_type)
{
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);
  return std::distance(begin, std::min_element(begin, end));
}

void llclog::replacement_cache_fill(uint32_t, long set, long way, champsim::address, champsim::address, champsim::address, access_type)
{
  last_used_cycles.at(static_cast<std::size_t>(set * NUM_WAY + way)) = cycle++;
}

// Called once per LLC access (hit or miss), which is exactly the stream Belady needs.
void llclog::update_replacement_state(uint32_t, long set, long way, champsim::address full_addr, champsim::address, champsim::address,
                                      access_type type, uint8_t hit)
{
  if (out) {
    struct {
      uint64_t line;
      uint32_t set;
      uint32_t type;
    } rec{full_addr.slice_upper(offset_bits).to<uint64_t>(), static_cast<uint32_t>(set), static_cast<uint32_t>(champsim::to_underlying(type))};
    std::fwrite(&rec, sizeof rec, 1, out);
    ++n_logged;
  }
  if (hit && access_type{type} != access_type::WRITE)
    last_used_cycles.at(static_cast<std::size_t>(set * NUM_WAY + way)) = cycle++;
}

void llclog::replacement_final_stats()
{
  if (out)
    std::fflush(out);
  std::cout << "llclog: logged " << n_logged << " LLC accesses" << std::endl;
}
