// LRU that also logs every LLC access (set, line address, type) to a binary file, so Belady's
// optimal policy can be computed offline on the exact same access stream.
// Output: $LLCLOG_PATH (default llc_access.bin), 16-byte records: u64 line, u32 set, u32 type.
#ifndef REPLACEMENT_LLCLOG_H
#define REPLACEMENT_LLCLOG_H

#include <cstdio>
#include <vector>

#include "cache.h"
#include "modules.h"

class llclog : public champsim::modules::replacement
{
  long NUM_WAY;
  std::vector<uint64_t> last_used_cycles;
  uint64_t cycle = 0;
  champsim::data::bits offset_bits;
  std::FILE* out = nullptr;
  uint64_t n_logged = 0;

public:
  explicit llclog(CACHE* cache);
  ~llclog();
  llclog(const llclog&) = delete;
  llclog& operator=(const llclog&) = delete;
  llclog(llclog&& other) noexcept;
  llclog& operator=(llclog&&) = delete;

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                   champsim::address full_addr, access_type type);
  void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                              access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                access_type type, uint8_t hit);
  void replacement_final_stats();
};

#endif
