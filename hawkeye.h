// Hawkeye LLC replacement (Jain & Lin, ISCA 2016), ported to the ChampSim module API.
// Follows the CRC-2 Hawkeye design: OPTgen on sampled sets trains a PC-indexed
// predictor; lines from cache-friendly PCs insert at RRPV 0, cache-averse at RRPV 7.
// Differences from the CRC-2 code: sampler uses full line addresses (no partial tags),
// no separate prefetch predictor (our configs run without prefetchers).
#ifndef REPLACEMENT_HAWKEYE_H
#define REPLACEMENT_HAWKEYE_H

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "cache.h"
#include "modules.h"

struct hawkeye : public champsim::modules::replacement {
  static constexpr int MAX_RRPV = 7;              // 3-bit RRIP
  static constexpr std::size_t PRED_SIZE = 2048;  // predictor entries
  static constexpr int PRED_MAX = 31;             // 5-bit counters
  static constexpr int PRED_INIT = (PRED_MAX + 1) / 2;
  static constexpr long SAMPLED_SETS = 64;

  struct history_entry {
    uint64_t pc = 0;
    uint64_t last_quanta = 0;
    uint64_t lru = 0;
  };

  struct optgen {
    std::vector<uint32_t> occupancy;
    uint32_t capacity;
    optgen(std::size_t size, uint32_t cap) : occupancy(size, 0), capacity(cap) {}
    void add_access(uint64_t q) { occupancy[q % occupancy.size()] = 0; }
    bool should_cache(uint64_t curr, uint64_t last);
  };

  long NUM_SET, NUM_WAY;
  champsim::data::bits offset_bits;
  long sample_stride;
  std::size_t window;  // OPTgen history = 8x associativity, in per-set access quanta

  std::vector<int> rrpv;
  std::vector<uint64_t> line_pc;
  std::vector<int> predictor;

  std::vector<optgen> opt;
  std::vector<uint64_t> set_timer;
  std::vector<std::unordered_map<uint64_t, history_entry>> sampler;

  uint64_t stat_friendly = 0, stat_averse = 0;

  explicit hawkeye(CACHE* cache);

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                   champsim::address full_addr, access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                champsim::address victim_addr, access_type type, uint8_t hit);
  void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                              champsim::address victim_addr, access_type type);
  void replacement_final_stats();

private:
  bool is_sampled(long set) const { return set % sample_stride == 0; }
  std::size_t pred_idx(uint64_t pc) const;
  bool friendly(uint64_t pc) const { return predictor[pred_idx(pc)] >= PRED_INIT; }
  void train(uint64_t pc, bool up);
  void insert(long set, long way, uint64_t pc, bool fill);
  int& rr(long set, long way) { return rrpv[static_cast<std::size_t>(set * NUM_WAY + way)]; }
};

#endif
