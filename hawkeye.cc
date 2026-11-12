#include "hawkeye.h"

#include <algorithm>
#include <iostream>

hawkeye::hawkeye(CACHE* cache)
    : replacement(cache), NUM_SET(cache->NUM_SET), NUM_WAY(cache->NUM_WAY), offset_bits(cache->OFFSET_BITS), sample_stride(std::max<long>(1, NUM_SET / SAMPLED_SETS)),
      window(static_cast<std::size_t>(8 * NUM_WAY)), rrpv(static_cast<std::size_t>(NUM_SET * NUM_WAY), MAX_RRPV),
      line_pc(static_cast<std::size_t>(NUM_SET * NUM_WAY), 0), predictor(PRED_SIZE, PRED_INIT)
{
  long n = (NUM_SET + sample_stride - 1) / sample_stride;
  opt.assign(static_cast<std::size_t>(n), optgen{window, static_cast<uint32_t>(NUM_WAY)});
  set_timer.assign(static_cast<std::size_t>(n), 0);
  sampler.resize(static_cast<std::size_t>(n));
}

bool hawkeye::optgen::should_cache(uint64_t curr, uint64_t last)
{
  // Belady would have cached this line iff the cache was never full over its liveness interval.
  for (uint64_t q = last; q != curr; ++q)
    if (occupancy[q % occupancy.size()] >= capacity)
      return false;
  for (uint64_t q = last; q != curr; ++q)
    ++occupancy[q % occupancy.size()];
  return true;
}

std::size_t hawkeye::pred_idx(uint64_t pc) const
{
  pc ^= pc >> 17;
  pc *= 0x9E3779B97F4A7C15ULL;
  return static_cast<std::size_t>(pc >> 32) % PRED_SIZE;
}

void hawkeye::train(uint64_t pc, bool up)
{
  int& c = predictor[pred_idx(pc)];
  c = up ? std::min(c + 1, PRED_MAX) : std::max(c - 1, 0);
}

long hawkeye::find_victim(uint32_t, uint64_t, long set, const champsim::cache_block*, champsim::address, champsim::address, access_type)
{
  // Prefer a cache-averse line (RRPV 7); otherwise evict the oldest friendly line and detrain its PC.
  for (long w = 0; w < NUM_WAY; ++w)
    if (rr(set, w) == MAX_RRPV)
      return w;

  long victim = 0;
  for (long w = 1; w < NUM_WAY; ++w)
    if (rr(set, w) > rr(set, victim))
      victim = w;

  if (is_sampled(set))
    train(line_pc[static_cast<std::size_t>(set * NUM_WAY + victim)], false);
  return victim;
}

// Called on every access (hit or miss lookup): OPTgen training on sampled sets, promotion on hits.
void hawkeye::update_replacement_state(uint32_t, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address,
                                       access_type type, uint8_t hit)
{
  if (access_type{type} == access_type::WRITE)
    return;  // writebacks carry no PC signal

  const uint64_t pc = ip.to<uint64_t>();

  if (is_sampled(set)) {
    const uint64_t line = full_addr.slice_upper(offset_bits).to<uint64_t>();
    auto s = static_cast<std::size_t>(set / sample_stride);
    auto& hist = sampler[s];
    const uint64_t curr = set_timer[s]++;

    if (auto it = hist.find(line); it != hist.end()) {
      const bool in_window = (curr - it->second.last_quanta) < window;
      train(it->second.pc, in_window && opt[s].should_cache(curr, it->second.last_quanta));
      it->second.pc = pc;
      it->second.last_quanta = curr;
      it->second.lru = curr;
    } else {
      if (hist.size() >= window) {  // keep ~8x associativity worth of history, LRU
        auto oldest = std::min_element(hist.begin(), hist.end(), [](auto& a, auto& b) { return a.second.lru < b.second.lru; });
        hist.erase(oldest);
      }
      hist[line] = history_entry{pc, curr, curr};
    }
    opt[s].add_access(curr);
  }

  if (hit && way < NUM_WAY)
    insert(set, way, pc, false);
}

// Called when a line is filled after a miss.
void hawkeye::replacement_cache_fill(uint32_t, long set, long way, champsim::address, champsim::address ip, champsim::address, access_type type)
{
  if (way >= NUM_WAY)
    return;  // bypassed
  if (access_type{type} == access_type::WRITE) {
    rr(set, way) = MAX_RRPV;
    return;
  }
  insert(set, way, ip.to<uint64_t>(), true);
}

void hawkeye::insert(long set, long way, uint64_t pc, bool fill)
{
  const bool fr = friendly(pc);
  (fr ? stat_friendly : stat_averse)++;
  line_pc[static_cast<std::size_t>(set * NUM_WAY + way)] = pc;

  if (!fr) {
    rr(set, way) = MAX_RRPV;
    return;
  }
  if (fill)  // age other friendly lines on a friendly fill
    for (long w = 0; w < NUM_WAY; ++w)
      if (w != way && rr(set, w) < MAX_RRPV - 1)
        ++rr(set, w);
  rr(set, way) = 0;
}

void hawkeye::replacement_final_stats()
{
  auto total = stat_friendly + stat_averse;
  std::cout << "Hawkeye predictions: friendly " << stat_friendly << " averse " << stat_averse;
  if (total)
    std::cout << " (" << 100.0 * static_cast<double>(stat_averse) / static_cast<double>(total) << "% averse)";
  std::cout << std::endl;
}
