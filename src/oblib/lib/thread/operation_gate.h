/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>

// Contract: callers retain the object until all invocations return.
// reset requires complete external quiescence, including rejected invocations.
namespace oceanbase {
namespace common {

template <size_t N>
class OperationGate {
  static_assert(N > 0, "nonempty gate");
  static constexpr uint64_t CLOSED = uint64_t{1} << 63;
  // 128-byte stride separates writers on both 64- and 128-byte cache lines.
  // No over-alignment requirement is imposed on the containing pool allocator.
  struct Slot {
    std::atomic<uint64_t> state{0};
    char padding[128 - sizeof(std::atomic<uint64_t>)];
  };
  struct StopFlag {
    std::atomic<bool> value{false};
    char padding[128 - sizeof(std::atomic<bool>)];
  } stopped_;
  Slot slots_[N];
public:
  bool stopped() const { return stopped_.value.load(std::memory_order_acquire); }
  bool enter(size_t id, size_t &token) {
    if (stopped()) return false;
    token = id % N;
    auto &s = slots_[token].state;
    auto v = s.load(std::memory_order_acquire);
    while (!(v & CLOSED)) {
      if (s.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel,
                                  std::memory_order_acquire)) {
        // Preserve the common stop flag as the rejection point across shards.
        if (!stopped()) return true;
        leave(token);
        return false;
      }
    }
    return false;
  }
  void leave(size_t token) {
    slots_[token].state.fetch_sub(1, std::memory_order_release);
  }
  void close() {
    stopped_.value.store(true, std::memory_order_release);
    // Must close ALL shards before wait starts inspecting counts.
    for (auto &s : slots_) s.state.fetch_or(CLOSED, std::memory_order_acq_rel);
  }
  bool drained() const {
    for (auto &s : slots_) {
      if (s.state.load(std::memory_order_acquire) != CLOSED) return false;
    }
    return true;
  }
  void reset() {
    for (auto &s : slots_) s.state.store(0, std::memory_order_relaxed);
    stopped_.value.store(false, std::memory_order_release);
  }
};

} // namespace common
} // namespace oceanbase
