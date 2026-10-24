#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <cmath>

namespace strata::core::detail {

// Non-owning views: the complement and exchange allocations retain their original
// owners until FileExpertSource::close(). Ownership changes via slot IDs; the
// physical buffer addresses, bytes and CUDA aliases stay in place. This first path requires uniform, fully mapped/pinned expert slots.
class ExchangeStorage {
public:
    struct View {
        uint8_t* host = nullptr;
        const uint8_t* device = nullptr;
    };

    bool initialize(const std::vector<uint64_t>& offsets, uint8_t* resident_host,
                    const uint8_t* resident_device, uint64_t resident_bytes,
                    uint8_t* exchange_host, const uint8_t* exchange_device,
                    size_t count, size_t blob_bytes, std::string& error) {
        error.clear();
        if (active()) { error = "exchange storage already initialized"; return false; }
        if (!resident_host || !resident_device || !exchange_host || !exchange_device ||
            !blob_bytes || !count || !resident_bytes || resident_bytes % blob_bytes ||
            count > std::numeric_limits<size_t>::max() / blob_bytes) {
            error = "invalid uniform mapped exchange geometry"; return false;
        }
        const size_t resident_slots = (size_t)(resident_bytes / blob_bytes);
        if (resident_slots > std::numeric_limits<size_t>::max() - count) {
            error = "exchange slot count overflow"; return false;
        }
        auto experts = std::make_unique<std::atomic<size_t>[]>(offsets.size());
        std::vector<size_t> spares(count);
        std::vector<View> buffers(resident_slots + count);
        for (size_t slot = 0; slot < resident_slots; ++slot)
            buffers[slot] = {resident_host + slot * blob_bytes, resident_device + slot * blob_bytes};
        std::vector<bool> used(resident_slots, false);
        for (size_t i = 0; i < offsets.size(); ++i) {
            const uint64_t at = offsets[i];
            experts[i].store(kAbsent, std::memory_order_relaxed);
            if (at == ~uint64_t{0}) continue;
            if (at >= resident_bytes || at % blob_bytes || used[(size_t)(at / blob_bytes)]) {
                error = "invalid or duplicate resident slot"; return false;
            }
            used[(size_t)(at / blob_bytes)] = true;
            experts[i].store((size_t)(at / blob_bytes), std::memory_order_relaxed);
        }
        for (bool present : used) if (!present) {
            error = "unassigned resident slot"; return false;
        }
        for (size_t q = 0; q < count; ++q) {
            spares[q] = resident_slots + q;
            buffers[spares[q]] = {exchange_host + q * blob_bytes, exchange_device + q * blob_bytes};
        }
        experts_ = std::move(experts);
        expert_count_ = offsets.size();
        spares_.swap(spares);
        buffers_.swap(buffers);
        blob_bytes_ = blob_bytes;
        return true;
    }

    bool active() const { return blob_bytes_ != 0; }
    View resident(size_t expert) const {
        if (expert >= expert_count_) return {};
        const size_t slot = experts_[expert].load(std::memory_order_acquire);
        return slot != kAbsent ? buffers_[slot] : View{};
    }
    View spare(size_t q) const { return q < spares_.size() ? buffers_[spares_[q]] : View{}; }

    // Only after every CPU reader and H2D copy of `in` has completed, and `out`
    // has landed in spare(q). No allocation, memcpy, or CUDA operation here.
    bool commit(size_t in, size_t out, size_t q, const uint8_t* staged, size_t bytes) {
        if (in >= expert_count_ || out >= expert_count_ || q >= spares_.size() ||
            !staged || bytes != blob_bytes_) return false;
        const size_t incoming_slot = experts_[in].load(std::memory_order_acquire);
        if (incoming_slot == kAbsent || experts_[out].load(std::memory_order_acquire) != kAbsent ||
            buffers_[spares_[q]].host != staged) return false;
        // A background router lookahead can query residency to skip redundant
        // file prefetches. Publish one atomic slot ID; the pointer/alias pair is
        // immutable. Actual compute readers still obey the completion contract.
        experts_[out].store(spares_[q], std::memory_order_release);
        spares_[q] = incoming_slot;
        experts_[in].store(kAbsent, std::memory_order_release);
        ++exchanges_;
        avoided_bytes_ += bytes;
        return true;
    }

    uint64_t exchanges() const { return exchanges_; }
    uint64_t avoided_bytes() const { return avoided_bytes_; } // memcpy payload, not read+write total
    void clear() {
        experts_.reset(); expert_count_ = 0; buffers_.clear(); spares_.clear();
        blob_bytes_ = 0; exchanges_ = avoided_bytes_ = 0;
    }

private:
    static constexpr size_t kAbsent = std::numeric_limits<size_t>::max();
    std::unique_ptr<std::atomic<size_t>[]> experts_;
    size_t expert_count_ = 0;
    std::vector<View> buffers_; // immutable physical addresses after initialize
    std::vector<size_t> spares_; // touched only by the serialized cache manager
    size_t blob_bytes_ = 0;
    uint64_t exchanges_ = 0, avoided_bytes_ = 0;
};

} // namespace strata::core::detail
