// include/strata/copy_queue.hpp - SYCL port, 0.1.43: a queue on the GPU's dedicated copy (blitter) engine.
//
// WHY.  With the Level Zero "v2" adapter (the default on Xe2 / Battlemage and newer), every SYCL queue, including a queue
// that only holds host-to-device copies, runs on the compute engine: a 2 MB expert copy (~80 us at 25 GB/s) blocks the
// kernels behind it for that long.  A prompt chunk streams ~10,000 experts: ~0.8 s of the 3.8 s of a 4K prompt on an Arc
// Pro B70 is the compute engine waiting for its own copies (unitrace: every H2D copy sits on "L0 Compute Engine<0,0>"
// and none overlaps a kernel).  The device does expose a copy-only queue group (zeDeviceGetCommandQueueGroupProperties:
// group 1, flags COPY, 1 queue); the v2 adapter just does not use it for in-order queues.
//
// WHAT.  make_copy_engine_queue() creates an immediate command list on that group and wraps it with
// sycl::make_queue<ext_oneapi_level_zero>, so everything built on SYCL queues (memcpy, events, ext_oneapi_submit_barrier
// across queues) keeps working, only the engine changes.  It returns nullptr (the caller keeps an ordinary in-order queue)
// when the backend is not Level Zero, the device has no copy-only group, the loader or the call fails, or
// STRATA_COPY_ENGINE=0 (default: on for a Battlemage card, which is where it was measured; =1 forces it on any Level Zero
// device, =0 off).  Same bytes, same kernels: the tokens do not change.
#pragma once
#include <sycl/sycl.hpp>

namespace strata {
/// `what` names the first user in the one log line.  One queue per process (an in-order queue, the same one for every
/// caller: copies are serial on one engine anyway); never freed, so a caller must not delete it.
sycl::queue* make_copy_engine_queue(const char* what);
/// true once a copy-engine queue has been made in this process
bool copy_engine_active();
/// true for the queue make_copy_engine_queue() returned: it runs copies and barriers only, no kernel (not even queue::fill)
bool is_copy_engine_queue(const sycl::queue* q);
}  // namespace strata
