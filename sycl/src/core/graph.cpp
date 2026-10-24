// src/core/graph.cpp - P2.S5: the GraphRegistry implementation.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/graph.hpp"

#include <immintrin.h>

#include <chrono>
#include <cstdio>
#include <chrono>

namespace strata::core {
namespace {

/// Fills `err` from the CUDA runtime, naming the call that failed.  A bare "invalid argument" with no call
/// site is the least useful error this API can produce and the easiest to avoid.
bool fail(std::string &err, const char *what, dpct::err0 e) {
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    err = std::string(what) + ": " + dpct::get_error_string_dummy(e);
    return false;
}

}  // namespace

CapturedGraph& CapturedGraph::operator=(CapturedGraph&& o) noexcept {
    if (this != &o) {
        reset();
        graph_ = o.graph_;
        exec_ = o.exec_;
        done_ = o.done_;
        nodes_ = o.nodes_;
        o.graph_ = nullptr;
        o.exec_ = nullptr;
        o.done_ = nullptr;
        o.nodes_ = 0;
    }
    return *this;
}

void CapturedGraph::reset() {
    // ORDER MATTERS: destroying a graph exec BLOCKS until the launch it is destroying has completed, so
    // destroying it before the event is harmless but destroying it after a spin that was supposed to observe
    // completion would silently become the thing that caused it.  `bench/micro/graph_capture.cu` was fooled by
    // exactly this once - it looked like captured graphs made a doorbell visible and direct launches did not.
    if (exec_) { delete (exec_); exec_ = nullptr; }
    if (graph_) { delete (graph_); graph_ = nullptr; }
    if (done_) { dpct::destroy_event(done_); done_ = nullptr; }
    nodes_ = 0;
}

bool CapturedGraph::begin(void *stream, std::string &err) try {
    if (graph_ || exec_) { err = "begin: this CapturedGraph is already recorded"; return false; }
    const dpct::err0 e = DPCT_CHECK_ERROR(
        dpct::experimental::begin_recording(strata::q_of(stream)));
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (e != 0) return fail(err, "cudaStreamBeginCapture", e);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool CapturedGraph::end(void *stream, std::string &err) try {
    dpct::err0 e = DPCT_CHECK_ERROR(
        dpct::experimental::end_recording(strata::q_of(stream), &graph_));
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (e != 0) {
        graph_ = nullptr; return fail(err, "cudaStreamEndCapture", e);
    }

    nodes_ = 0;
    e = DPCT_CHECK_ERROR(
        dpct::experimental::get_nodes(graph_, nullptr, &nodes_));
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (e != 0) return fail(err, "cudaGraphGetNodes", e);
    // A capture that recorded NOTHING is a wiring mistake, and a graph that replays nothing produces no error
    // and no output - the silent kind of failure this project keeps paying for.
    if (nodes_ == 0) { err = "end: the capture recorded ZERO nodes - the body launched nothing"; reset(); return false; }

    e = DPCT_CHECK_ERROR(
        exec_ = new sycl::ext::oneapi::experimental::command_graph<
            sycl::ext::oneapi::experimental::graph_state::executable>(
            (graph_)->finalize()));
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (e != 0) return fail(err, "cudaGraphInstantiate", e);

    // The completion event is recorded ONCE and reused: `launch` records it again after each replay, which is
    // what makes `wait_ms` a query rather than a sync.
    e = DPCT_CHECK_ERROR(done_ = new sycl::event());
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (e != 0) return fail(err, "cudaEventCreateWithFlags", e);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool CapturedGraph::launch(void *stream, std::string &err) const try {
    if (!exec_) { err = "launch: not recorded"; return false; }
    dpct::err0 e =
        DPCT_CHECK_ERROR(strata::q_of(stream)->ext_oneapi_graph(*exec_));
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (e != 0) return fail(err, "cudaGraphLaunch", e);
    /*
    DPCT1012: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    /*
    DPCT1024: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    done__ct1 = std::chrono::steady_clock::now();
    e = DPCT_CHECK_ERROR(*done_ = strata::q_of(stream)
                                      ->ext_oneapi_submit_barrier());
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (e != 0) return fail(err, "cudaEventRecord", e);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool CapturedGraph::wait_ms(int timeout_ms) const try {
    if (!done_) return false;
    // A BOUNDED wait, and the bound is the point: an unbounded spin turns a protocol bug into a hung run, and
    // a hung run says nothing about which side is stuck.  `cudaEventQuery` is a QUERY - it does not block and
    // it does not synchronise - so this satisfies P2.X3 while still giving the driver the call it needs to
    // flush the submission.  See NOTE 1 in the header: without a driver call here the work never starts.
    //
    // The deadline is a REAL CLOCK, not a count of pause instructions: `_mm_pause` is a few cycles, so
    // counting pauses as microseconds would make the timeout tens of times longer than the caller asked for -
    // a timeout that does not time out is worse than none, because it reports a hang as a pass.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const dpct::err0 q = dpct::sycl_event_query(done_);
        if (q == 0) return true;
        if (q != 1) return false; // a real error, not "not finished"
        if (std::chrono::steady_clock::now() >= deadline) return false;
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool GraphRegistry::record(LayerType type, int n_tokens, const std::function<void()>& body, std::string& err) {
    const Key k{(int) type, n_tokens};
    if (graphs_.count(k)) return true;              // already recorded: the point of a registry

    CapturedGraph g;
    if (!g.begin(stream_, err)) return false;
    body();                                        // the caller launches into the captured stream
    // The body must not have failed silently.  An error state left on the stream would make EndCapture
    // succeed with a broken graph, so it is checked and cleared first.
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 body_err = 0;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (body_err != 0) {
        // Abandon the capture without instantiating anything.
        dpct::experimental::command_graph_ptr junk = nullptr;
        /*
        DPCT1001: The statement could not be removed.
        */
        dpct::experimental::end_recording(strata::q_of(stream_), &junk);
        if (junk) delete (junk);
        return fail(err, "the capture body left a CUDA error", body_err);
    }
    if (!g.end(stream_, err)) return false;

    graphs_.emplace(k, std::move(g));
    ++captures_;
    return true;
}

const CapturedGraph* GraphRegistry::find(LayerType type, int n_tokens) const {
    const auto it = graphs_.find(Key{(int) type, n_tokens});
    return it == graphs_.end() ? nullptr : &it->second;
}

bool GraphRegistry::launch(LayerType type, int n_tokens, int timeout_ms, std::string& err) const {
    const CapturedGraph* g = find(type, n_tokens);
    if (!g) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "launch: no graph recorded for %s n=%d", to_string(type), n_tokens);
        err = buf;
        return false;
    }
    if (!g->launch(stream_, err)) return false;
    if (!g->wait_ms(timeout_ms)) { err = "launch: timed out waiting for the graph to complete"; return false; }
    return true;
}

}  // namespace strata::core
