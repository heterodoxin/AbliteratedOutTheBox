#pragma once
// Multi-GPU: make `device` current for a scope and restore the caller's device after it.  An object that owns CUDA
// streams, graphs and buffers on one device (a verify stage, the drafter) wraps its public calls in this, so the
// caller's thread may be on any device.  A negative device, or the device already current, does nothing.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace strata::core {

struct OnDevice {
    int previous = -1;
    explicit OnDevice(int device) try {
        int cur = 0;
        /*
        DPCT1093: The "device" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        if (device >= 0 &&
            DPCT_CHECK_ERROR(cur = dpct::get_current_device_id()) == 0 &&
            cur != device && DPCT_CHECK_ERROR(dpct::select_device(device)) == 0)
            previous = cur;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    /*
    DPCT1093: The "previous" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    ~OnDevice() { if (previous >= 0) dpct::select_device(previous); }
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;
};

}  // namespace strata::core
