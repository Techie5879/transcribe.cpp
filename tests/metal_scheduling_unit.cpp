#include "ggml-backend-impl.h"
#include "transcribe-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #condition); \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

struct State {
    int  count           = 1;
    int  changes         = 0;
    int  syncs           = 0;
    bool fail_eight_once = false;
};

const char * name(ggml_backend_reg_t reg) {
    return static_cast<const char *>(reg->context);
}

enum ggml_backend_dev_type type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

int get(ggml_backend_t backend) {
    return static_cast<State *>(backend->context)->count;
}

void set(ggml_backend_t backend, int count) {
    auto & state = *static_cast<State *>(backend->context);
    state.count  = count;
    ++state.changes;
    if (count == 8 && state.fail_eight_once) {
        state.fail_eight_once = false;
        throw std::runtime_error("simulated setting failure");
    }
}

void sync(ggml_backend_t backend) {
    ++static_cast<State *>(backend->context)->syncs;
}

void * proc(ggml_backend_reg_t, const char * value) {
    if (std::strcmp(value, "ggml_backend_metal_get_n_cb") == 0) {
        return reinterpret_cast<void *>(get);
    }
    if (std::strcmp(value, "ggml_backend_metal_set_n_cb") == 0) {
        return reinterpret_cast<void *>(set);
    }
    return nullptr;
}

void early_return(ggml_backend_t backend) {
    transcribe::MetalCommandBufferScope scope(backend, 8);
    CHECK(get(backend) == 8);
}
}  // namespace

int main() {
    State            state;
    ggml_backend_reg reg{};
    reg.iface.get_name         = name;
    reg.iface.get_proc_address = proc;
    reg.context                = const_cast<char *>("MTL");
    ggml_backend_device device{};
    device.iface.get_type = type;
    device.reg            = &reg;
    ggml_backend backend{};
    backend.device            = &device;
    backend.context           = &state;
    backend.iface.synchronize = sync;
    {
        transcribe::MetalCommandBufferScope normal(&backend, 0);
        CHECK(state.count == 1 && state.changes == 0);
    }
    early_return(&backend);
    CHECK(state.count == 1 && state.changes == 2 && state.syncs == 2);
    {
        transcribe::MetalCommandBufferScope outer(&backend, 8);
        {
            transcribe::MetalCommandBufferScope inner(&backend, 4);
            CHECK(state.count == 4);
        }
        CHECK(state.count == 8);
    }
    CHECK(state.count == 1);
    state.fail_eight_once = true;
    try {
        transcribe::MetalCommandBufferScope scope(&backend, 8);
        CHECK(false);
    } catch (const std::runtime_error &) {
    }
    CHECK(state.count == 1);
    try {
        transcribe::MetalCommandBufferScope scope(&backend, 8);
        throw std::runtime_error("simulated decode failure");
    } catch (const std::runtime_error &) {
    }
    CHECK(state.count == 1);
    const int changes = state.changes;
    reg.context       = const_cast<char *>("Vulkan");
    {
        transcribe::MetalCommandBufferScope non_metal(&backend, 8);
        CHECK(state.count == 1 && state.changes == changes);
    }
    return 0;
}
