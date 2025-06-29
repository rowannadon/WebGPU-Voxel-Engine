

#include "webgpu/webgpu.hpp"

using namespace wgpu;

/**
 * Utility function to get a WebGPU adapter, so that
 *     WGPUAdapter adapter = requestAdapter(options);
 * is roughly equivalent to
 *     const adapter = await navigator.gpu.requestAdapter(options);
 */
Adapter requestAdapterSync(Instance instance, RequestAdapterOptions const* options);

/**
 * Utility function to get a WebGPU device, so that
 *     WGPUAdapter device = requestDevice(adapter, options);
 * is roughly equivalent to
 *     const device = await adapter.requestDevice(descriptor);
 * It is very similar to requestAdapter
 */
Device requestDeviceSync(Adapter adapter, DeviceDescriptor const* descriptor);

/**
 * An example of how we can inspect the capabilities of the hardware through
 * the adapter object.
 */
void inspectAdapter(Adapter adapter);

/**
 * Display information about a device
 */
void inspectDevice(Device device);


/**
 * Round 'value' up to the next multiplier of 'step'.
 */
uint32_t ceilToNextMultiple(uint32_t value, uint32_t step);