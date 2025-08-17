// AsyncDepthReadback.h - Proper asynchronous depth buffer readback

#pragma once
#include <webgpu/webgpu.hpp>
#include <vector>
#include <future>
#include <atomic>
#include <memory>

using namespace wgpu;

class AsyncDepthReadback {
private:
    struct ReadbackBuffer {
        Buffer buffer;
        size_t size;
        uint32_t width;
        uint32_t height;
        bool isMapped = false;
        std::vector<float> data;
    };

    WebGPUContext* context;
    std::vector<std::unique_ptr<ReadbackBuffer>> buffers;
    int currentBufferIndex = 0;
    static constexpr int NUM_BUFFERS = 3; // Triple buffering for readback

    std::atomic<bool> readbackInProgress{ false };
    std::atomic<bool> dataReady{ false };
    std::vector<float> latestDepthData;
    uint32_t latestWidth = 0;
    uint32_t latestHeight = 0;

public:
    AsyncDepthReadback(WebGPUContext* ctx) : context(ctx) {}

    bool initialize(uint32_t width, uint32_t height) {
        size_t bufferSize = width * height * sizeof(float);

        for (int i = 0; i < NUM_BUFFERS; i++) {
            auto readbackBuffer = std::make_unique<ReadbackBuffer>();
            readbackBuffer->width = width;
            readbackBuffer->height = height;
            readbackBuffer->size = bufferSize;
            readbackBuffer->data.resize(width * height);

            BufferDescriptor bufferDesc;
            bufferDesc.size = bufferSize;
            bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::MapRead;
            bufferDesc.mappedAtCreation = false;
            bufferDesc.label = StringView(("Depth readback buffer " + std::to_string(i)).c_str());

            readbackBuffer->buffer = context->getDevice().createBuffer(bufferDesc);
            buffers.push_back(std::move(readbackBuffer));
        }

        latestDepthData.resize(width * height, 1.0f);
        latestWidth = width;
        latestHeight = height;

        return true;
    }

    void startReadback(CommandEncoder encoder, Texture depthTexture, uint32_t width, uint32_t height) {
        if (readbackInProgress) return; // Skip if previous readback still in progress

        auto& buffer = buffers[currentBufferIndex];

        // Copy texture to buffer
        ImageCopyTexture src;
        src.texture = depthTexture;
        src.mipLevel = 0;
        src.origin = { 0, 0, 0 };
        src.aspect = TextureAspect::All;

        ImageCopyBuffer dst;
        dst.buffer = buffer->buffer;
        dst.layout.offset = 0;
        dst.layout.bytesPerRow = width * sizeof(float);
        dst.layout.rowsPerImage = height;

        Extent3D copySize;
        copySize.width = width;
        copySize.height = height;
        copySize.depthOrArrayLayers = 1;

        encoder.copyTextureToBuffer(src, dst, copySize);
        readbackInProgress = true;
    }

    void processReadback() {
        if (!readbackInProgress) return;

        auto& buffer = buffers[currentBufferIndex];

        if (!buffer->isMapped) {
            // Start async mapping
            buffer->buffer.mapAsync(
                MapMode::Read,
                0,
                buffer->size,
                [this, buffer = buffer.get()](MapAsyncStatus status) {
                    if (status == MapAsyncStatus::Success) {
                        // Copy data from mapped buffer
                        const float* mappedData = static_cast<const float*>(
                            buffer->buffer.getConstMappedRange(0, buffer->size)
                            );

                        std::copy(mappedData, mappedData + (buffer->size / sizeof(float)), buffer->data.begin());

                        buffer->buffer.unmap();
                        buffer->isMapped = false;

                        // Update latest data atomically
                        {
                            latestDepthData = buffer->data;
                            latestWidth = buffer->width;
                            latestHeight = buffer->height;
                            dataReady = true;
                        }

                        readbackInProgress = false;
                    }
                    else {
                        readbackInProgress = false;
                    }
                }
            );
            buffer->isMapped = true;
        }

        // Move to next buffer for next frame
        currentBufferIndex = (currentBufferIndex + 1) % NUM_BUFFERS;
    }

    bool isDataReady() const {
        return dataReady.load();
    }

    std::vector<float> getLatestDepthData() const {
        return latestDepthData; // This creates a copy - consider using reference if performance critical
    }

    uint32_t getLatestWidth() const { return latestWidth; }
    uint32_t getLatestHeight() const { return latestHeight; }

    float sampleDepth(float normalizedX, float normalizedY) const {
        if (!dataReady || latestDepthData.empty()) return 1.0f;

        int x = static_cast<int>(normalizedX * latestWidth);
        int y = static_cast<int>(normalizedY * latestHeight);

        x = std::max(0, std::min(static_cast<int>(latestWidth - 1), x));
        y = std::max(0, std::min(static_cast<int>(latestHeight - 1), y));

        return latestDepthData[y * latestWidth + x];
    }

    void terminate() {
        for (auto& buffer : buffers) {
            if (buffer->isMapped) {
                buffer->buffer.unmap();
            }
            buffer->buffer.release();
        }
        buffers.clear();
    }
};

// Enhanced WebGPURenderer integration
class WebGPURenderer {
private:
    // ... existing members ...

    // Enhanced depth pre-pass system
    std::unique_ptr<DepthPrePassPipeline> depthPrePassPipeline;
    std::unique_ptr<AsyncDepthReadback> asyncDepthReadback;

public:
    bool initialize() {
        // ... existing initialization ...

        // Initialize depth pre-pass pipeline
        depthPrePassPipeline = std::make_unique<DepthPrePassPipeline>();
        depthPrePassPipeline->init(bufferManager.get(), textureManager.get(), pipelineManager.get(),
            modelManager.get(), context.get());

        // Initialize async depth readback
        asyncDepthReadback = std::make_unique<AsyncDepthReadback>(context.get());

        int width, height;
        glfwGetFramebufferSize(context->getWindow(), &width, &height);
        uint32_t prePassWidth = std::max(1u, static_cast<uint32_t>(width / 4));
        uint32_t prePassHeight = std::max(1u, static_cast<uint32_t>(height / 4));

        if (!asyncDepthReadback->initialize(prePassWidth, prePassHeight)) {
            return false;
        }

        // Create depth pre-pass resources and pipeline
        depthPrePassPipeline->createResources();
        depthPrePassPipeline->createPipeline();
        depthPrePassPipeline->createBindGroup();

        // ... rest of initialization ...
        return true;
    }

    std::vector<DAIC> cullOccludedDAICs(const std::vector<DAIC>& daics, const MyUniforms& uniforms) {
        if (!asyncDepthReadback->isDataReady()) {
            return daics; // No culling if no depth data available
        }

        std::vector<DAIC> culledDAICs;
        culledDAICs.reserve(daics.size());

        for (const auto& daic : daics) {
            if (!isOccluded(daic, uniforms)) {
                culledDAICs.push_back(daic);
            }
        }

        // Add some statistics
        static int frameCount = 0;
        if (++frameCount % 60 == 0) { // Log every 60 frames
            float cullRatio = 1.0f - (static_cast<float>(culledDAICs.size()) / daics.size());
            printf("High-Z Culling: %.1f%% of DAICs culled (%zu -> %zu)\n",
                cullRatio * 100.0f, daics.size(), culledDAICs.size());
        }

        return culledDAICs;
    }

    bool isOccluded(const DAIC& daic, const MyUniforms& uniforms) {
        // Transform DAIC position to normalized device coordinates
        glm::vec4 worldPos(daic.x, daic.y, daic.z, 1.0f);
        glm::mat4 viewProj = uniforms.projectionMatrix * uniforms.viewMatrix;
        glm::vec4 clipPos = viewProj * worldPos;

        if (clipPos.w <= 0.0f) return true; // Behind camera

        glm::vec3 ndcPos = glm::vec3(clipPos) / clipPos.w;

        // Check frustum bounds
        if (ndcPos.x < -1.0f || ndcPos.x > 1.0f ||
            ndcPos.y < -1.0f || ndcPos.y > 1.0f ||
            ndcPos.z < 0.0f || ndcPos.z > 1.0f) {
            return true; // Outside frustum
        }

        // Convert to texture coordinates
        float texX = (ndcPos.x + 1.0f) * 0.5f;
        float texY = (1.0f - ndcPos.y) * 0.5f; // Flip Y

        // Sample depth buffer
        float existingDepth = asyncDepthReadback->sampleDepth(texX, texY);

        // Occlusion test with bias
        float bias = 0.001f;
        return ndcPos.z > (existingDepth + bias);
    }

    void renderFrame(MyUniforms& uniforms, ColumnDAICs chunkRenderData) {
        auto [surfaceTexture, targetView] = GetNextSurfaceViewData();
        if (!targetView) return;

        CommandEncoderDescriptor encoderDesc = Default;
        encoderDesc.label = StringView("Frame command encoder");
        CommandEncoder encoder = context->getDevice().createCommandEncoder(encoderDesc);

        // Update uniforms
        {
            MyUniforms uOpaque = uniforms;
            uOpaque.transparent = 0u;
            bufferManager->writeBuffer("uniform_buffer_opaque", 0, &uOpaque, sizeof(MyUniforms));

            MyUniforms uTransp = uniforms;
            uTransp.transparent = 1u;
            bufferManager->writeBuffer("uniform_buffer_transparent", 0, &uTransp, sizeof(MyUniforms));
        }

        // Process any pending async depth readback from previous frame
        asyncDepthReadback->processReadback();

        // Perform high-Z culling using previous frame's depth data
        auto culledOpaqueDAICs = cullOccludedDAICs(chunkRenderData.opaqueDAICs, uniforms);
        auto culledTransparentDAICs = cullOccludedDAICs(chunkRenderData.transparentDAICs, uniforms);

        // Upload culled DAIC data for current frame
        if (!culledOpaqueDAICs.empty()) {
            context->getQueue().writeBuffer(
                opaqueIndirectBuffer,
                0,
                culledOpaqueDAICs.data(),
                culledOpaqueDAICs.size() * sizeof(DAIC)
            );
        }

        if (!culledTransparentDAICs.empty()) {
            context->getQueue().writeBuffer(
                transparentIndirectBuffer,
                0,
                culledTransparentDAICs.data(),
                culledTransparentDAICs.size() * sizeof(DAIC)
            );
        }

        // DEPTH PRE-PASS: Render low-resolution depth for next frame's culling
        if (!culledOpaqueDAICs.empty()) {
            depthPrePassPipeline->render(culledOpaqueDAICs.size(), opaqueIndirectBuffer, encoder);

            // Start async readback for next frame
            Texture depthTexture = textureManager->getTexture("depth_readback_texture");
            asyncDepthReadback->startReadback(encoder, depthTexture,
                depthPrePassPipeline->getWidth(),
                depthPrePassPipeline->getHeight());
        }

        // Upload shadow DAIC data
        if (!chunkRenderData.transparentShadowDAICs.empty()) {
            context->getQueue().writeBuffer(
                transparentShadowIndirectBuffer, 0,
                chunkRenderData.transparentShadowDAICs.data(),
                chunkRenderData.transparentShadowDAICs.size() * sizeof(DAIC)
            );
        }

        if (!chunkRenderData.opaqueShadowDAICs.empty()) {
            context->getQueue().writeBuffer(
                opaqueShadowIndirectBuffer, 0,
                chunkRenderData.opaqueShadowDAICs.data(),
                chunkRenderData.opaqueShadowDAICs.size() * sizeof(DAIC)
            );
        }

        // Continue with regular rendering pipeline...
        noisePipeline.render(encoder);
        transmittancePipeline.render(encoder);
        multiScatteringPipeline.render(encoder);
        skyViewPipeline.render(encoder);
        aerialPerspectivePipeline.render(encoder);

        // Shadow passes
        if (!chunkRenderData.opaqueShadowDAICs.empty()) {
            shadowPipeline.render(chunkRenderData.opaqueShadowDAICs.size(),
                opaqueShadowIndirectBuffer, encoder,
                "shadow_uniforms_group_opaque", LoadOp::Clear);
        }

        // Main rendering with culled geometry
        if (!culledOpaqueDAICs.empty()) {
            voxelPipeline.render(culledOpaqueDAICs.size(), opaqueIndirectBuffer, targetView, encoder);
        }

        if (!culledTransparentDAICs.empty()) {
            transparentVoxelPipeline.render(culledTransparentDAICs.size(),
                transparentIndirectBuffer, targetView, encoder);
        }

        skyPipeline.render(targetView, encoder);

        // GUI rendering
        {
            RenderPassDescriptor renderPassDesc = {};
            RenderPassColorAttachment colorAttachment = {};
            colorAttachment.view = targetView;
            colorAttachment.loadOp = LoadOp::Load;
            colorAttachment.storeOp = StoreOp::Store;
#ifndef WEBGPU_BACKEND_WGPU
            colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif
            renderPassDesc.colorAttachmentCount = 1;
            renderPassDesc.colorAttachments = &colorAttachment;

            RenderPassEncoder imguiRenderPass = encoder.beginRenderPass(renderPassDesc);
            ImGui::Render();
            ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), imguiRenderPass);
            imguiRenderPass.end();
            imguiRenderPass.release();
        }

        CommandBufferDescriptor cmdBufferDescriptor = {};
        CommandBuffer command = encoder.finish(cmdBufferDescriptor);
        encoder.release();

        context->getQueue().submit(1, &command);
        command.release();

#ifdef WEBGPU_BACKEND_DAWN
        context->getDevice().tick();
#endif

        targetView.release();
        context->getSurface().present();

#ifdef WEBGPU_BACKEND_DAWN
        context->getDevice().tick();
#endif
    }

    void terminate() {
        if (asyncDepthReadback) {
            asyncDepthReadback->terminate();
        }

        // ... existing cleanup ...
        opaqueIndirectBuffer.release();
        transparentIndirectBuffer.release();
        transparentShadowIndirectBuffer.release();
        opaqueShadowIndirectBuffer.release();

        textureManager->terminate();
        pipelineManager->terminate();
        bufferManager->terminate();
    }
};