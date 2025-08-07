// Updated StorageBufferPool.h with bind group support and optimized slot allocation
#include <unordered_map>
#include <webgpu/webgpu.hpp>
#include "../FaceAttributes.h"
#include <mutex>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace wgpu;

struct SizeClass {
    int maxFaces;
    int maxIndices;
    int slotCount;
    size_t faceChunkSize;
    size_t indexChunkSize;
    int firstSlotIndex;  // Index of first slot in this size class
    int lastFreeHint;    // Hint for where to start searching for free slots

    SizeClass(int faces, int count)
        : maxFaces(faces), maxIndices(faces * 6), slotCount(count), firstSlotIndex(0), lastFreeHint(0) {
        const size_t FACE_STRIDE = sizeof(FaceAttributes);
        const size_t INDEX_STRIDE = sizeof(uint16_t);

        // Calculate aligned chunk sizes
        faceChunkSize = ((maxFaces * FACE_STRIDE + 3) / 4) * 4;
        indexChunkSize = ((maxIndices * INDEX_STRIDE + 3) / 4) * 4;
    }
};

struct SlotInfoPool {
    int sizeClassIndex;
    int slotWithinClass;
    size_t faceOffset;
    size_t indexOffset;
    bool occupied;

    SlotInfoPool() : sizeClassIndex(-1), slotWithinClass(-1), faceOffset(0), indexOffset(0), occupied(false) {}
    SlotInfoPool(int classIdx, int slotIdx, size_t faceOff, size_t indexOff)
        : sizeClassIndex(classIdx), slotWithinClass(slotIdx), faceOffset(faceOff), indexOffset(indexOff), occupied(false) {
    }
};

// Structure to match the shader's BufferMetadata uniform
struct BufferMetadata {
    uint32_t totalFaces;        // Total number of faces across all slots
    uint32_t slotCount;         // Total number of slots
    uint32_t padding0;          // Padding for alignment
    uint32_t padding1;          // Padding for alignment
};

// Structure to match the shader's SlotInfoPool storage buffer
struct ShaderSlotInfoPool {
    uint32_t faceOffset;        // Offset in faces (not bytes)
    uint32_t indexOffset;       // Offset in indices (not bytes)  
    uint32_t maxFaces;          // Maximum faces this slot can hold
    uint32_t maxIndices;        // Maximum indices this slot can hold
};

class StorageBufferPool {
    Device device;
    Queue queue;
    Buffer vertexBuffer;  // Stores face data
    Buffer indexBuffer;   // Stores indices for vertex pulling
    std::unordered_map<std::string, int> idToSlotMap;  // Maps ID to global slot index
    std::vector<SlotInfoPool> slots;  // Global slot information
    std::vector<SizeClass> sizeClasses;
    std::mutex dataMutex;

    size_t totalFaceBufferSize = 0;
    size_t totalIndexBufferSize = 0;

    // Bind group support
    BindGroupLayout bindGroupLayout;
    BindGroup bindGroup;
    Buffer bufferMetadataBuffer;
    Buffer SlotInfoPoolBuffer;
    BufferMetadata currentMetadata;
    std::vector<ShaderSlotInfoPool> shaderSlotInfoPoolData;

public:
    // Initialize with default size classes
    StorageBufferPool() {
        // Define default size classes: {maxFaces, slotCount}
        addSizeClass(1024, 8000);    // Small meshes
        addSizeClass(4096, 6000);    // Medium meshes  
        addSizeClass(16384, 3000);   // Large meshes
        addSizeClass(65536, 1000);   // Extra large meshes
    }

    // Add a size class (must be called before init())
    void addSizeClass(int maxFaces, int slotCount) {
        sizeClasses.emplace_back(maxFaces, slotCount);
    }

    // Clear all size classes and add custom ones
    void setSizeClasses(const std::vector<std::pair<int, int>>& classes) {
        sizeClasses.clear();
        for (const auto& [maxFaces, slotCount] : classes) {
            addSizeClass(maxFaces, slotCount);
        }
    }

    void init(Device d, Queue q) {
        device = d;
        queue = q;
        initBuffers();
        initBindGroupLayout();
        initBindGroup();
    }

    void initBuffers() {
        if (sizeClasses.empty()) {
            std::cerr << "No size classes defined!" << std::endl;
            return;
        }

        // Calculate total buffer sizes and create slot layout
        size_t currentFaceOffset = 0;
        size_t currentIndexOffset = 0;
        int globalSlotIndex = 0;

        for (int classIdx = 0; classIdx < sizeClasses.size(); ++classIdx) {
            auto& sizeClass = sizeClasses[classIdx];
            sizeClass.firstSlotIndex = globalSlotIndex;  // Record where this size class starts

            for (int slotIdx = 0; slotIdx < sizeClass.slotCount; ++slotIdx) {
                slots.emplace_back(classIdx, slotIdx, currentFaceOffset, currentIndexOffset);
                currentFaceOffset += sizeClass.faceChunkSize;
                currentIndexOffset += sizeClass.indexChunkSize;
                globalSlotIndex++;
            }
        }

        totalFaceBufferSize = currentFaceOffset;
        totalIndexBufferSize = currentIndexOffset;

        // Create face data buffer
        BufferDescriptor faceBufferDesc;
        faceBufferDesc.size = totalFaceBufferSize;
        faceBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
        faceBufferDesc.mappedAtCreation = false;
        faceBufferDesc.label = StringView("Storage Pool Face Data Buffer");
        vertexBuffer = device.createBuffer(faceBufferDesc);

        // Create index buffer
        BufferDescriptor indexBufferDesc;
        indexBufferDesc.size = totalIndexBufferSize;
        indexBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Index | BufferUsage::Storage;
        indexBufferDesc.mappedAtCreation = false;
        indexBufferDesc.label = StringView("Storage Pool Index Buffer");
        indexBuffer = device.createBuffer(indexBufferDesc);

        // Create metadata buffer
        BufferDescriptor metadataBufferDesc;
        metadataBufferDesc.size = sizeof(BufferMetadata);
        metadataBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
        metadataBufferDesc.mappedAtCreation = false;
        metadataBufferDesc.label = StringView("Buffer Metadata");
        bufferMetadataBuffer = device.createBuffer(metadataBufferDesc);

        // Create slot info buffer
        BufferDescriptor SlotInfoPoolBufferDesc;
        SlotInfoPoolBufferDesc.size = slots.size() * sizeof(ShaderSlotInfoPool);
        SlotInfoPoolBufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Storage;
        SlotInfoPoolBufferDesc.mappedAtCreation = false;
        SlotInfoPoolBufferDesc.label = StringView("Slot Info Buffer");
        SlotInfoPoolBuffer = device.createBuffer(SlotInfoPoolBufferDesc);

        // Initialize metadata
        updateMetadata();

        std::cout << "StorageBufferPool initialized with " << sizeClasses.size() << " size classes:" << std::endl;
        for (int i = 0; i < sizeClasses.size(); ++i) {
            const auto& sc = sizeClasses[i];
            std::cout << "  Class " << i << ": " << sc.maxFaces << " faces, " << sc.slotCount << " slots" << std::endl;
        }
        std::cout << "Total slots: " << slots.size() << std::endl;
        std::cout << "Face buffer: " << totalFaceBufferSize << " bytes" << std::endl;
        std::cout << "Index buffer: " << totalIndexBufferSize << " bytes" << std::endl;
    }

    void initBindGroupLayout() {
        std::vector<BindGroupLayoutEntry> entries(3, Default);

        // Vertex data storage buffer (binding 0)
        entries[0].binding = 0;
        entries[0].visibility = ShaderStage::Vertex;
        entries[0].buffer.type = BufferBindingType::ReadOnlyStorage;
        entries[0].buffer.minBindingSize = sizeof(FaceAttributes);

        // Buffer metadata uniform (binding 1)
        entries[1].binding = 1;
        entries[1].visibility = ShaderStage::Vertex;
        entries[1].buffer.type = BufferBindingType::Uniform;
        entries[1].buffer.minBindingSize = sizeof(BufferMetadata);

        // Slot info storage buffer (binding 2)
        entries[2].binding = 2;
        entries[2].visibility = ShaderStage::Vertex;
        entries[2].buffer.type = BufferBindingType::ReadOnlyStorage;
        entries[2].buffer.minBindingSize = sizeof(ShaderSlotInfoPool);

        BindGroupLayoutDescriptor layoutDesc;
        layoutDesc.entryCount = entries.size();
        layoutDesc.entries = entries.data();
        bindGroupLayout = device.createBindGroupLayout(layoutDesc);
    }

    void initBindGroup() {
        std::vector<BindGroupEntry> entries(3);

        // Vertex data buffer binding
        entries[0].binding = 0;
        entries[0].buffer = vertexBuffer;
        entries[0].offset = 0;
        entries[0].size = totalFaceBufferSize;

        // Buffer metadata binding
        entries[1].binding = 1;
        entries[1].buffer = bufferMetadataBuffer;
        entries[1].offset = 0;
        entries[1].size = sizeof(BufferMetadata);

        // Slot info binding
        entries[2].binding = 2;
        entries[2].buffer = SlotInfoPoolBuffer;
        entries[2].offset = 0;
        entries[2].size = slots.size() * sizeof(ShaderSlotInfoPool);

        BindGroupDescriptor bindGroupDesc;
        bindGroupDesc.layout = bindGroupLayout;
        bindGroupDesc.entryCount = entries.size();
        bindGroupDesc.entries = entries.data();
        bindGroup = device.createBindGroup(bindGroupDesc);
    }

    void updateMetadata() {
        // Update metadata
        currentMetadata.totalFaces = static_cast<uint32_t>(totalFaceBufferSize / sizeof(FaceAttributes));
        currentMetadata.slotCount = static_cast<uint32_t>(slots.size());
        currentMetadata.padding0 = 0;
        currentMetadata.padding1 = 0;

        // Update slot info data for shader
        shaderSlotInfoPoolData.clear();
        shaderSlotInfoPoolData.reserve(slots.size());

        for (const auto& slot : slots) {
            ShaderSlotInfoPool shaderInfo;
            shaderInfo.faceOffset = static_cast<uint32_t>(slot.faceOffset / sizeof(FaceAttributes));
            shaderInfo.indexOffset = static_cast<uint32_t>(slot.indexOffset / sizeof(uint16_t));

            if (slot.sizeClassIndex >= 0 && slot.sizeClassIndex < sizeClasses.size()) {
                shaderInfo.maxFaces = static_cast<uint32_t>(sizeClasses[slot.sizeClassIndex].maxFaces);
                shaderInfo.maxIndices = static_cast<uint32_t>(sizeClasses[slot.sizeClassIndex].maxIndices);
            }
            else {
                shaderInfo.maxFaces = 0;
                shaderInfo.maxIndices = 0;
            }

            shaderSlotInfoPoolData.push_back(shaderInfo);
        }

        // Upload to GPU
        queue.writeBuffer(bufferMetadataBuffer, 0, &currentMetadata, sizeof(BufferMetadata));
        if (!shaderSlotInfoPoolData.empty()) {
            queue.writeBuffer(SlotInfoPoolBuffer, 0, shaderSlotInfoPoolData.data(),
                shaderSlotInfoPoolData.size() * sizeof(ShaderSlotInfoPool));
        }
    }

    int findBestSizeClass(size_t numFaces) {
        int bestClass = -1;
        int bestSize = INT_MAX;

        for (int i = 0; i < sizeClasses.size(); ++i) {
            if (sizeClasses[i].maxFaces >= numFaces && sizeClasses[i].maxFaces < bestSize) {
                bestClass = i;
                bestSize = sizeClasses[i].maxFaces;
            }
        }

        return bestClass;
    }

    int findFreeSlotInClass(int sizeClassIndex) {
        if (sizeClassIndex < 0 || sizeClassIndex >= sizeClasses.size()) {
            return -1;
        }

        auto& sizeClass = sizeClasses[sizeClassIndex];
        const int firstSlot = sizeClass.firstSlotIndex;
        const int lastSlot = firstSlot + sizeClass.slotCount;
        const int startPos = firstSlot + sizeClass.lastFreeHint;

        // First, search from the hint position to the end of this size class
        for (int i = startPos; i < lastSlot; ++i) {
            if (!slots[i].occupied) {
                // Update hint to next position (with wraparound)
                sizeClass.lastFreeHint = (i - firstSlot + 1) % sizeClass.slotCount;
                return i;
            }
        }

        // If not found, search from the beginning up to the hint position (wraparound search)
        for (int i = firstSlot; i < startPos; ++i) {
            if (!slots[i].occupied) {
                // Update hint to next position (with wraparound)
                sizeClass.lastFreeHint = (i - firstSlot + 1) % sizeClass.slotCount;
                return i;
            }
        }

        // No free slots found
        return -1;
    }

    int allocateSlot(const std::string& id, size_t numFaces) {
        std::lock_guard<std::mutex> lock(dataMutex);

        // Check if already allocated
        auto it = idToSlotMap.find(id);
        if (it != idToSlotMap.end()) {
            return it->second;
        }

        // Find best size class
        int sizeClassIndex = findBestSizeClass(numFaces);
        if (sizeClassIndex == -1) {
            std::cerr << "No suitable size class for " << numFaces << " faces" << std::endl;
            return -1;
        }

        // Find free slot in that class
        int slotIndex = findFreeSlotInClass(sizeClassIndex);
        if (slotIndex == -1) {
            std::cerr << "No free slots in size class " << sizeClassIndex
                << " (max faces: " << sizeClasses[sizeClassIndex].maxFaces << ")" << std::endl;
            return -1;
        }

        // Allocate the slot
        slots[slotIndex].occupied = true;
        idToSlotMap[id] = slotIndex;

        //std::cout << "Allocated slot " << slotIndex << " in size class " << sizeClassIndex
        //    << " for " << numFaces << " faces (ID: " << id << ")" << std::endl;

        return slotIndex;
    }

    void writeToSlot(const std::string& id, std::vector<FaceAttributes>& faceData, std::vector<uint16_t>& indexData) {
        std::lock_guard<std::mutex> lock(dataMutex);

        auto it = idToSlotMap.find(id);
        if (it == idToSlotMap.end()) {
            std::cerr << "Slot not found for ID: " << id << std::endl;
            return;
        }

        int slotIndex = it->second;
        const SlotInfoPool& slot = slots[slotIndex];
        const SizeClass& sizeClass = sizeClasses[slot.sizeClassIndex];

        // Validate data size
        if (faceData.size() > sizeClass.maxFaces) {
            std::cerr << "Face data too large for slot: " << faceData.size()
                << " > " << sizeClass.maxFaces << std::endl;
            return;
        }

        if (indexData.size() > sizeClass.maxIndices) {
            std::cerr << "Index data too large for slot: " << indexData.size()
                << " > " << sizeClass.maxIndices << std::endl;
            return;
        }

        // Calculate data sizes
        const size_t FACE_STRIDE = sizeof(FaceAttributes);
        const size_t INDEX_STRIDE = sizeof(uint16_t);

        size_t faceDataSize = faceData.size() * FACE_STRIDE;
        size_t indexDataSize = indexData.size() * INDEX_STRIDE;

        // Write face data
        if (!faceData.empty()) {
            queue.writeBuffer(vertexBuffer, slot.faceOffset, faceData.data(), faceDataSize);
        }

        // Write index data
        if (!indexData.empty()) {
            queue.writeBuffer(indexBuffer, slot.indexOffset, indexData.data(), indexDataSize);
        }

        //std::cout << "Written " << faceData.size() << " faces and " << indexData.size()
        //    << " indices to slot " << slotIndex << " (ID: " << id << ")" << std::endl;
    }

    void deAllocateSlot(const std::string& id) {
        std::lock_guard<std::mutex> lock(dataMutex);

        auto it = idToSlotMap.find(id);
        if (it != idToSlotMap.end()) {
            int slotIndex = it->second;

            // Update the hint for this size class to point to the newly freed slot
            const SlotInfoPool& slot = slots[slotIndex];
            if (slot.sizeClassIndex >= 0 && slot.sizeClassIndex < sizeClasses.size()) {
                auto& sizeClass = sizeClasses[slot.sizeClassIndex];
                sizeClass.lastFreeHint = slotIndex - sizeClass.firstSlotIndex;
            }

            slots[slotIndex].occupied = false;
            idToSlotMap.erase(it);

            std::cout << "Deallocated slot " << slotIndex << " (ID: " << id << ")" << std::endl;
        }
    }

    // Getters
    Buffer getIndexBuffer() { return indexBuffer; }
    Buffer getVertexBuffer() { return vertexBuffer; }
    uint32_t getVertexBufferSize() { return static_cast<uint32_t>(totalFaceBufferSize); }
    uint32_t getIndexBufferSize() { return static_cast<uint32_t>(totalIndexBufferSize); }

    BindGroupLayout getBindGroupLayout() { return bindGroupLayout; }
    BindGroup getBindGroup() { return bindGroup; }

    // Get slot information
    size_t getFaceOffset(int slotIndex) {
        return slotIndex < slots.size() ? slots[slotIndex].faceOffset : 0;
    }

    size_t getIndexOffset(int slotIndex) {
        return slotIndex < slots.size() ? slots[slotIndex].indexOffset : 0;
    }

    uint32_t getFaceOffsetInElements(int slotIndex) {
        return static_cast<uint32_t>(getFaceOffset(slotIndex) / sizeof(FaceAttributes));
    }

    uint32_t getIndexOffsetInElements(int slotIndex) {
        return static_cast<uint32_t>(getIndexOffset(slotIndex) / sizeof(uint16_t));
    }

    // Get slot capacity information
    int getSlotMaxFaces(int slotIndex) {
        if (slotIndex < slots.size()) {
            return sizeClasses[slots[slotIndex].sizeClassIndex].maxFaces;
        }
        return 0;
    }

    int getSlotMaxIndices(int slotIndex) {
        if (slotIndex < slots.size()) {
            return sizeClasses[slots[slotIndex].sizeClassIndex].maxIndices;
        }
        return 0;
    }

    // Utility function to get memory usage statistics
    void printStats() {
        std::lock_guard<std::mutex> lock(dataMutex);

        std::vector<int> occupiedPerClass(sizeClasses.size(), 0);
        std::vector<int> totalPerClass(sizeClasses.size(), 0);

        for (const auto& slot : slots) {
            totalPerClass[slot.sizeClassIndex]++;
            if (slot.occupied) {
                occupiedPerClass[slot.sizeClassIndex]++;
            }
        }

        std::cout << "\nBuffer Pool Statistics:" << std::endl;
        for (int i = 0; i < sizeClasses.size(); ++i) {
            const auto& sc = sizeClasses[i];
            std::cout << "Size Class " << i << " (" << sc.maxFaces << " faces): "
                << occupiedPerClass[i] << "/" << totalPerClass[i] << " occupied - " << ((float)occupiedPerClass[i] / (float)totalPerClass[i]) * 100.0f << " %" << "\n";
        }
        std::cout << "Total allocated IDs: " << idToSlotMap.size() << std::endl;
        std::cout << std::endl;
    }
};