// ImprovedChunkWorkerSystem.h
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <memory>
#include <chrono>
#include <iostream>
#include "glm/glm.hpp"
#include "ChunkColumn.h"

using glm::ivec3;
using glm::ivec2;

// Enhanced work item with better priority handling
class ChunkWorkItem {
public:
    enum Type {
        GenerateTerrain,
        GenerateTopsoil,
        GenerateMesh,
        GenerateTrees,
        RegenerateMesh,
        COUNT // For validation
    };

    enum Priority {
        LOW = 0,
        NORMAL = 50,
        HIGH = 100,
        CRITICAL = 200
    };

private:
    Type type;
    std::shared_ptr<ChunkColumn> chunk;
    ivec2 position;
    std::array<std::shared_ptr<ChunkColumn>, 4> neighbors;
    int priority;
    int id;
    std::chrono::steady_clock::time_point creation_time;
    static inline std::atomic<int> ChunkWorkItem::next_id{ 0 };

public:
    // Default constructor
    ChunkWorkItem()
        : type(GenerateTerrain), chunk(nullptr), position(0, 0), neighbors{},
        priority(NORMAL), id(next_id++), creation_time(std::chrono::steady_clock::now()) {
    }

    ChunkWorkItem(Type t, std::shared_ptr<ChunkColumn> c, ivec2 pos, int prio = NORMAL)
        : type(t), chunk(c), position(pos), neighbors{}, priority(prio),
        id(next_id++), creation_time(std::chrono::steady_clock::now()) {
    }

    ChunkWorkItem(Type t, std::shared_ptr<ChunkColumn> c, ivec2 pos,
        std::array<std::shared_ptr<ChunkColumn>, 4> neighs, int prio = NORMAL)
        : type(t), chunk(c), position(pos), neighbors(neighs), priority(prio),
        id(next_id++), creation_time(std::chrono::steady_clock::now()) {
    }

    // Priority comparison for queue (higher priority first)
    bool operator<(const ChunkWorkItem& other) const {
        if (priority != other.priority) {
            return priority < other.priority; // Higher priority first
        }
        return id > other.id; // FIFO for same priority
    }

    // Getters
    Type getType() const { return type; }
    std::shared_ptr<ChunkColumn> getChunk() const { return chunk; }
    ivec2 getPosition() const { return position; }
    const std::array<std::shared_ptr<ChunkColumn>, 4>& getNeighbors() const { return neighbors; }
    int getPriority() const { return priority; }
    int getId() const { return id; }

    // Check if work item is still valid
    bool isValid() const {
        return chunk && chunk->getState() != ColumnState::Unloading;
    }

    // Get age of work item
    std::chrono::milliseconds getAge() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - creation_time);
    }
};

// Thread-safe priority queue for work items
class ChunkWorkQueue {
private:
    std::priority_queue<ChunkWorkItem> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> shutdown{ false };
    std::atomic<size_t> total_queued{ 0 };
    std::atomic<size_t> total_processed{ 0 };

public:
    bool push(const ChunkWorkItem& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (shutdown) return false;

        queue.push(item);
        total_queued++;
        cv.notify_one();
        return true;
    }

    bool pop(ChunkWorkItem& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mtx);

        if (cv.wait_for(lock, timeout, [this] { return !queue.empty() || shutdown; })) {
            if (shutdown && queue.empty()) {
                return false;
            }

            if (!queue.empty()) {
                item = queue.top();
                queue.pop();
                total_processed++;
                return true;
            }
        }
        return false;
    }

    void requestShutdown() {
        shutdown = true;
        cv.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }

    size_t getTotalQueued() const { return total_queued; }
    size_t getTotalProcessed() const { return total_processed; }

    // Remove invalid work items (cleanup)
    size_t removeInvalidItems() {
        std::lock_guard<std::mutex> lock(mtx);
        std::priority_queue<ChunkWorkItem> validItems;
        size_t removed = 0;

        while (!queue.empty()) {
            ChunkWorkItem item = queue.top();
            queue.pop();

            if (item.isValid()) {
                validItems.push(item);
            }
            else {
                removed++;
            }
        }

        queue = std::move(validItems);
        return removed;
    }
};

// Enhanced chunk worker system with thread pool architecture
class ChunkWorkerSystem {
private:
    ChunkWorkQueue work_queue;

    std::vector<std::thread> worker_threads;

    std::atomic<bool> running{ true };
    std::atomic<int> active_workers{ 0 };
    std::atomic<size_t> total_work_items_processed{ 0 };

    // Configuration
    static constexpr int DEFAULT_WORKER_COUNT = 12;
    static constexpr size_t MAX_QUEUE_SIZE = 10000;
    static constexpr auto CLEANUP_INTERVAL = std::chrono::seconds(30);

    // Statistics
    std::atomic<size_t> terrain_generated{ 0 };
    std::atomic<size_t> topsoil_generated{ 0 };
    std::atomic<size_t> meshes_generated{ 0 };
    std::atomic<size_t> failed_operations{ 0 };

    // Last cleanup time
    std::chrono::steady_clock::time_point last_cleanup;

public:
    ChunkWorkerSystem(int num_workers = DEFAULT_WORKER_COUNT) : last_cleanup(std::chrono::steady_clock::now()) {
        // Create worker threads
        for (int i = 0; i < num_workers; ++i) {
            worker_threads.emplace_back(&ChunkWorkerSystem::workerLoop, this, i);
        }

        // Create result processor thread
        //result_processor_thread = std::thread(&ChunkWorkerSystem::resultProcessorLoop, this);
    }

    ~ChunkWorkerSystem() {
        shutdown();
    }

    void shutdown() {
        running = false;

        // Signal shutdown to queues
        work_queue.requestShutdown();

        // Wait for all threads to finish
        for (auto& worker : worker_threads) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        worker_threads.clear();
    }

    // Queue work items with better error handling
    bool queueTerrainGeneration(std::shared_ptr<ChunkColumn> chunk, ivec2 position, int distance) {
        if (!chunk || !validateChunkForWork(chunk)) return false;

        if (work_queue.size() >= MAX_QUEUE_SIZE) {
            return false;
        }

        ChunkWorkItem item(ChunkWorkItem::GenerateTerrain, chunk, position, ChunkWorkItem::LOW);
        return work_queue.push(item);
    }

    bool queueTopsoilGeneration(std::shared_ptr<ChunkColumn> chunk, ivec2 position,
        std::array<std::shared_ptr<ChunkColumn>, 4> neighbors) {
        if (!chunk || !validateChunkForWork(chunk)) return false;

        if (work_queue.size() >= MAX_QUEUE_SIZE) {
            return false;
        }

        ChunkWorkItem item(ChunkWorkItem::GenerateTopsoil, chunk, position, neighbors, ChunkWorkItem::NORMAL);
        return work_queue.push(item);
    }

    bool queueTreeGeneration(std::shared_ptr<ChunkColumn> chunk, ivec2 position,
        std::array<std::shared_ptr<ChunkColumn>, 4> neighbors) {
        if (!chunk || !validateChunkForWork(chunk)) return false;

        if (work_queue.size() >= MAX_QUEUE_SIZE) {
            return false;
        }

        ChunkWorkItem item(ChunkWorkItem::GenerateTrees, chunk, position, neighbors, ChunkWorkItem::HIGH);
        return work_queue.push(item);
    }

    bool queueMeshGeneration(std::shared_ptr<ChunkColumn> chunk, ivec2 position,
        std::array<std::shared_ptr<ChunkColumn>, 4> neighbors) {
        if (!chunk || !validateChunkForWork(chunk)) return false;

        if (work_queue.size() >= MAX_QUEUE_SIZE) {
            return false;
        }

        ChunkWorkItem item(ChunkWorkItem::GenerateMesh, chunk, position, neighbors, ChunkWorkItem::CRITICAL);
        return work_queue.push(item);
    }

    bool queueMeshRegeneration(std::shared_ptr<ChunkColumn> chunk, ivec2 position,
        std::array<std::shared_ptr<ChunkColumn>, 4> neighbors) {
        if (!chunk || !validateChunkForWork(chunk)) return false;

        if (work_queue.size() >= MAX_QUEUE_SIZE) {
            return false;
        }

        ChunkWorkItem item(ChunkWorkItem::RegenerateMesh, chunk, position, neighbors, ChunkWorkItem::CRITICAL);
        return work_queue.push(item);
    }

    // Statistics and monitoring
    size_t getQueueSize() const { return work_queue.size(); }
    int getActiveWorkers() const { return active_workers; }
    size_t getTotalProcessed() const { return total_work_items_processed; }

    struct Statistics {
        size_t queue_size;
        size_t result_queue_size;
        int active_workers;
        size_t total_processed;
        size_t terrain_generated;
        size_t topsoil_generated;
        size_t meshes_generated;
        size_t failed_operations;
        size_t total_queued;
        double success_rate;
    };

    Statistics getStatistics() const {
        Statistics stats;
        stats.queue_size = getQueueSize();
        stats.active_workers = getActiveWorkers();
        stats.total_processed = total_work_items_processed;
        stats.terrain_generated = terrain_generated;
        stats.topsoil_generated = topsoil_generated;
        stats.meshes_generated = meshes_generated;
        stats.failed_operations = failed_operations;
        stats.total_queued = work_queue.getTotalQueued();

        if (stats.total_processed > 0) {
            stats.success_rate = 1.0 - (static_cast<double>(stats.failed_operations) / stats.total_processed);
        }
        else {
            stats.success_rate = 0.0;
        }

        return stats;
    }

private:
    bool validateChunkForWork(std::shared_ptr<ChunkColumn> chunk) const {
        return chunk && chunk->getState() != ColumnState::Unloading;
    }

    void workerLoop(int worker_id) {
        while (running) {
            ChunkWorkItem workItem;
            if (work_queue.pop(workItem)) {
                active_workers++;
                auto start_time = std::chrono::steady_clock::now();

                bool success = false;
                std::string error_message;

                try {
                    // Validate work item before processing
                    if (!workItem.isValid()) {
                        error_message = "Invalid work item - chunk was unloaded";
                    }
                    else {
                        success = processWorkItem(workItem, error_message);
                    }
                }
                catch (const std::exception& e) {
                    error_message = std::string("Exception: ") + e.what();
                }

                auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time);

                total_work_items_processed++;
                if (!success) {
                    failed_operations++;
                }

                active_workers--;
            }

            // Periodic cleanup
            performPeriodicCleanup();
        }
    }

    bool processWorkItem(const ChunkWorkItem& workItem, std::string& error_message) {
        switch (workItem.getType()) {
        case ChunkWorkItem::GenerateTerrain:
            return processTerrainGeneration(workItem, error_message);
        case ChunkWorkItem::GenerateTopsoil:
            return processTopsoilGeneration(workItem, error_message);
        case ChunkWorkItem::GenerateTrees:
            return processTreeGeneration(workItem, error_message);
        case ChunkWorkItem::GenerateMesh:
        case ChunkWorkItem::RegenerateMesh:
            return processMeshGeneration(workItem, error_message);
        default:
            error_message = "Unknown work item type";
            return false;
        }
    }

    bool processTerrainGeneration(const ChunkWorkItem& workItem, std::string& error_message) {
        auto chunk = workItem.getChunk();
        if (!chunk) {
            error_message = "Null chunk";
            return false;
        }

        ColumnState currentState = chunk->getState();
        if (currentState != ColumnState::GeneratingTerrain) {
            error_message = "Chunk not in generating terrain state";
            return false;
        }

        chunk->generateTerrain();
        terrain_generated++;
        return true;
    }

    bool processTopsoilGeneration(const ChunkWorkItem& workItem, std::string& error_message) {
        auto chunk = workItem.getChunk();
        if (!chunk) {
            error_message = "Null chunk";
            return false;
        }

        ColumnState currentState = chunk->getState();
        if (currentState != ColumnState::GeneratingTopsoil) {
            error_message = "Chunk not in TerrainReady state";
            return false;
        }

        chunk->generateTopsoil(workItem.getNeighbors());
        topsoil_generated++;
        return true;
    }

    bool processTreeGeneration(const ChunkWorkItem& workItem, std::string& error_message) {
        auto chunk = workItem.getChunk();
        if (!chunk) {
            error_message = "Null chunk";
            return false;
        }

        ColumnState currentState = chunk->getState();
        if (currentState != ColumnState::GeneratingTrees) {
            error_message = "Chunk not in TopsoilReady state";
            return false;
        }

        chunk->generateTrees(workItem.getNeighbors());
        return true;
    }

    bool processMeshGeneration(const ChunkWorkItem& workItem, std::string& error_message) {
        auto chunk = workItem.getChunk();
        if (!chunk) {
            error_message = "Null chunk";
            return false;
        }

        ColumnState currentState = chunk->getState();
        if (currentState != ColumnState::GeneratingMesh) {
            error_message = "Chunk not in completed generation state";
            return false;
        }

        bool success = chunk->generateAllMeshes(workItem.getNeighbors());
        if (success) {
            meshes_generated++;
        }
        else {
            error_message = "Mesh generation failed";
        }
        return success;
    }

    void performPeriodicCleanup() {
        auto now = std::chrono::steady_clock::now();
        if (now - last_cleanup > CLEANUP_INTERVAL) {
            size_t removed = work_queue.removeInvalidItems();
            if (removed > 0) {
                std::cout << "Cleaned up " << removed << " invalid work items" << std::endl;
            }
            last_cleanup = now;
        }
    }
};