// ChunkWorkerSystem.h - Fixed version to reduce stuttering
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include "glm/glm.hpp"
#include "ThreadSafeChunk.h"

using glm::ivec3;

struct ChunkWorkItem {
    enum Type {
        GenerateTerrain,
        GenerateMesh,
        GenerateTopsoil,
        RegenerateMesh,
    };

    Type type;
    std::shared_ptr<ThreadSafeChunk> chunk;
    ivec3 position;
    std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors;
    int priority; // NEW: Priority level (higher = more urgent)

    ChunkWorkItem(Type t, std::shared_ptr<ThreadSafeChunk> c, ivec3 pos, int prio = 0)
        : type(t), chunk(c), position(pos), neighbors{}, priority(prio) {
    }

    ChunkWorkItem(Type t, std::shared_ptr<ThreadSafeChunk> c, ivec3 pos,
        std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighs, int prio = 0)
        : type(t), chunk(c), position(pos), neighbors(neighs), priority(prio) {
    }

    bool operator<(const ChunkWorkItem& other) const {
        return priority < other.priority;
    }

};

class ChunkWorkerSystem {
private:
    std::vector<std::thread> workers;

    std::priority_queue<ChunkWorkItem> workQueue;
    //std::queue<ChunkWorkItem> normalWorkQueue;

    mutable std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::atomic<bool> shouldStop{ false };

    static constexpr int NUM_WORKER_THREADS = 8;
    static constexpr size_t MAX_QUEUE_SIZE = 10000;
    static constexpr int HIGH_PRIORITY = 100;
    static constexpr int NORMAL_PRIORITY = 0;

public:
    ChunkWorkerSystem() {
        // Create worker threads
        for (int i = 0; i < NUM_WORKER_THREADS; ++i) {
            workers.emplace_back(&ChunkWorkerSystem::workerThreadFunction, this);
        }
    }

    ~ChunkWorkerSystem() {
        shutdown();
    }

    void shutdown() {
        shouldStop.store(true);
        queueCondition.notify_all();

        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers.clear();
    }

    void queueMeshRegeneration(std::shared_ptr<ThreadSafeChunk> chunk, ivec3 position,
        std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors) {
        if (!chunk) return;

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (workQueue.size() >= MAX_QUEUE_SIZE) {
                return;
            }
            workQueue.emplace(ChunkWorkItem::RegenerateMesh, chunk, position, neighbors, HIGH_PRIORITY);
        }
        queueCondition.notify_all();
    }

    void queueTerrainGeneration(std::shared_ptr<ThreadSafeChunk> chunk, ivec3 position) {
        if (!chunk) return;

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (workQueue.size() >= MAX_QUEUE_SIZE) {
                return;
            }

            workQueue.emplace(ChunkWorkItem::GenerateTerrain, chunk, position, NORMAL_PRIORITY);
        }
        queueCondition.notify_one();
    }

    void queueTopsoilGeneration(std::shared_ptr<ThreadSafeChunk> chunk, ivec3 position,
        std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors) {
        if (!chunk) return;

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (workQueue.size() >= MAX_QUEUE_SIZE) {
                return;
            }
            workQueue.emplace(ChunkWorkItem::GenerateTopsoil, chunk, position, neighbors, HIGH_PRIORITY);
        }
        queueCondition.notify_one();
    }

    void queueMeshGeneration(std::shared_ptr<ThreadSafeChunk> chunk, ivec3 position,
        std::array<std::shared_ptr<ThreadSafeChunk>, 6> neighbors) {
        if (!chunk) return;

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (workQueue.size() >= MAX_QUEUE_SIZE) {
                return;
            }
            workQueue.emplace(ChunkWorkItem::GenerateMesh, chunk, position, neighbors, HIGH_PRIORITY);
        }
        queueCondition.notify_one();
    }

    size_t getQueueSize() const {
        std::lock_guard<std::mutex> lock(queueMutex);
        return workQueue.size();
    }

private:
    void workerThreadFunction() {
        while (!shouldStop.load()) {
            ChunkWorkItem workItem{ ChunkWorkItem::GenerateTerrain, nullptr, ivec3(0) };
            bool hasWork = false;

            {
                std::unique_lock<std::mutex> lock(queueMutex);
                if (queueCondition.wait_for(lock, std::chrono::milliseconds(100),
                    [this] { return !workQueue.empty() || shouldStop.load(); })) {

                    if (shouldStop.load()) {
                        break;
                    }

                    if (!workQueue.empty()) {
                        workItem = workQueue.top();
                        workQueue.pop();
                        hasWork = true;
                    }
                }
            }

            if (hasWork && workItem.chunk) {
                try {
                    switch (workItem.type) {
                    case ChunkWorkItem::GenerateTerrain:
                        processTerrainGeneration(workItem);
                        break;
                    case ChunkWorkItem::GenerateTopsoil:
                        processTopsoilGeneration(workItem);
                        break;
                    case ChunkWorkItem::GenerateMesh:
                    case ChunkWorkItem::RegenerateMesh:
                        processMeshGeneration(workItem);
                        break;
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Worker thread error: " << e.what() << std::endl;
                }
            }
        }
    }

    void processTerrainGeneration(const ChunkWorkItem& workItem) {
        try {
            if (!workItem.chunk) {
                return;
            }

            /*ChunkState currentState = workItem.chunk->getState();
            if (currentState != ChunkState::Empty) {
                return;
            }

            if (currentState == ChunkState::Unloading) {
                return;
            }*/

            workItem.chunk->generateTerrain();
        }
        catch (const std::exception& e) {
            std::cerr << "Terrain generation error: " << e.what() << std::endl;
            if (workItem.chunk && workItem.chunk->getState() != ChunkState::Unloading) {
                workItem.chunk->setState(ChunkState::TerrainReady);
            }
        }
    }

    void processTopsoilGeneration(const ChunkWorkItem& workItem) {
        try {
            if (!workItem.chunk) {
                return;
            }

            /*ChunkState currentState = workItem.chunk->getState();

            if (currentState != ChunkState::TerrainReady) {
                return;
            }

            if (currentState == ChunkState::Unloading) {
                return;
            }*/

            workItem.chunk->generateTopsoil(workItem.neighbors);
        }
        catch (const std::exception& e) {
            std::cerr << "Topsoil generation error: " << e.what() << std::endl;
            if (workItem.chunk && workItem.chunk->getState() != ChunkState::Unloading) {
                workItem.chunk->setState(ChunkState::TopsoilReady);
            }
        }
    }

    void processMeshGeneration(const ChunkWorkItem& workItem) {
        try {
            if (!workItem.chunk) {
                return;
            }

            ChunkState currentState = workItem.chunk->getState();
            /*if (currentState != ChunkState::GeneratingMesh && currentState != ChunkState::RegeneratingMesh) {
                return;
            }

            if (currentState == ChunkState::Unloading) {
                return;
            }*/

            if (workItem.chunk->getSolidVoxels() == 0) {
                workItem.chunk->setState(ChunkState::MeshReady);
                return;
            }

            workItem.chunk->generateMesh(workItem.neighbors);
        }
        catch (const std::exception& e) {
            std::cerr << "Mesh generation error: " << e.what() << std::endl;
            if (workItem.chunk && workItem.chunk->getState() != ChunkState::Unloading) {
                workItem.chunk->setState(ChunkState::MeshReady);
            }
        }
    }
};