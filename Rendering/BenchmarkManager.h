#ifndef BENCHMARK_MANAGER_H
#define BENCHMARK_MANAGER_H
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <webgpu/webgpu.hpp>
using namespace wgpu;

struct TimestampQuery {
    QuerySet querySet;
    Buffer resolveBuffer;
    Buffer readbackBuffer;
    uint32_t queryCount;
    Future mapHandle;  // Changed from BufferMapCallback to Future
    std::function<void(double)> callback;
    bool isMapped = false;  // Track mapping state
};

class BenchmarkManager {
private:
    std::unordered_map<std::string, std::shared_ptr<TimestampQuery>> queries;
    Device device;
    Queue queue;
    bool timestampSupported;

    // Helper methods
    double calculateFrameTime(const uint64_t* timestamps, uint32_t count);
    bool isQueryBusy(const std::string& name);  // Check if query is being mapped

public:
    BenchmarkManager(Device d, Queue q);

    // Core functionality
    bool initialize();
    QuerySet createQuerySet(const std::string& name, uint32_t queryCount = 2);
    QuerySet getQuerySet(const std::string& name);

    // Timestamp operations
    void writeTimestamp(const std::string& queryName, CommandEncoder& encoder, uint32_t queryIndex);
    void resolveTimestamps(const std::string& queryName, CommandEncoder& encoder);
    void readTimestamps(const std::string& queryName, std::function<void(double)> callback = nullptr);

    // Convenience methods
    void beginFrame(const std::string& queryName, CommandEncoder& encoder);
    void endFrame(const std::string& queryName, CommandEncoder& encoder);
    void processFrameTime(const std::string& queryName, std::function<void(double)> callback = nullptr);

    // Utility
    bool isTimestampSupported() const { return timestampSupported; }
    void deleteQuery(const std::string& name);
    void terminate();
};

#endif // BENCHMARK_MANAGER_H