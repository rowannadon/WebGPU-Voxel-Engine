struct RLEPair {
    uint16_t value;
    uint16_t count;
};

class RunLengthEncoder {
public:
    // Encode with length parameter - works with raw pointer or array data
    static std::vector<RLEPair> encode(const uint16_t* input, size_t length) {
        std::vector<RLEPair> encoded;

        // Handle empty input case
        if (length == 0 || input == nullptr) {
            return encoded;
        }

        uint16_t currentValue = input[0];
        uint16_t currentCount = 1;

        for (size_t i = 1; i < length; ++i) {
            if (input[i] == currentValue && currentCount < UINT16_MAX) {
                currentCount++;
            }
            else {
                // Save the current run
                encoded.push_back({ currentValue, currentCount });
                // Start a new run
                currentValue = input[i];
                currentCount = 1;
            }
        }

        // Don't forget the last run
        encoded.push_back({ currentValue, currentCount });
        return encoded;
    }

    // Convenience overload for std::array
    template<size_t N>
    static std::vector<RLEPair> encode(const std::array<uint16_t, N>& input) {
        return encode(input.data(), N);
    }

    // Convenience overload for std::vector
    static std::vector<RLEPair> encode(const std::vector<uint16_t>& input) {
        return encode(input.data(), input.size());
    }

    // Decode with length parameter - fills provided buffer
    static void decode(const std::vector<RLEPair>& encoded, uint16_t* output, size_t length) {
        // Initialize output buffer to zeros
        std::fill(output, output + length, 0);

        // Handle empty encoded data
        if (encoded.empty() || output == nullptr || length == 0) {
            return;
        }

        size_t index = 0;
        for (const auto& pair : encoded) {
            // Validate the pair data before using it
            if (pair.count == 0) {
                continue; // Skip invalid pairs with zero count
            }

            for (uint16_t i = 0; i < pair.count && index < length; ++i) {
                output[index++] = pair.value;
            }

            // Safety check - if we've filled the buffer, break
            if (index >= length) {
                break;
            }
        }
    }

    // Convenience overload for std::array
    template<size_t N>
    static std::array<uint16_t, N> decode(const std::vector<RLEPair>& encoded) {
        std::array<uint16_t, N> decoded;
        decode(encoded, decoded.data(), N);
        return decoded;
    }

    // Convenience overload that returns std::vector
    static std::vector<uint16_t> decode(const std::vector<RLEPair>& encoded, size_t length) {
        std::vector<uint16_t> decoded(length, 0);
        decode(encoded, decoded.data(), length);
        return decoded;
    }

    // Calculate compression ratio with length parameter
    static double getCompressionRatio(const std::vector<RLEPair>& encoded, size_t originalLength) {
        if (encoded.empty() || originalLength == 0) {
            return 1.0; // No compression if empty
        }

        size_t originalSize = originalLength * sizeof(uint16_t);
        size_t compressedSize = encoded.size() * sizeof(RLEPair);
        return static_cast<double>(originalSize) / compressedSize;
    }
};