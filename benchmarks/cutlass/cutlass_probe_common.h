#pragma once

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

namespace uxsched::cutlass_probe
{

inline uint64_t Fnva64(const void *data, size_t bytes)
{
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    const auto *ptr = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<uint64_t>(ptr[i]);
        hash *= kPrime;
    }
    return hash;
}

inline std::string Hex64(uint64_t value)
{
    std::ostringstream os;
    os << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return os.str();
}

inline std::string JsonEscape(const std::string &value)
{
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += ch; break;
        }
    }
    return out;
}

inline void JsonField(std::ostream &os, const char *name, const std::string &value, bool comma = true)
{
    os << '"' << name << "\":\"" << JsonEscape(value) << '"';
    if (comma) os << ',';
}

inline void JsonField(std::ostream &os, const char *name, const char *value, bool comma = true)
{
    JsonField(os, name, std::string(value == nullptr ? "" : value), comma);
}

inline void JsonField(std::ostream &os, const char *name, bool value, bool comma = true)
{
    os << '"' << name << "\":" << (value ? "true" : "false");
    if (comma) os << ',';
}

template <typename T>
inline void JsonNumberField(std::ostream &os, const char *name, T value, bool comma = true)
{
    os << '"' << name << "\":" << value;
    if (comma) os << ',';
}

inline void JsonFloatField(std::ostream &os, const char *name, double value, bool comma = true)
{
    os << '"' << name << "\":" << std::setprecision(17) << value;
    if (comma) os << ',';
}

} // namespace uxsched::cutlass_probe
