/*
 * WaveAccel - AI Accelerator Runtime Prototype
 *
 * ONNX memory and graph-manifest loader.
 *
 * Author: Santosh Kumar
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include "waveaccel/model_memory.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace waveaccel {

namespace {

std::size_t parse_size(
    const std::string& key,
    const std::string& value
) {
    try {
        return static_cast<std::size_t>(std::stoull(value));
    } catch (const std::exception&) {
        throw std::runtime_error(
            "invalid numeric value for manifest key: " + key
        );
    }
}

bool parse_indexed_key(
    const std::string& key,
    const std::string& prefix,
    std::size_t& index,
    std::string& field
) {
    if (key.rfind(prefix, 0) != 0) {
        return false;
    }

    const std::string remainder =
        key.substr(prefix.size());

    const auto separator = remainder.find('_');

    if (separator == std::string::npos) {
        return false;
    }

    const std::string index_string =
        remainder.substr(0, separator);

    if (index_string.empty()) {
        return false;
    }

    for (const char ch : index_string) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }

    index = parse_size(key, index_string);
    field = remainder.substr(separator + 1);

    return !field.empty();
}

std::vector<std::string> split_csv(
    const std::string& value
) {
    std::vector<std::string> result;

    if (value.empty()) {
        return result;
    }

    std::size_t begin = 0;

    while (begin <= value.size()) {
        const auto separator = value.find(',', begin);

        if (separator == std::string::npos) {
            result.push_back(value.substr(begin));
            break;
        }

        result.push_back(
            value.substr(begin, separator - begin)
        );

        begin = separator + 1;
    }

    return result;
}

void ensure_initializer_slot(
    ModelMemoryInfo& info,
    std::size_t index
) {
    if (info.initializers.size() <= index) {
        info.initializers.resize(index + 1);
    }
}

void ensure_node_slot(
    ModelMemoryInfo& info,
    std::size_t index
) {
    if (info.nodes.size() <= index) {
        info.nodes.resize(index + 1);
    }
}

bool initializer_exists(
    const ModelMemoryInfo& info,
    const std::string& name
) {
    return std::any_of(
        info.initializers.begin(),
        info.initializers.end(),
        [&name](const ModelTensorInfo& tensor) {
            return tensor.name == name;
        }
    );
}

}  // namespace

ModelMemoryInfo load_model_memory_manifest(const std::string& path) {
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error(
            "unable to open model memory manifest: " + path
        );
    }

    ModelMemoryInfo info;
    std::string line;

    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }

        const auto separator = line.find('=');

        if (separator == std::string::npos) {
            throw std::runtime_error(
                "invalid manifest line: " + line
            );
        }

        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);

        if (key == "batch_size") {
            info.batch_size = parse_size(key, value);
        } else if (key == "input_bytes") {
            info.input_bytes = parse_size(key, value);
        } else if (key == "output_bytes") {
            info.output_bytes = parse_size(key, value);
        } else if (key == "initializer_count") {
            info.initializer_count = parse_size(key, value);
        } else if (key == "initializer_elements") {
            info.initializer_elements = parse_size(key, value);
        } else if (key == "initializer_bytes") {
            info.initializer_bytes = parse_size(key, value);
        } else if (key == "largest_initializer_name") {
            info.largest_initializer_name = value;
        } else if (key == "largest_initializer_bytes") {
            info.largest_initializer_bytes = parse_size(key, value);
        } else if (key == "largest_activation_name") {
            info.largest_activation_name = value;
        } else if (key == "largest_activation_bytes") {
            info.largest_activation_bytes = parse_size(key, value);
        } else if (key == "operator_sequence") {
            info.operator_sequence = value;
        } else if (key == "node_count") {
            info.node_count = parse_size(key, value);
        } else {
            std::size_t index = 0;
            std::string field;

            if (parse_indexed_key(
                    key,
                    "initializer_",
                    index,
                    field
                )) {
                ensure_initializer_slot(info, index);
                auto& tensor = info.initializers[index];

                if (field == "name") {
                    tensor.name = value;
                } else if (field == "shape") {
                    tensor.shape = value;
                } else if (field == "elements") {
                    tensor.elements = parse_size(key, value);
                } else if (field == "bytes") {
                    tensor.bytes = parse_size(key, value);
                }

                continue;
            }

            if (parse_indexed_key(
                    key,
                    "node_",
                    index,
                    field
                )) {
                ensure_node_slot(info, index);
                auto& node = info.nodes[index];

                if (field == "op") {
                    node.op = value;
                } else if (field == "inputs") {
                    node.inputs = split_csv(value);
                } else if (field == "outputs") {
                    node.outputs = split_csv(value);
                } else if (field == "initializer_count") {
                    node.initializer_count =
                        parse_size(key, value);
                } else if (field == "initializers") {
                    node.initializers = split_csv(value);
                }

                continue;
            }
        }
    }

    if (info.input_bytes == 0 ||
        info.output_bytes == 0 ||
        info.initializer_bytes == 0) {
        throw std::runtime_error(
            "model memory manifest is missing required values"
        );
    }

    if (info.initializer_count != info.initializers.size()) {
        throw std::runtime_error(
            "initializer count does not match tensor records"
        );
    }

    if (info.node_count != info.nodes.size()) {
        throw std::runtime_error(
            "node count does not match graph-node records"
        );
    }

    std::size_t calculated_elements = 0;
    std::size_t calculated_bytes = 0;

    for (const auto& tensor : info.initializers) {
        if (tensor.name.empty() ||
            tensor.shape.empty() ||
            tensor.elements == 0 ||
            tensor.bytes == 0) {
            throw std::runtime_error(
                "initializer tensor record is incomplete"
            );
        }

        calculated_elements += tensor.elements;
        calculated_bytes += tensor.bytes;
    }

    if (calculated_elements != info.initializer_elements) {
        throw std::runtime_error(
            "initializer element total does not match manifest"
        );
    }

    if (calculated_bytes != info.initializer_bytes) {
        throw std::runtime_error(
            "initializer byte total does not match manifest"
        );
    }

    for (const auto& node : info.nodes) {
        if (node.op.empty()) {
            throw std::runtime_error(
                "graph node is missing an operator type"
            );
        }

        if (node.outputs.empty()) {
            throw std::runtime_error(
                "graph node is missing output tensors"
            );
        }

        if (node.initializer_count != node.initializers.size()) {
            throw std::runtime_error(
                "graph node initializer count does not match "
                "initializer dependency list"
            );
        }

        for (const auto& initializer_name : node.initializers) {
            if (!initializer_exists(info, initializer_name)) {
                throw std::runtime_error(
                    "graph node references unknown initializer: " +
                    initializer_name
                );
            }
        }
    }

    return info;
}

}  // namespace waveaccel
