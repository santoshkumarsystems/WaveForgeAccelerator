// Author: Santosh Kumar
// Copyright 2026 Santosh Kumar
// SPDX-License-Identifier: Apache-2.0

#include "waveaccel/execution_plan.hpp"

#include <charconv>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace waveaccel {
namespace {

using Fields = std::unordered_map<std::string, std::string>;

std::string trim_cr(std::string value) {
    if (!value.empty() && value.back() == '\r') {
        value.pop_back();
    }
    return value;
}

Fields read_fields(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open execution-plan manifest: " + path);
    }

    std::string magic;
    if (!std::getline(input, magic)) {
        throw std::runtime_error("Execution-plan manifest is empty: " + path);
    }
    magic = trim_cr(std::move(magic));

    if (magic != "WAVEACCEL_EXECUTION_PLAN_V1") {
        throw std::runtime_error(
            "Unsupported execution-plan manifest header: " + magic);
    }

    Fields fields;
    std::string line;
    std::size_t line_number = 1;

    while (std::getline(input, line)) {
        ++line_number;
        line = trim_cr(std::move(line));

        if (line.empty() || line[0] == '#') {
            continue;
        }

        const auto equals = line.find('=');
        if (equals == std::string::npos || equals == 0) {
            throw std::runtime_error(
                "Malformed execution-plan line " +
                std::to_string(line_number) + ": " + line);
        }

        auto key = line.substr(0, equals);
        auto value = line.substr(equals + 1);

        if (!fields.emplace(std::move(key), std::move(value)).second) {
            throw std::runtime_error(
                "Duplicate execution-plan key at line " +
                std::to_string(line_number));
        }
    }

    return fields;
}

const std::string& required(
    const Fields& fields,
    const std::string& key) {

    const auto it = fields.find(key);
    if (it == fields.end()) {
        throw std::runtime_error(
            "Missing execution-plan key: " + key);
    }
    return it->second;
}

std::size_t parse_size(
    const Fields& fields,
    const std::string& key) {

    const auto& text = required(fields, key);
    std::size_t value{};

    const char* first = text.data();
    const char* last = first + text.size();
    const auto [ptr, error] = std::from_chars(first, last, value);

    if (error != std::errc{} || ptr != last) {
        throw std::runtime_error(
            "Invalid integer for execution-plan key " +
            key + ": " + text);
    }

    return value;
}

std::vector<std::string> parse_actions(
    const Fields& fields,
    const std::string& prefix) {

    const auto count = parse_size(fields, prefix + "action_count");
    std::vector<std::string> actions;
    actions.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        actions.push_back(required(
            fields,
            prefix + "action." + std::to_string(i)));
    }

    return actions;
}

}  // namespace

ExecutionPlan load_execution_plan_manifest(
    const std::string& path) {

    const auto fields = read_fields(path);

    ExecutionPlan plan;
    plan.schema_version = static_cast<std::uint32_t>(
        parse_size(fields, "schema_version"));

    if (plan.schema_version != 1) {
        throw std::runtime_error(
            "Unsupported execution-plan schema version: " +
            std::to_string(plan.schema_version));
    }

    plan.format = required(fields, "format");
    if (plan.format != "waveaccel.execution_plan") {
        throw std::runtime_error(
            "Unexpected execution-plan format: " + plan.format);
    }

    plan.tvm_version = required(fields, "producer.tvm_version");

    plan.target.name = required(fields, "target.name");

    plan.target.contract_schema_version =
        static_cast<std::uint32_t>(
            parse_size(
                fields,
                "target.contract_schema_version"));

    if (plan.target.contract_schema_version != 1) {
        throw std::runtime_error(
            "Unsupported WaveAccel target-contract schema version: " +
            std::to_string(plan.target.contract_schema_version));
    }

    plan.target.dtype =
        required(fields, "target.dtype");

    plan.target.element_bytes =
        parse_size(fields, "target.element_bytes");

    if (plan.target.element_bytes == 0) {
        throw std::runtime_error(
            "WaveAccel target element size must be positive");
    }

    const auto supported_op_count =
        parse_size(fields, "target.supported_op_count");

    if (supported_op_count == 0) {
        throw std::runtime_error(
            "WaveAccel target must declare at least one supported op");
    }

    plan.target.supported_ops.reserve(supported_op_count);

    for (std::size_t i = 0; i < supported_op_count; ++i) {
        plan.target.supported_ops.push_back(
            required(
                fields,
                "target.supported_op." + std::to_string(i)));
    }

    plan.target.model_initializer_memory_space =
        required(
            fields,
            "target.memory_space.model_initializers");

    plan.target.initializer_staging_memory_space =
        required(
            fields,
            "target.memory_space.initializer_staging");

    plan.target.gemm_tiling_axis =
        required(
            fields,
            "target.gemm_policy.tiling_axis");

    plan.target.gemm_tile_unit =
        required(
            fields,
            "target.gemm_policy.tile_unit");

    plan.target.gemm_tiled_fixed_initializer =
        required(
            fields,
            "target.gemm_policy.tiled_fixed_initializer");

    plan.target.sram_bytes =
        parse_size(fields, "target.sram_bytes");
    plan.target.transient_reservation_bytes =
        parse_size(fields, "target.transient_reservation_bytes");
    plan.target.initializer_staging_capacity_bytes =
        parse_size(fields, "target.initializer_staging_capacity_bytes");

    if (plan.target.transient_reservation_bytes >
        plan.target.sram_bytes) {
        throw std::runtime_error(
            "Transient reservation exceeds SRAM capacity");
    }

    const auto expected_capacity =
        plan.target.sram_bytes -
        plan.target.transient_reservation_bytes;

    if (expected_capacity !=
        plan.target.initializer_staging_capacity_bytes) {
        throw std::runtime_error(
            "Initializer staging capacity is inconsistent with target SRAM");
    }

    const auto node_count =
        parse_size(fields, "node_count");
    plan.nodes.reserve(node_count);

    for (std::size_t n = 0; n < node_count; ++n) {
        const std::string prefix =
            "node." + std::to_string(n) + ".";

        ExecutionPlanNode node;
        node.node_index =
            parse_size(fields, prefix + "node_index");

        if (node.node_index != n) {
            throw std::runtime_error(
                "Execution-plan node indices must be contiguous");
        }

        node.op_type = required(fields, prefix + "op_type");
        node.strategy = required(fields, prefix + "strategy");
        node.initializer_total_bytes =
            parse_size(fields, prefix + "initializer_total_bytes");
        node.actions = parse_actions(fields, prefix);

        if (node.op_type == "Gemm") {
            node.tvm_primfunc =
                required(fields, prefix + "tvm_primfunc");
            node.in_features =
                parse_size(fields, prefix + "in_features");
            node.out_features =
                parse_size(fields, prefix + "out_features");
            node.weight_bytes =
                parse_size(fields, prefix + "weight_bytes");
            node.weight_row_bytes =
                parse_size(fields, prefix + "weight_row_bytes");
            node.bias_bytes =
                parse_size(fields, prefix + "bias_bytes");

            if (node.weight_bytes + node.bias_bytes !=
                node.initializer_total_bytes) {
                throw std::runtime_error(
                    "Gemm initializer byte total mismatch at node " +
                    std::to_string(n));
            }

            if (node.strategy == "TILED") {
                node.compiler_tile_width =
                    parse_size(fields, prefix + "compiler_tile_width");

                const auto tile_count =
                    parse_size(fields, prefix + "tile_count");

                node.tiles.reserve(tile_count);

                std::size_t expected_output_start = 0;
                std::size_t tiled_weight_bytes = 0;

                for (std::size_t t = 0; t < tile_count; ++t) {
                    const std::string tile_prefix =
                        prefix + "tile." + std::to_string(t) + ".";

                    ExecutionPlanTile tile;
                    tile.tile_index =
                        parse_size(fields, tile_prefix + "tile_index");
                    tile.output_start =
                        parse_size(fields, tile_prefix + "output_start");
                    tile.output_end =
                        parse_size(fields, tile_prefix + "output_end");
                    tile.output_count =
                        parse_size(fields, tile_prefix + "output_count");
                    tile.weight_bytes =
                        parse_size(fields, tile_prefix + "weight_bytes");
                    tile.actions =
                        parse_actions(fields, tile_prefix);

                    if (tile.tile_index != t) {
                        throw std::runtime_error(
                            "Tile indices must be contiguous at node " +
                            std::to_string(n));
                    }

                    if (tile.output_start != expected_output_start) {
                        throw std::runtime_error(
                            "Tile output ranges are not contiguous at node " +
                            std::to_string(n));
                    }

                    if (tile.output_count == 0 ||
                        tile.output_end + 1 !=
                            tile.output_start + tile.output_count) {
                        throw std::runtime_error(
                            "Invalid tile output range at node " +
                            std::to_string(n));
                    }

                    if (tile.weight_bytes !=
                        tile.output_count * node.weight_row_bytes) {
                        throw std::runtime_error(
                            "Tile weight bytes do not match row geometry at node " +
                            std::to_string(n));
                    }

                    expected_output_start =
                        tile.output_end + 1;
                    tiled_weight_bytes +=
                        tile.weight_bytes;

                    node.tiles.push_back(std::move(tile));
                }

                if (expected_output_start != node.out_features) {
                    throw std::runtime_error(
                        "Tiles do not cover all Gemm outputs at node " +
                        std::to_string(n));
                }

                if (tiled_weight_bytes != node.weight_bytes) {
                    throw std::runtime_error(
                        "Tiled weight bytes do not cover full Gemm weight at node " +
                        std::to_string(n));
                }
            }
        }

        plan.nodes.push_back(std::move(node));
    }

    return plan;
}

}  // namespace waveaccel
