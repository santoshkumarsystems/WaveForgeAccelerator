/**
 * @file generate_cuwaves_dataset.cpp
 * @brief Generate WaveAccel training data through the public cuWaves API.
 *
 * WaveAccel does not duplicate the cuWaves wave equation.
 * Every field is produced by cuwaves::compute_wave_cpu().
 *
 * V1 inverse problem:
 *
 *     64 spatial field samples -> amplitude + wavelength
 *
 * Copyright 2026 Santosh Kumar
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuwaves/wave1d.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

constexpr std::size_t kFieldSamples = 64;

constexpr float kX0Meters = 0.0f;
constexpr float kDxMeters = 0.05f;
constexpr float kTimeSeconds = 0.0f;

constexpr float kFrequencyHz = 1.0f;
constexpr float kPhaseRad = 0.0f;

constexpr float kMinAmplitude = 0.5f;
constexpr float kMaxAmplitude = 3.0f;

constexpr float kMinWavelengthMeters = 0.4f;
constexpr float kMaxWavelengthMeters = 2.0f;

void write_header(std::ofstream& out) {
    out << "sample_id"
        << ",amplitude"
        << ",wavelength"
        << ",frequency_hz"
        << ",phase_rad"
        << ",time_s"
        << ",x0_m"
        << ",dx_m";

    for (std::size_t i = 0; i < kFieldSamples; ++i) {
        out << ",x" << i;
    }

    out << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output_path =
            argc >= 2
                ? std::filesystem::path(argv[1])
                : std::filesystem::path(
                      "data/raw/cuwaves_wave_dataset.csv"
                  );

        const std::size_t dataset_size =
            argc >= 3
                ? static_cast<std::size_t>(
                      std::stoull(argv[2])
                  )
                : 10000U;

        const unsigned int seed =
            argc >= 4
                ? static_cast<unsigned int>(
                      std::stoul(argv[3])
                  )
                : 20261004U;

        if (dataset_size == 0U) {
            throw std::invalid_argument(
                "dataset size must be greater than zero"
            );
        }

        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(
                output_path.parent_path()
            );
        }

        std::ofstream out(output_path);

        if (!out) {
            throw std::runtime_error(
                "unable to open output file: "
                + output_path.string()
            );
        }

        std::mt19937 rng(seed);

        std::uniform_real_distribution<float> amplitude_dist(
            kMinAmplitude,
            kMaxAmplitude
        );

        std::uniform_real_distribution<float> wavelength_dist(
            kMinWavelengthMeters,
            kMaxWavelengthMeters
        );

        write_header(out);
        out << std::setprecision(9);

        for (
            std::size_t sample_id = 0;
            sample_id < dataset_size;
            ++sample_id
        ) {
            const float amplitude =
                amplitude_dist(rng);

            const float wavelength =
                wavelength_dist(rng);

            const cuwaves::Wave1DConfig config{
                amplitude,
                kFrequencyHz,
                wavelength,
                kPhaseRad
            };

            const auto field =
                cuwaves::compute_wave_cpu(
                    kFieldSamples,
                    kX0Meters,
                    kDxMeters,
                    kTimeSeconds,
                    config
                );

            out << sample_id
                << ',' << amplitude
                << ',' << wavelength
                << ',' << kFrequencyHz
                << ',' << kPhaseRad
                << ',' << kTimeSeconds
                << ',' << kX0Meters
                << ',' << kDxMeters;

            for (const float value : field) {
                out << ',' << value;
            }

            out << '\n';
        }

        std::cout
            << "WaveAccel dataset generated through cuWaves\n"
            << "cuWaves release:  v0.0.1\n"
            << "samples:          " << dataset_size << '\n'
            << "field width:      " << kFieldSamples << '\n'
            << "target:           amplitude + wavelength\n"
            << "seed:             " << seed << '\n'
            << "output:           " << output_path << '\n';

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr
            << "error: "
            << ex.what()
            << '\n';

        return 1;
    }
}
