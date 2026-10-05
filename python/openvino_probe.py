"""
Probe OpenVINO devices available to WaveAccel.

Author: Santosh Kumar
Copyright 2026 Santosh Kumar
SPDX-License-Identifier: Apache-2.0
"""

import openvino as ov


def main() -> None:
    print("OpenVINO:", ov.__version__)

    core = ov.Core()

    print("Available OpenVINO devices:")
    for device in core.available_devices:
        print(f"  {device}")


if __name__ == "__main__":
    main()
