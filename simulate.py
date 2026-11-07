"""Simulate a dual-beam transmission sensor and a single-slit pattern."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SEED = 42
WAVELENGTH_M = 650e-9
SLIT_WIDTH_M = 80e-6
SCREEN_DISTANCE_M = 0.60
CUVETTE_LENGTH_M = 0.01
EXTINCTION_PER_M = 160.0  # At relative concentration 1.
MODULATION_HZ = 500
FLICKER_HZ = 120
CYCLE_COUNT = 100
CALIBRATION_CONCENTRATIONS = (0.25, 0.50, 1.00)
UNKNOWN_CONCENTRATION = 0.65
UNKNOWN_LASER_SCALE = 0.80


@dataclass(frozen=True)
class SensorTrace:
    sample_on: FloatArray
    sample_off: FloatArray
    reference_on: FloatArray
    reference_off: FloatArray


def true_transmittance(concentration: float) -> float:
    return float(np.exp(-EXTINCTION_PER_M * CUVETTE_LENGTH_M * concentration))


def diffraction_phase(position_m: FloatArray) -> FloatArray:
    angle = np.arctan(position_m / SCREEN_DISTANCE_M)
    return np.pi * SLIT_WIDTH_M * np.sin(angle) / WAVELENGTH_M


def diffuse_intensity(position_m: FloatArray, concentration: float) -> FloatArray:
    # A phenomenological broad halo for the illustration.
    return 0.12 * concentration * np.exp(-0.5 * (position_m / 0.007) ** 2)


def diffraction_intensity(position_m: FloatArray, concentration: float) -> FloatArray:
    phase = diffraction_phase(position_m)
    coherent_intensity = true_transmittance(concentration) * np.sinc(phase / np.pi) ** 2
    return coherent_intensity + diffuse_intensity(position_m, concentration)


def measure(
    concentration: float, laser_scale: float, rng: np.random.Generator
) -> SensorTrace:
    time_on = np.arange(CYCLE_COUNT) / MODULATION_HZ
    time_off = time_on + 0.5 / MODULATION_HZ
    ambient_phase = rng.uniform(0.0, 2.0 * np.pi)
    flicker_on = np.sin(2.0 * np.pi * FLICKER_HZ * time_on + ambient_phase)
    flicker_off = np.sin(2.0 * np.pi * FLICKER_HZ * time_off + ambient_phase)
    noise = rng.normal(0.0, 0.003, size=(4, CYCLE_COUNT))

    sample_on = 0.19 + 0.035 * flicker_on + noise[0]
    sample_on += 1.2 * laser_scale * true_transmittance(concentration)
    sample_off = 0.19 + 0.035 * flicker_off + noise[1]
    reference_on = 0.24 + 0.045 * flicker_on + noise[2] + 0.9 * laser_scale
    reference_off = 0.24 + 0.045 * flicker_off + noise[3]
    return SensorTrace(sample_on, sample_off, reference_on, reference_off)


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("relative_concentration,true_transmittance,sample_on_mean_V")
    for concentration in (0.0, *CALIBRATION_CONCENTRATIONS, UNKNOWN_CONCENTRATION):
        scale = UNKNOWN_LASER_SCALE if concentration == UNKNOWN_CONCENTRATION else 1.0
        trace = measure(concentration, scale, rng)
        print(f"{concentration:.2f},{true_transmittance(concentration):.6f},{trace.sample_on.mean():.6f}")

    center_intensity = diffraction_intensity(np.array([0.0]), UNKNOWN_CONCENTRATION)[0]
    print(f"Unknown sample central diffraction intensity: {center_intensity:.6f} a.u.")


if __name__ == "__main__":
    main()
