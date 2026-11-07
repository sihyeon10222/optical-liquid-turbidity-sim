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
MIN_REFERENCE_SIGNAL_V = 0.01


@dataclass(frozen=True)
class SensorTrace:
    sample_on: FloatArray
    sample_off: FloatArray
    reference_on: FloatArray
    reference_off: FloatArray


@dataclass(frozen=True)
class TransmissionReading:
    name: str
    concentration: float
    laser_scale: float
    raw_transmittance: float
    corrected_transmittance: float


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


def central_taylor_intensity(
    position_m: FloatArray, concentration: float
) -> FloatArray:
    phase = diffraction_phase(position_m)
    if np.any(np.abs(phase) > 1.0):
        raise ValueError("The central diffraction approximation requires |phase| <= 1.")
    phase_squared = phase**2
    intensity_factor = 1.0 - phase_squared / 3.0 + 2.0 * phase_squared**2 / 45.0
    return (
        true_transmittance(concentration) * intensity_factor
        + diffuse_intensity(position_m, concentration)
    )


def fringe_visibility(concentration: float) -> float:
    first_zero_angle = np.arcsin(WAVELENGTH_M / SLIT_WIDTH_M)
    first_zero_m = SCREEN_DISTANCE_M * np.tan(first_zero_angle)
    center, first_zero = diffraction_intensity(
        np.array([0.0, first_zero_m]), concentration
    )
    return float((center - first_zero) / (center + first_zero))


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


def sample_reference_ratio(trace: SensorTrace) -> float:
    reference_signal = float(np.mean(trace.reference_on - trace.reference_off))
    if reference_signal <= MIN_REFERENCE_SIGNAL_V:
        raise ValueError("Reference signal is too weak for a transmission estimate.")
    sample_signal = float(np.mean(trace.sample_on - trace.sample_off))
    if sample_signal <= 0.0:
        raise ValueError("Sample signal is too weak for a transmission estimate.")
    return sample_signal / reference_signal


def collect_readings() -> list[TransmissionReading]:
    rng = np.random.default_rng(SEED)
    blank = measure(0.0, 1.0, rng)
    blank_ratio = sample_reference_ratio(blank)
    readings = [TransmissionReading("blank", 0.0, 1.0, 1.0, 1.0)]
    cases = [
        (f"calibration_{concentration:.0%}", concentration, 1.0)
        for concentration in CALIBRATION_CONCENTRATIONS
    ]
    cases.append(("unknown", UNKNOWN_CONCENTRATION, UNKNOWN_LASER_SCALE))
    for name, concentration, scale in cases:
        trace = measure(concentration, scale, rng)
        readings.append(
            TransmissionReading(
                name=name,
                concentration=concentration,
                laser_scale=scale,
                raw_transmittance=float(trace.sample_on.mean() / blank.sample_on.mean()),
                corrected_transmittance=sample_reference_ratio(trace) / blank_ratio,
            )
        )
    return readings


def fit_attenuation(calibration: list[TransmissionReading]) -> float:
    concentrations = np.array([reading.concentration for reading in calibration])
    optical_depths = -np.log([reading.corrected_transmittance for reading in calibration])
    attenuation = float(
        np.dot(concentrations, optical_depths) / np.dot(concentrations, concentrations)
    )
    if attenuation <= 0.0:
        raise ValueError("Calibration does not show positive attenuation.")
    return attenuation


def estimate_concentration(transmittance: float, attenuation: float) -> float:
    return float(-np.log(transmittance) / attenuation)


def main() -> None:
    _, *calibration, unknown = collect_readings()
    attenuation = fit_attenuation(calibration)
    estimate = estimate_concentration(unknown.corrected_transmittance, attenuation)
    truth = true_transmittance(unknown.concentration)
    print(f"Fitted attenuation: {attenuation:.6f}")
    print(
        f"Unknown transmission (true / raw / corrected): {truth:.6f} / "
        f"{unknown.raw_transmittance:.6f} / {unknown.corrected_transmittance:.6f}"
    )
    print(
        "Absolute transmission error (raw / corrected): "
        f"{abs(unknown.raw_transmittance - truth):.6f} / "
        f"{abs(unknown.corrected_transmittance - truth):.6f}"
    )
    print(
        f"Estimated concentration: {estimate:.4%} "
        f"(absolute error {abs(estimate - unknown.concentration) * 100:.4f} pp)"
    )
    if not 0.0 <= estimate <= max(item.concentration for item in calibration):
        print("Warning: estimate is outside the calibrated range.")
    print(f"Unknown fringe visibility: {fringe_visibility(unknown.concentration):.6f}")


if __name__ == "__main__":
    main()
