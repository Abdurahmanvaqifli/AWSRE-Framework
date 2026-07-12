"""
AWSRE Benchmark Configuration Test
"""

from benchmark.config import (
    BenchmarkConfig,
    create_development_config,
    create_full_config,
    create_smoke_test_config,
)


def run_test():
    print("=" * 72)
    print("AWSRE BENCHMARK CONFIGURATION TEST")
    print("=" * 72)

    smoke = create_smoke_test_config()

    assert isinstance(
        smoke,
        BenchmarkConfig,
    )

    assert smoke.max_host_images == 2
    assert smoke.methods == ["DCT"]
    assert smoke.expected_experiment_count == 16

    smoke.create_directories()

    config_path = smoke.save_json()

    assert config_path.exists()
    assert config_path.stat().st_size > 0

    development = create_development_config()

    assert development.max_host_images == 10
    assert development.expected_experiment_count > 0

    full = create_full_config()

    assert full.max_host_images == 100
    assert len(full.attacks) == 14
    assert full.expected_experiment_count > (
        development.expected_experiment_count
    )

    print(
        "\nSmoke experiments:",
        smoke.expected_experiment_count,
    )

    print(
        "Development experiments:",
        development.expected_experiment_count,
    )

    print(
        "Full experiments:",
        full.expected_experiment_count,
    )

    print(
        "Saved configuration:",
        config_path,
    )

    print("\n✅ BENCHMARK CONFIGURATION TEST PASSED")


if __name__ == "__main__":
    run_test()
