from astrai.dataset import RDSampler


def test_random_sampler_consistency(random_dataset):
    """Test RandomSampler produces consistent results with same seed"""
    dataset = random_dataset

    # Create two samplers with same seed
    sampler1 = RDSampler(dataset, seed=42)
    sampler2 = RDSampler(dataset, seed=42)

    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))

    assert indices1 == indices2


def test_random_sampler_different_seeds(random_dataset):
    """Test RandomSampler produces different results with different seeds"""
    dataset = random_dataset

    # Create two samplers with different seeds
    sampler1 = RDSampler(dataset, seed=42)
    sampler2 = RDSampler(dataset, seed=123)

    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))

    # Very high probability they should be different
    assert indices1 != indices2


def test_sampler_across_epochs(random_dataset):
    """Test sampler behavior across multiple epochs"""
    dataset = random_dataset
    n = len(dataset)

    sampler = RDSampler(dataset, seed=42)

    # Get indices for first epoch
    epoch1_indices = list(iter(sampler))
    assert len(epoch1_indices) == n

    # Get indices for second epoch
    epoch2_indices = list(iter(sampler))
    assert len(epoch2_indices) == n

    # Check that epochs have different order (should be random)
    assert epoch1_indices != epoch2_indices

    # Check that all indices are present in each epoch
    assert set(epoch1_indices) == set(range(n))
    assert set(epoch2_indices) == set(range(n))
