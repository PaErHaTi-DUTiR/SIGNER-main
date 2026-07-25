from scripts.verify_release import verify_no_secrets, verify_sample


def test_sample_integrity():
    assert verify_sample() == []


def test_repository_contains_no_obvious_secrets():
    assert verify_no_secrets() == []

