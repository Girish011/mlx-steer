def test_public_api_imports():
    import mlx_steer

    assert hasattr(mlx_steer, "SteeringEngine")
    assert hasattr(mlx_steer, "CHIMonitor")
    assert hasattr(mlx_steer, "compute_steering_vectors")
    assert hasattr(mlx_steer, "get_all_pairs")

