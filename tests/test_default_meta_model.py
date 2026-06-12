def test_default_meta_model_loads():
    from dataset_complexity_profiler import DatasetProfiler

    profiler = DatasetProfiler(auto_load_meta_model=True)
    assert profiler.is_fitted is True
