from dataset_complexity_profiler.complexity_report import interpret_id_profile


def test_interpret_id_profile():
    report = interpret_id_profile(
        {
            "intrinsic_dim_pca_95": 200,
            "intrinsic_dim_twonn": 12,
        }
    )
    assert report["narrative"]
    assert "estimators" in report
    assert any("nonlinear" in str(item).lower() for item in report["narrative"])
