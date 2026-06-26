"""Tests for the Pipeline Benchmark module (video_analysis/benchmark.py)."""

import time
from video_analysis.benchmark import GPUProfiler, PipelineBenchmark, StageRecord


class TestGPUProfiler:
    def test_basic_profiling(self):
        """GPUProfiler captures elapsed time and VRAM stats even without GPU."""
        with GPUProfiler("test") as prof:
            time.sleep(0.01)
        assert prof.elapsed >= 0.01
        assert prof.label == "test"
        # start_mib may be >0 on GPU-equipped machines — just check it's a number
        assert isinstance(prof.start_mib, (int, float))
        assert prof.peak_mib >= 0.0

    def test_to_stage_record(self):
        prof = GPUProfiler("test")
        with prof:
            time.sleep(0.005)
        rec = prof.to_stage_record("my_stage")
        assert isinstance(rec, StageRecord)
        assert rec.name == "my_stage"
        assert rec.duration_s >= 0.005

    def test_gpu_used_flag_no_gpu(self):
        prof = GPUProfiler()
        assert prof._gpu_available is False


class TestPipelineBenchmark:
    def test_empty_benchmark(self):
        bm = PipelineBenchmark("empty")
        bm.finish()
        report = bm.report()
        assert "Pipeline Benchmark: empty" in report
        assert "Stages: 0" in report

    def test_benchmark_stages(self):
        bm = PipelineBenchmark("test")
        with bm:
            with GPUProfiler("stage1") as p1:
                time.sleep(0.01)
            bm.record_stage("stage1", p1)

            with GPUProfiler("stage2") as p2:
                time.sleep(0.005)
            bm.record_stage("stage2", p2)

        assert len(bm.stages) == 2
        assert bm.stages[0].name == "stage1"
        assert bm.stages[1].name == "stage2"
        assert bm.total_duration >= 0.015

    def test_as_dict(self):
        bm = PipelineBenchmark("test_dict")
        with GPUProfiler("stage1") as p:
            time.sleep(0.01)
        bm.record_stage("stage1", p)
        bm.finish()

        d = bm.as_dict()
        assert d["label"] == "test_dict"
        assert len(d["stages"]) == 1
        assert d["stages"][0]["name"] == "stage1"
        assert d["stages"][0]["duration_s"] >= 0.01

    def test_context_manager(self):
        with PipelineBenchmark("ctx") as bm:
            with GPUProfiler("s1") as p:
                time.sleep(0.005)
            bm.record_stage("s1", p)
        assert len(bm.stages) == 1

    def test_report_formatting(self):
        bm = PipelineBenchmark("fmt_test")
        with GPUProfiler("s1") as p:
            time.sleep(0.005)
        bm.record_stage("s1", p)
        bm.finish()
        report = bm.report()
        assert "Stage" in report
        assert "Duration (s)" in report
        assert "VRAM Start" in report
        assert "VRAM Peak" in report
