import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
AUTOBOOST = ROOT / "auto-boost-3.0"
if str(AUTOBOOST) not in sys.path:
    sys.path.insert(0, str(AUTOBOOST))

import ab_psd  # noqa: E402
import auto_boost  # noqa: E402
from utils.pipeline_runtime import PROJECT_PSD_SCRIPT, load_toolchain  # noqa: E402


def write_minimal_scenes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"frames": 1, "scenes": [{"start_frame": 0, "end_frame": 1}]}),
        encoding="utf-8",
    )


class AbPsdVpyTest(unittest.TestCase):
    def test_toolchain_prefers_project_local_psd(self) -> None:
        old_value = os.environ.pop("PBBATCH_PSD_SCRIPT", None)
        try:
            self.assertEqual(Path(load_toolchain().psd_script), PROJECT_PSD_SCRIPT)
        finally:
            if old_value is not None:
                os.environ["PBBATCH_PSD_SCRIPT"] = old_value

    def test_run_psd_forwards_vspipe_args(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            psd_script = root / "Progressive-Scene-Detection.py"
            input_vpy = root / "proxy.vpy"
            source = root / "source.mkv"
            scenes = root / "psd" / "scenes.json"
            psd_script.write_text("# test psd\n", encoding="utf-8")
            input_vpy.write_text("# test vpy\n", encoding="utf-8")
            source.write_bytes(b"")
            captured: dict[str, object] = {}

            def fake_run_cmd(cmd, **kwargs):
                captured["cmd"] = list(cmd)
                captured["kwargs"] = dict(kwargs)
                write_minimal_scenes(scenes)

            old_run_cmd = ab_psd.run_cmd
            try:
                ab_psd.run_cmd = fake_run_cmd
                ab_psd.run_psd(
                    psd_script=psd_script,
                    psd_python=None,
                    input_file=input_vpy,
                    base_scenes_path=scenes,
                    extra_args="",
                    vspipe_args=[f"src={source}", "pass_name=fast"],
                    event_source=source,
                )
            finally:
                ab_psd.run_cmd = old_run_cmd

            cmd = captured["cmd"]
            self.assertEqual(cmd[cmd.index("-i") + 1], str(input_vpy))
            self.assertEqual(cmd[cmd.index("--vspipe-arg") + 1], f"src={source}")
            self.assertIn("pass_name=fast", cmd)

    def test_auto_boost_uses_proxy_vpy_for_psd_input(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mkv"
            fast_vpy = root / "fast.vpy"
            proxy_vpy = root / "proxy.vpy"
            psd_script = root / "Progressive-Scene-Detection.py"
            workdir = root / "work"
            for path in (source, fast_vpy, proxy_vpy, psd_script):
                path.write_text("", encoding="utf-8")
            captured: dict[str, object] = {}

            def fake_run_psd(**kwargs):
                captured.update(kwargs)
                write_minimal_scenes(kwargs["base_scenes_path"])

            old_argv = sys.argv[:]
            old_run_psd = auto_boost.run_psd
            old_which = auto_boost.which_or_none
            try:
                auto_boost.run_psd = fake_run_psd
                auto_boost.which_or_none = lambda _exe: "av1an"
                sys.argv = [
                    "auto_boost.py",
                    "--input",
                    str(source),
                    "--temp",
                    str(workdir),
                    "--run-stages",
                    "psd",
                    "--sdm",
                    "psd",
                    "--psd-script",
                    str(psd_script),
                    "--fastpass-vpy",
                    str(fast_vpy),
                    "--fastpass-proxy",
                    str(proxy_vpy),
                    "--fastpass-vspipe-arg",
                    "pass_name=fast",
                ]
                self.assertEqual(auto_boost.main(), 0)
            finally:
                sys.argv = old_argv
                auto_boost.run_psd = old_run_psd
                auto_boost.which_or_none = old_which

            self.assertEqual(captured["input_file"], proxy_vpy.resolve())
            self.assertEqual(captured["event_source"], source.resolve())
            self.assertEqual(captured["vspipe_args"], [f"src={source.resolve()}", "pass_name=fast"])


if __name__ == "__main__":
    unittest.main()
