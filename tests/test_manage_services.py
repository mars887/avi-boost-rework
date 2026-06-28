import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils import workdir_layout as layout
from utils.manage import av1an_state, reset as manage_reset, zone_patch
from utils.manage.analytics import collect_pass_rows, sort_pass_rows, SortSpec
from utils.manage.av1an_state import (
    ChunkPatch,
    FrameRange,
    MoveTarget,
    chunk_name,
    load_chunks,
    load_done,
    merge_chunks,
    move_chunks,
    reindex_chunks_transactionally,
    reshape_chunk_range,
    save_chunks,
    set_chunk_frame_range,
    sort_chunks,
    split_chunk,
    swap_chunks,
    update_chunk_params,
)
from utils.manage.config import (
    compute_config_fingerprint,
    confirm_fingerprint,
    stale_config_warnings,
    validate_zone_text,
)
from utils.manage.backup import ManageTransaction
from utils.manage.context import MODE_FULL, context_from_plan, context_from_workdir, make_runner_item
from utils.manage.discovery import discover_workdirs, is_workdir, resolve_argument_to_refs
from utils.manage.scenes import (
    SceneSelector,
    ZonePatch,
    load_scene_file,
    patch_scene_region,
    rebuild_split_scenes_for_chunks,
    validate_scene_file,
)
from utils.manage.status import (
    STATE_COMPLETED,
    STATE_NOT_STARTED,
    STATE_STALE_MARKER,
    get_stage_statuses,
    is_runner_active,
    summarize_workdir,
)
from utils.manage.store import WorkdirStore, import_workdir_files
from utils.plan import FilePlan, PlanMeta, PlanPaths, VideoPlan, save_plan
from utils.runner_state import (
    STAGE_AUTOBOOST_SCENE,
    STAGE_DEMUX,
    STAGE_FASTPASS,
    STAGE_HDR_PATCH,
    STAGE_MAINPASS,
    STAGE_MUX,
    STAGE_SSIMU2,
    STAGE_VERIFY,
    STAGE_ZONE_BOUNDARIES,
    STAGE_ZONE_RECALC,
    autoboost_stage4_scenes,
    display_stage_plan,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_chunk(index: int, start: int, end: int, temp: str, **extra):
    chunk = {
        "temp": temp,
        "index": index,
        "input": {"path": "x", "video_params": []},
        "proxy": None,
        "output_ext": "ivf",
        "start_frame": start,
        "end_frame": end,
        "frame_rate": 24.0,
        "passes": 1,
        "video_params": ["--crf", "30"],
        "encoder": "svt-av1",
        "pb_debug": {"keep": True},
    }
    chunk.update(extra)
    return chunk


def win_token(text: str):
    """Native OsString token as av1an's default 'safe' chunks-cmd-format emits."""
    return {"Windows": [ord(ch) for ch in text]}


def vspipe_cmd(start: int, end: int, *, native: bool = True):
    """vspipe source_cmd for a chunk [start, end): ``-s start -e end-1``."""
    tokens = ["vspipe", "main.vpy", "-c", "y4m", "-", "-s", str(start), "-e", str(end - 1), "-a", "src=x"]
    return [win_token(tok) for tok in tokens] if native else list(tokens)


class ManageFixture:
    """Temp source dir + plan + workdir with av1an fastpass state."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.source = self.root / "episode.mkv"
        self.source.write_bytes(b"\x00" * 2048)
        plan = FilePlan(
            meta=PlanMeta(name="episode"),
            paths=PlanPaths(source="episode.mkv"),
            video=VideoPlan(track_id=0, action="edit"),
        )
        self.plan_path = self.root / "episode.plan"
        save_plan(plan, self.plan_path)
        self.workdir = self.root / "episode"

        self.state_dir = self.workdir / ".state"
        self.meta_dir = self.workdir / "00_meta"
        self.video_dir = self.workdir / "video"
        self.video_state = self.video_dir / ".state"
        self.fastpass = self.video_dir / "fastpass"
        self.mainpass = self.video_dir / "mainpass"
        for directory in (
            self.state_dir,
            self.meta_dir,
            self.video_state,
            self.fastpass / "encode",
            self.fastpass / "split",
            self.mainpass / "encode",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (self.workdir / "zoned_command.txt").write_text("", encoding="utf-8")

        temp = str(self.fastpass)
        chunks = [
            make_chunk(0, 0, 100, temp),
            make_chunk(1, 100, 200, temp),
            make_chunk(2, 200, 300, temp),
        ]
        write_json(self.fastpass / "chunks.json", chunks)
        write_json(
            self.fastpass / "done.json",
            {
                "frames": 300,
                "done": {
                    "00000": {"frames": 100, "size_bytes": 11},
                    "00001": {"frames": 100, "size_bytes": 22},
                    "00002": {"frames": 100, "size_bytes": 33},
                },
                "audio_done": False,
                "pb_extra": 1,
            },
        )
        for index in range(3):
            (self.fastpass / "encode" / f"{chunk_name(index)}.ivf").write_bytes(b"v" * (10 + index))
        (self.fastpass / "split" / "00002_fpf.log").write_text("fpf", encoding="utf-8")
        (self.fastpass / "split" / "v_00002_probe.bin").write_bytes(b"p")
        (self.fastpass / "split" / "2.json").write_text("{}", encoding="utf-8")

        scenes = {
            "frames": 300,
            "scenes": [
                {"start_frame": 0, "end_frame": 100, "zone_overrides": None},
                {"start_frame": 100, "end_frame": 200, "zone_overrides": None},
                {"start_frame": 200, "end_frame": 300, "zone_overrides": None},
            ],
            "split_scenes": [
                {
                    "start_frame": 0,
                    "end_frame": 100,
                    "zone_overrides": {"video_params": ["--crf", "28"], "min_scene_len": 24},
                },
                {
                    "start_frame": 100,
                    "end_frame": 200,
                    "zone_overrides": {"video_params": ["--crf", "30"], "min_scene_len": 24},
                },
                {
                    "start_frame": 200,
                    "end_frame": 300,
                    "zone_overrides": {"video_params": ["--crf", "32"], "min_scene_len": 24},
                },
            ],
            "pb_meta": {"keep": "yes"},
        }
        write_json(self.video_dir / "scenes.json", scenes)
        write_json(self.fastpass / "scenes.json", scenes)
        # av1an scene detection (--sc-only) writes the base scenes here.
        self.psd = self.video_dir / "psd"
        self.psd.mkdir(parents=True, exist_ok=True)
        write_json(self.psd / "scenes.psd.json", scenes)

        # completed early stages
        (self.state_dir / "DEMUX_DONE").write_text("ok\n", encoding="utf-8")
        write_json(self.meta_dir / "demux_manifest.json", {"source": str(self.source), "subs": []})
        (self.video_state / "SCENE_DETECTION_COMPLETED").write_text("ok\n", encoding="utf-8")
        (self.video_state / "FASTPASS_COMPLETED").write_text("ok\n", encoding="utf-8")
        (self.video_state / "SSIMU2_COMPLETED").write_text("ok\n", encoding="utf-8")
        (self.fastpass / "episode.fastpass.mkv").write_bytes(b"f" * 2048)
        (self.fastpass / "episode_ssimu2.log").write_text(
            "skip: 3\n0: 70.5\n3: 71.2\n6: 69.0\n99: 80.0\n", encoding="utf-8"
        )
        write_json(
            self.meta_dir / "source_info.json",
            {"source": str(self.source), "plan": str(self.plan_path)},
        )
        events = [
            {
                "event": "runner",
                "plan": str(self.plan_path),
                "mode": "full",
                "stage": STAGE_DEMUX,
                "status": "completed",
                "message": "",
                "timestamp": 100.0,
                "session_id": "s1",
                "plan_run_id": "r1",
                "source": str(self.source),
                "workdir": str(self.workdir),
                "progress": -1.0,
                "started_at": 90.0,
                "ended_at": 100.0,
                "elapsed_seconds": 10.0,
            }
        ]
        with (self.meta_dir / "runner_events.jsonl").open("w", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event) + "\n")

    def context(self):
        return context_from_plan(self.plan_path)


class ContextDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_context_from_plan(self) -> None:
        ctx = self.fixture.context()
        self.assertEqual(ctx.workdir, self.fixture.workdir)
        self.assertEqual(ctx.mode, "full")
        self.assertIn(STAGE_FASTPASS, ctx.stage_names)
        self.assertIn(STAGE_MAINPASS, ctx.stage_names)

    def test_context_from_workdir_recovers_plan(self) -> None:
        ctx = context_from_workdir(self.fixture.workdir)
        self.assertIsNotNone(ctx.resolved_plan)
        self.assertEqual(ctx.plan_path, self.fixture.plan_path)

    def test_context_from_workdir_without_plan(self) -> None:
        self.fixture.plan_path.unlink()
        (self.fixture.meta_dir / "source_info.json").unlink()
        ctx = context_from_workdir(self.fixture.workdir)
        self.assertIsNone(ctx.resolved_plan)
        self.assertTrue(ctx.stage_names)
        self.assertTrue(any("best-effort" in warning for warning in ctx.warnings))

    def test_discovery_workdir_and_plan(self) -> None:
        refs = resolve_argument_to_refs(self.fixture.workdir)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].workdir, self.fixture.workdir)
        self.assertEqual(refs[0].plan_path, self.fixture.plan_path)

        refs = resolve_argument_to_refs(self.fixture.plan_path)
        self.assertEqual(refs[0].workdir, self.fixture.workdir)

        refs = discover_workdirs([str(self.fixture.root)])
        self.assertEqual(len(refs), 1)

    def test_is_workdir(self) -> None:
        self.assertTrue(is_workdir(self.fixture.workdir))
        self.assertFalse(is_workdir(self.fixture.root / "missing"))


class StatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.ctx = self.fixture.context()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_runner_not_active(self) -> None:
        self.assertFalse(is_runner_active(self.ctx))

    def test_stage_states(self) -> None:
        statuses = {status.name: status for status in get_stage_statuses(self.ctx)}
        self.assertEqual(statuses[STAGE_DEMUX].state, STATE_COMPLETED)
        self.assertEqual(statuses[STAGE_MAINPASS].state, STATE_NOT_STARTED)
        # fastpass artifacts valid -> completed
        self.assertEqual(statuses[STAGE_FASTPASS].state, STATE_COMPLETED)

    def test_stale_marker_detection(self) -> None:
        (self.fixture.meta_dir / "demux_manifest.json").unlink()
        statuses = {status.name: status for status in get_stage_statuses(self.ctx)}
        self.assertEqual(statuses[STAGE_DEMUX].state, STATE_STALE_MARKER)

    def test_summary(self) -> None:
        summary = summarize_workdir(self.ctx)
        self.assertEqual(summary.mode, "full")
        self.assertGreater(summary.total_stages, 5)
        self.assertFalse(summary.runner.active)


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_docs_and_events(self) -> None:
        store = WorkdirStore(Path(self._tmp.name) / "store.sqlite")
        with store:
            store.put_doc("ui_state", "tab", {"value": "Status"})
            self.assertEqual(store.get_doc("ui_state", "tab"), {"value": "Status"})
            store.put_doc("ui_state", "tab", {"value": "Scenes"})
            self.assertEqual(store.get_doc("ui_state", "tab"), {"value": "Scenes"})
            store.append_event("stage_events", {"stage": "Demux", "status": "completed", "timestamp": 1.0})
            store.append_event("stage_events", {"stage": "Demux", "status": "failed", "timestamp": 2.0})
            latest = store.latest_event("stage_events", stage="Demux")
            self.assertEqual(latest["status"], "failed")

    def test_transaction_rollback(self) -> None:
        store = WorkdirStore(Path(self._tmp.name) / "store.sqlite")
        with store:
            store.put_doc("docs", "keep", {"value": 1})
            with self.assertRaises(RuntimeError):
                with store.transaction():
                    store.put_doc("docs", "keep", {"value": 2})
                    raise RuntimeError("boom")
            self.assertEqual(store.get_doc("docs", "keep"), {"value": 1})

    def test_import_jsonl_tail(self) -> None:
        store = WorkdirStore(Path(self._tmp.name) / "store.sqlite")
        with store:
            stats = import_workdir_files(store, self.fixture.workdir)
            self.assertEqual(stats["runner_events"], 1)
            # second import reads nothing new
            stats = import_workdir_files(store, self.fixture.workdir)
            self.assertEqual(stats["runner_events"], 0)
            # append one more line -> only the tail is read
            with (self.fixture.meta_dir / "runner_events.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"stage": "Fastpass", "status": "started", "timestamp": 3.0}) + "\n")
            stats = import_workdir_files(store, self.fixture.workdir)
            self.assertEqual(stats["runner_events"], 1)
            events = store.events("stage_events")
            self.assertEqual(len(events), 2)


class ChunksDoneTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.pass_dir = self.fixture.fastpass

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parse_and_save_roundtrip(self) -> None:
        chunks = load_chunks(self.pass_dir)
        self.assertEqual([chunk.index for chunk in chunks], [0, 1, 2])
        save_chunks(self.pass_dir, chunks)
        raw = json.loads((self.pass_dir / "chunks.json").read_text(encoding="utf-8"))
        self.assertIsInstance(raw, list)
        self.assertEqual(raw[0]["pb_debug"], {"keep": True})  # unknown fields preserved

        done = load_done(self.pass_dir)
        self.assertEqual(done.frames, 300)
        self.assertTrue(done.is_done(1))
        self.assertEqual(done.data["pb_extra"], 1)

    def test_mark_chunk_not_done_delete(self) -> None:
        av1an_state.mark_chunk_not_done(self.pass_dir, 2, policy="delete")
        done = load_done(self.pass_dir)
        self.assertFalse(done.is_done(2))
        self.assertEqual(done.frames, 300)  # project frame count untouched
        self.assertFalse((self.pass_dir / "encode" / "00002.ivf").exists())
        self.assertFalse((self.pass_dir / "split" / "00002_fpf.log").exists())
        self.assertFalse((self.pass_dir / "split" / "v_00002_probe.bin").exists())
        self.assertFalse((self.pass_dir / "split" / "2.json").exists())

    def test_mark_chunk_not_done_quarantine(self) -> None:
        av1an_state.mark_chunk_not_done(self.pass_dir, 1, policy="quarantine")
        self.assertFalse((self.pass_dir / "encode" / "00001.ivf").exists())
        quarantined = list((self.pass_dir / "encode").glob("00001_deleted-*.ivf"))
        self.assertEqual(len(quarantined), 1)

    def test_swap_and_move(self) -> None:
        swap_chunks(self.pass_dir, 0, 2)
        self.assertEqual([chunk.index for chunk in load_chunks(self.pass_dir)], [2, 1, 0])
        move_chunks(self.pass_dir, [1], MoveTarget(kind="start"))
        self.assertEqual([chunk.index for chunk in load_chunks(self.pass_dir)], [1, 2, 0])
        move_chunks(self.pass_dir, [1, 2], MoveTarget(kind="after", anchor_index=0), order="descending")
        self.assertEqual([chunk.index for chunk in load_chunks(self.pass_dir)], [0, 2, 1])

    def test_sort(self) -> None:
        sort_chunks(self.pass_dir, "long-to-short")
        self.assertEqual(len(load_chunks(self.pass_dir)), 3)
        sort_chunks(self.pass_dir, "sequential")
        self.assertEqual([chunk.start_frame for chunk in load_chunks(self.pass_dir)], [0, 100, 200])
        sort_chunks(self.pass_dir, "ssimu2", key_values={0: 70.0, 1: 60.0, 2: 80.0})
        self.assertEqual([chunk.index for chunk in load_chunks(self.pass_dir)], [1, 0, 2])

    def test_update_chunk_params_resets_done(self) -> None:
        was_done = update_chunk_params(self.pass_dir, 0, ChunkPatch(video_params=["--crf", "20"]))
        self.assertTrue(was_done)
        done = load_done(self.pass_dir)
        self.assertFalse(done.is_done(0))
        chunks = load_chunks(self.pass_dir)
        self.assertEqual(chunks[0].video_params, ["--crf", "20"])
        self.assertFalse((self.pass_dir / "encode" / "00000.ivf").exists())


class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.pass_dir = self.fixture.fastpass

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reindex(self) -> None:
        result = reindex_chunks_transactionally(self.pass_dir, {1: 5, 2: 6})
        self.assertEqual(result.mapping, {1: 5, 2: 6})
        chunks = {chunk.index for chunk in load_chunks(self.pass_dir)}
        self.assertEqual(chunks, {0, 5, 6})
        done = load_done(self.pass_dir)
        self.assertEqual(set(done.done.keys()), {"00000", "00005", "00006"})
        self.assertTrue((self.pass_dir / "encode" / "00005.ivf").exists())
        self.assertTrue((self.pass_dir / "encode" / "00006.ivf").exists())
        self.assertTrue((self.pass_dir / "split" / "00006_fpf.log").exists())
        self.assertTrue((self.pass_dir / "split" / "v_00006_probe.bin").exists())
        self.assertTrue((self.pass_dir / "split" / "6.json").exists())

    def test_split_chunk(self) -> None:
        result = split_chunk(self.pass_dir, 1, 150)
        self.assertEqual(result.reset_indices, [1])
        self.assertEqual(result.new_indices, [2])
        chunks = load_chunks(self.pass_dir)
        self.assertEqual(
            [(chunk.index, chunk.start_frame, chunk.end_frame) for chunk in chunks],
            [(0, 0, 100), (1, 100, 150), (2, 150, 200), (3, 200, 300)],
        )
        done = load_done(self.pass_dir)
        # chunk 0 untouched, old chunk 2 remapped to 3 with state preserved
        self.assertTrue(done.is_done(0))
        self.assertFalse(done.is_done(1))
        self.assertFalse(done.is_done(2))
        self.assertTrue(done.is_done(3))
        self.assertTrue((self.pass_dir / "encode" / "00003.ivf").exists())
        self.assertFalse((self.pass_dir / "encode" / "00001.ivf").exists())
        self.assertTrue((self.pass_dir / "split" / "00003_fpf.log").exists())

    def test_merge_chunks(self) -> None:
        split_chunk(self.pass_dir, 1, 150)  # 0,1,2,3
        result = merge_chunks(self.pass_dir, [1, 2])
        self.assertEqual(result.new_indices, [1])
        chunks = load_chunks(self.pass_dir)
        self.assertEqual(
            [(chunk.index, chunk.start_frame, chunk.end_frame) for chunk in chunks],
            [(0, 0, 100), (1, 100, 200), (2, 200, 300)],
        )
        done = load_done(self.pass_dir)
        self.assertTrue(done.is_done(0))
        self.assertFalse(done.is_done(1))
        self.assertTrue(done.is_done(2))  # remapped from 3
        self.assertTrue((self.pass_dir / "encode" / "00002.ivf").exists())

    def test_merge_requires_adjacency(self) -> None:
        with self.assertRaises(av1an_state.Av1anStateError):
            merge_chunks(self.pass_dir, [0, 2])

    def test_reshape_range(self) -> None:
        result = reshape_chunk_range(self.pass_dir, FrameRange(50, 250))
        chunks = load_chunks(self.pass_dir)
        ranges = sorted((chunk.start_frame, chunk.end_frame) for chunk in chunks)
        self.assertEqual(ranges, [(0, 50), (50, 250), (250, 300)])
        self.assertTrue(result.reset_indices)

    def test_rebuild_split_scenes(self) -> None:
        split_chunk(self.pass_dir, 1, 150)
        chunks = load_chunks(self.pass_dir)
        scene_path = self.fixture.video_dir / "scenes.json"
        rebuild_split_scenes_for_chunks(scene_path, chunks)
        scene_file = load_scene_file(scene_path)
        self.assertEqual(
            [(scene["start_frame"], scene["end_frame"]) for scene in scene_file.split_scenes],
            [(0, 100), (100, 150), (150, 200), (200, 300)],
        )
        # zone overrides inherited from covering old scene
        self.assertEqual(scene_file.split_scenes[2]["zone_overrides"]["video_params"], ["--crf", "30"])
        self.assertEqual(scene_file.data["pb_meta"], {"keep": "yes"})

    def test_rebuild_keeps_scenes_mirroring_split_scenes(self) -> None:
        # fixture scenes/split_scenes share boundaries -> both must follow chunks
        split_chunk(self.pass_dir, 1, 150)
        chunks = load_chunks(self.pass_dir)
        scene_path = self.fixture.video_dir / "scenes.json"
        rebuild_split_scenes_for_chunks(scene_path, chunks)
        scene_file = load_scene_file(scene_path)
        expected = [(0, 100), (100, 150), (150, 200), (200, 300)]
        self.assertEqual([(s["start_frame"], s["end_frame"]) for s in scene_file.split_scenes], expected)
        self.assertEqual([(s["start_frame"], s["end_frame"]) for s in scene_file.scenes], expected)


def decode_cmd(cmd):
    out = []
    for token in cmd:
        if isinstance(token, str):
            out.append(token)
        else:
            out.append("".join(chr(unit) for unit in token["Windows"]))
    return out


def cmd_seek(cmd):
    """Return (start, end-inclusive) parsed from a decoded vspipe command."""
    tokens = decode_cmd(cmd)
    start = int(tokens[tokens.index("-s") + 1])
    end = int(tokens[tokens.index("-e") + 1])
    return start, end


class ChunkSourceCmdTest(unittest.TestCase):
    """Geometry edits must rewrite source_cmd/proxy_cmd or av1an FRAME MISMATCHes."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.pass_dir = self.fixture.fastpass
        temp = str(self.pass_dir)
        chunks = [
            make_chunk(0, 0, 100, temp, source_cmd=vspipe_cmd(0, 100), proxy_cmd=vspipe_cmd(0, 100)),
            make_chunk(1, 100, 200, temp, source_cmd=vspipe_cmd(100, 200), proxy_cmd=vspipe_cmd(100, 200)),
            make_chunk(2, 200, 300, temp, source_cmd=vspipe_cmd(200, 300), proxy_cmd=vspipe_cmd(200, 300)),
        ]
        write_json(self.pass_dir / "chunks.json", chunks)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seek_by_range(self):
        return {
            (c.start_frame, c.end_frame): (cmd_seek(c.data["source_cmd"]), cmd_seek(c.data["proxy_cmd"]))
            for c in load_chunks(self.pass_dir)
        }

    def test_set_chunk_frame_range_rewrites_native_tokens(self) -> None:
        chunk = load_chunks(self.pass_dir)[1]
        set_chunk_frame_range(chunk, 100, 175)
        self.assertEqual(chunk.start_frame, 100)
        self.assertEqual(chunk.end_frame, 175)
        self.assertEqual(cmd_seek(chunk.data["source_cmd"]), (100, 174))
        self.assertEqual(cmd_seek(chunk.data["proxy_cmd"]), (100, 174))
        # tokens stay in native OsString form, only the number changed
        self.assertIsInstance(chunk.data["source_cmd"][6], dict)

    def test_rewrite_text_and_select_formats(self) -> None:
        text_chunk = av1an_state.ChunkState(
            data=make_chunk(0, 0, 100, "t", source_cmd=vspipe_cmd(0, 100, native=False))
        )
        set_chunk_frame_range(text_chunk, 10, 60)
        self.assertEqual(text_chunk.data["source_cmd"][6], "10")
        self.assertEqual(text_chunk.data["source_cmd"][8], "59")

        select_chunk = av1an_state.ChunkState(
            data=make_chunk(0, 0, 100, "t", source_cmd=[
                "ffmpeg", "-i", "in.mkv", "-vf", r"select=between(n\,0\,99)", "-f", "yuv4mpegpipe", "-",
            ])
        )
        set_chunk_frame_range(select_chunk, 10, 60)
        self.assertEqual(select_chunk.data["source_cmd"][4], r"select=between(n\,10\,59)")

    def test_split_rewrites_both_sides(self) -> None:
        split_chunk(self.pass_dir, 1, 150)
        seeks = self._seek_by_range()
        self.assertEqual(seeks[(100, 150)][0], (100, 149))  # left source
        self.assertEqual(seeks[(100, 150)][1], (100, 149))  # left proxy
        self.assertEqual(seeks[(150, 200)][0], (150, 199))  # right source
        self.assertEqual(seeks[(150, 200)][1], (150, 199))  # right proxy
        # untouched chunk keeps its command
        self.assertEqual(seeks[(0, 100)][0], (0, 99))

    def test_merge_rewrites_first_chunk(self) -> None:
        merge_chunks(self.pass_dir, [0, 1])
        seeks = self._seek_by_range()
        self.assertEqual(seeks[(0, 200)][0], (0, 199))
        self.assertEqual(seeks[(0, 200)][1], (0, 199))
        self.assertEqual(seeks[(200, 300)][0], (200, 299))

    def test_reshape_rewrites_seam_chunks(self) -> None:
        reshape_chunk_range(self.pass_dir, FrameRange(50, 250))
        for chunk in load_chunks(self.pass_dir):
            self.assertEqual(
                cmd_seek(chunk.data["source_cmd"]),
                (chunk.start_frame, chunk.end_frame - 1),
                msg=f"chunk {chunk.index} [{chunk.start_frame},{chunk.end_frame})",
            )


class ZonePatchTest(unittest.TestCase):
    """Bulk command-driven video_params patching across many chunks + scenes."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.pass_dir = self.fixture.fastpass
        self.scene_path = self.fixture.video_dir / "scenes.json"
        self.video = zone_patch.build_video_info(None, 24.0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _patch(self, text: str):
        commands = zone_patch.parse_patch_commands(text, video=self.video, total_frames=300)
        return zone_patch.apply_param_commands_to_chunks(self.pass_dir, commands, policy="delete")

    def test_command_selects_and_edits_many_chunks(self) -> None:
        result = self._patch("100f300 - --crf 25 --preset 4")
        self.assertEqual(result.changed_indices, [1, 2])
        chunks = {c.index: c.video_params for c in load_chunks(self.pass_dir)}
        self.assertEqual(chunks[0], ["--crf", "30"])  # untouched
        self.assertEqual(chunks[1], ["--crf", "25.00", "--preset", "4"])
        self.assertEqual(chunks[2], ["--crf", "25.00", "--preset", "4"])

    def test_done_chunks_reset_and_outputs_removed(self) -> None:
        result = self._patch("100f300 - --crf 25")
        self.assertEqual(result.reset_indices, [1, 2])
        done = load_done(self.pass_dir)
        self.assertTrue(done.is_done(0))
        self.assertFalse(done.is_done(1))
        self.assertFalse(done.is_done(2))
        self.assertFalse((self.pass_dir / "encode" / "00001.ivf").exists())

    def test_relative_crf_uses_current_value(self) -> None:
        self._patch("0f100 - --crf -5")
        chunks = {c.index: c.video_params for c in load_chunks(self.pass_dir)}
        self.assertEqual(chunks[0], ["--crf", "25.00"])  # 30 - 5, formatted 2dp
        self.assertEqual(chunks[1], ["--crf", "30"])  # not selected

    def test_mirror_to_scene_file_keeps_other_overrides(self) -> None:
        result = self._patch("100f300 - --crf 25 --preset 4")
        updated = zone_patch.mirror_params_to_scene_file(self.scene_path, result.range_params)
        self.assertEqual(updated, 4)  # scenes[1,2] + split_scenes[1,2]
        scene_file = load_scene_file(self.scene_path)
        self.assertEqual(
            scene_file.split_scenes[1]["zone_overrides"]["video_params"], ["--crf", "25.00", "--preset", "4"]
        )
        self.assertEqual(scene_file.split_scenes[1]["zone_overrides"]["min_scene_len"], 24)  # preserved
        self.assertEqual(scene_file.scenes[2]["zone_overrides"]["video_params"], ["--crf", "25.00", "--preset", "4"])
        self.assertIsNone(scene_file.scenes[0].get("zone_overrides"))  # untouched

    def test_no_match_is_a_noop(self) -> None:
        result = self._patch("5000f6000 - --crf 25")
        self.assertEqual(result.changed_indices, [])
        chunks = {c.index: c.video_params for c in load_chunks(self.pass_dir)}
        self.assertEqual(chunks[0], ["--crf", "30"])

    def test_boundary_mode_is_rejected(self) -> None:
        with self.assertRaises(zone_patch.ZonePatchError):
            zone_patch.parse_patch_commands("100f300 | min=9 | --crf 25", video=self.video, total_frames=300)

    def test_scene_index_selector_matches_chunk_index(self) -> None:
        result = self._patch("2s2 - --crf 20")
        self.assertEqual(result.changed_indices, [2])


class EditParamsMultiTest(unittest.TestCase):
    """Multi-chunk Edit Params: merged {variable} view + per-chunk apply."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.pass_dir = Path(self._tmp.name) / "video" / "fastpass"
        (self.pass_dir / "encode").mkdir(parents=True)
        temp = str(self.pass_dir)
        write_json(
            self.pass_dir / "chunks.json",
            [
                make_chunk(0, 0, 100, temp, video_params=["--crf", "30", "--preset", "2", "--scd", "0"]),
                make_chunk(1, 100, 200, temp, video_params=["--crf", "31", "--preset", "2", "--scd", "0"]),
                make_chunk(2, 200, 300, temp, video_params=["--crf", "32", "--preset", "2", "--scd", "0"]),
            ],
        )
        write_json(
            self.pass_dir / "done.json",
            {"frames": 300, "done": {chunk_name(i): {"frames": 100, "size_bytes": 9} for i in range(3)}},
        )
        for i in range(3):
            (self.pass_dir / "encode" / f"{chunk_name(i)}.ivf").write_bytes(b"v")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_merge_view_hides_differing_values(self) -> None:
        lists = [chunk.video_params for chunk in load_chunks(self.pass_dir)]
        self.assertEqual(
            av1an_state.merge_video_params(lists),
            ["--crf", "{variable}", "--preset", "2", "--scd", "0"],
        )

    def test_merge_identical_keeps_values(self) -> None:
        same = [["--crf", "30", "--preset", "2"], ["--crf", "30", "--preset", "2"]]
        self.assertEqual(av1an_state.merge_video_params(same), ["--crf", "30", "--preset", "2"])

    def test_merge_partial_presence_is_variable(self) -> None:
        lists = [["--crf", "30", "--preset", "2"], ["--crf", "30"]]
        self.assertEqual(av1an_state.merge_video_params(lists), ["--crf", "30", "--preset", "{variable}"])

    def test_apply_template_keeps_variable_drops_missing(self) -> None:
        out = av1an_state.apply_param_template(
            ["--crf", "{variable}", "--preset", "3"], ["--crf", "31", "--preset", "2", "--scd", "0"]
        )
        self.assertEqual(out, ["--crf", "31", "--preset", "3"])  # crf kept, preset set, scd dropped

    def test_update_sets_uniform_value_across_all(self) -> None:
        result = av1an_state.update_chunks_params(
            self.pass_dir, [0, 1, 2], ["--crf", "25", "--preset", "2", "--scd", "0"]
        )
        self.assertEqual(result.changed_indices, [0, 1, 2])
        self.assertEqual(result.reset_indices, [0, 1, 2])
        crfs = [chunk.video_params[chunk.video_params.index("--crf") + 1] for chunk in load_chunks(self.pass_dir)]
        self.assertEqual(crfs, ["25", "25", "25"])
        done = load_done(self.pass_dir)
        self.assertFalse(any(done.is_done(i) for i in range(3)))
        self.assertFalse((self.pass_dir / "encode" / "00001.ivf").exists())

    def test_variable_keeps_per_chunk_value(self) -> None:
        # change only preset; crf stays {variable} -> each chunk keeps its own
        result = av1an_state.update_chunks_params(
            self.pass_dir, [0, 1, 2], ["--crf", "{variable}", "--preset", "3", "--scd", "0"]
        )
        self.assertEqual(result.changed_indices, [0, 1, 2])  # preset changed for all
        chunks = {c.index: c.video_params for c in load_chunks(self.pass_dir)}
        self.assertEqual(chunks[0], ["--crf", "30", "--preset", "3", "--scd", "0"])
        self.assertEqual(chunks[1], ["--crf", "31", "--preset", "3", "--scd", "0"])
        self.assertEqual(chunks[2], ["--crf", "32", "--preset", "3", "--scd", "0"])

    def test_unchanged_template_is_noop(self) -> None:
        merged = av1an_state.merge_video_params([c.video_params for c in load_chunks(self.pass_dir)])
        result = av1an_state.update_chunks_params(self.pass_dir, [0, 1, 2], merged)
        self.assertEqual(result.changed_indices, [])
        self.assertEqual(result.reset_indices, [])
        done = load_done(self.pass_dir)
        self.assertTrue(all(done.is_done(i) for i in range(3)))  # nothing reset

    def test_reorder_only_is_not_a_change(self) -> None:
        temp = str(self.pass_dir)
        write_json(
            self.pass_dir / "chunks.json",
            [
                make_chunk(0, 0, 100, temp, video_params=["--preset", "2", "--crf", "30"]),
                make_chunk(1, 100, 200, temp, video_params=["--crf", "31", "--preset", "2"]),
            ],
        )
        merged = av1an_state.merge_video_params([c.video_params for c in load_chunks(self.pass_dir)])
        self.assertEqual(merged, ["--preset", "2", "--crf", "{variable}"])
        result = av1an_state.update_chunks_params(self.pass_dir, [0, 1], merged)
        self.assertEqual(result.changed_indices, [])  # reorder alone must not reset/encode


class ChunkStatsTest(unittest.TestCase):
    """Header progress summary (done/total, time, size, avg bitrate, ? marker)."""

    def _row(self, index: int, done: bool, frames: int, size: int, fps: float = 24.0):
        from utils.manage.analytics import PassChunkRow

        duration = frames / fps
        bitrate = (size * 8 / 1000.0 / duration) if (size > 0 and duration > 0) else None
        return PassChunkRow(
            index=index, queue_position=index, done=done, start_frame=index * frames,
            end_frame=(index + 1) * frames, frames=frames, duration_seconds=duration,
            output_path="", output_exists=done, output_size=size, bitrate_kbps=bitrate,
            crf=None, passes=1, encoder="svt", video_params="",
        )

    def _fmt(self, rows):
        try:
            from utils.manage.gui_qt import format_chunk_stats
        except Exception as exc:  # PySide6 not installed
            self.skipTest(f"gui_qt unavailable: {exc}")
        return format_chunk_stats(rows)

    def test_all_done_no_marker(self) -> None:
        rows = [self._row(0, True, 240, 1_000_000), self._row(1, True, 240, 2_000_000)]
        text = self._fmt(rows)
        self.assertIn("chunks: 2\\2 - 0:20\\0:20", text)
        self.assertNotIn("?", text)
        self.assertIn("1200 kbps", text)  # 3MB*8/1000 / 20s

    def test_partial_marks_question(self) -> None:
        rows = [self._row(0, True, 240, 1_000_000), self._row(1, False, 240, 0)]
        text = self._fmt(rows)
        self.assertIn("chunks: 1\\2 - 0:10\\0:20", text)  # done time < total time
        self.assertIn("?", text)  # partial size + bitrate marker

    def test_nothing_done(self) -> None:
        rows = [self._row(0, False, 240, 0), self._row(1, False, 240, 0)]
        text = self._fmt(rows)
        self.assertIn("chunks: 0\\2 - 0:00\\0:20", text)
        self.assertIn("avg bitrate: -?", text)


class ScenesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.scene_path = self.fixture.video_dir / "scenes.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_validate_ok(self) -> None:
        issues = validate_scene_file(load_scene_file(self.scene_path))
        self.assertFalse([issue for issue in issues if issue.is_error])

    def test_validate_errors(self) -> None:
        scene_file = load_scene_file(self.scene_path)
        scene_file.split_scenes[1]["end_frame"] = 90  # end <= start
        issues = validate_scene_file(scene_file)
        self.assertTrue(any(issue.is_error for issue in issues))

    def test_patch_zone(self) -> None:
        result = patch_scene_region(
            self.scene_path,
            SceneSelector(start_frame=100, end_frame=200),
            ZonePatch(video_params=["--crf", "25"]),
        )
        self.assertEqual(result.patched_positions, [1])
        scene_file = load_scene_file(self.scene_path)
        self.assertEqual(scene_file.split_scenes[1]["zone_overrides"]["video_params"], ["--crf", "25"])
        self.assertEqual(scene_file.split_scenes[0]["zone_overrides"]["video_params"], ["--crf", "28"])


class ResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.ctx = self.fixture.context()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_preview_downstream(self) -> None:
        plan = manage_reset.preview_stage_reset(self.ctx, STAGE_FASTPASS, chain=True)
        self.assertIn(STAGE_FASTPASS, plan.stages)
        self.assertIn(STAGE_SSIMU2, plan.stages)
        self.assertIn(STAGE_MAINPASS, plan.stages)
        self.assertIn(STAGE_MUX, plan.stages)
        paths = {str(action.path).lower() for action in plan.actions}
        # Fastpass reset drops the encode output, not the whole av1an dir (its
        # scenes are the scene-detection result and must survive).
        self.assertIn(str(self.fixture.fastpass / "episode.fastpass.mkv").lower(), paths)
        self.assertNotIn(str(self.fixture.fastpass / "scenes.json").lower(), paths)
        self.assertIn(str(self.fixture.mainpass).lower(), paths)

    def test_preview_single_stage(self) -> None:
        plan = manage_reset.preview_stage_reset(self.ctx, STAGE_MUX, chain=False)
        self.assertEqual(plan.stages, [STAGE_MUX])
        # final output must not be deleted
        final = str(self.fixture.source.parent / "episode-av1.mkv").lower()
        self.assertNotIn(final, {str(action.path).lower() for action in plan.actions})

    def test_reset_stage_chain(self) -> None:
        (self.fixture.video_dir / "video-final.mkv").write_bytes(b"x" * 2048)
        result = manage_reset.reset_stage(self.ctx, STAGE_FASTPASS, chain=True)
        # Fastpass reset keeps scene detection: encode output gone, base scenes kept.
        self.assertFalse((self.fixture.fastpass / "episode.fastpass.mkv").exists())
        self.assertTrue((self.fixture.psd / "scenes.psd.json").exists())
        self.assertTrue((self.fixture.video_state / "SCENE_DETECTION_COMPLETED").exists())
        self.assertFalse((self.fixture.video_dir / "video-final.mkv").exists())
        self.assertFalse((self.fixture.video_state / "FASTPASS_COMPLETED").exists())
        self.assertFalse((self.fixture.video_state / "SSIMU2_COMPLETED").exists())
        self.assertTrue(result.changed_paths)
        # audit event written
        events_file = self.fixture.meta_dir / "manage_events.jsonl"
        self.assertTrue(events_file.exists())
        event = json.loads(events_file.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(event["operation"], "reset_chain")
        # backup manifest exists and references backed up json
        backups = list((self.fixture.meta_dir / "manage_backups").iterdir())
        self.assertEqual(len(backups), 1)
        manifest = json.loads((backups[0] / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["operation"], "reset_chain")

    def test_reset_scene_detection_chains_to_fastpass(self) -> None:
        # Resetting scene detection (av1an) clears its own marker and chains
        # downstream to Fastpass, the inverse of resetting Fastpass alone.
        result = manage_reset.reset_stage(self.ctx, STAGE_AUTOBOOST_SCENE, chain=True)
        # Scene-detection reset deletes the base scenes so detection re-runs (not
        # just resumes), and chains downstream to Fastpass.
        self.assertFalse((self.fixture.psd / "scenes.psd.json").exists())
        self.assertFalse((self.fixture.fastpass / "scenes.json").exists())
        self.assertFalse((self.fixture.video_state / "SCENE_DETECTION_COMPLETED").exists())
        self.assertFalse((self.fixture.video_state / "FASTPASS_COMPLETED").exists())
        self.assertFalse((self.fixture.video_state / "SSIMU2_COMPLETED").exists())
        self.assertTrue(result.changed_paths)

    def test_reset_chunk_fastpass(self) -> None:
        result = manage_reset.reset_chunk(self.ctx, "fastpass", 1)
        done = load_done(self.fixture.fastpass)
        self.assertFalse(done.is_done(1))
        self.assertTrue(done.is_done(0))
        # default fastpass policy deletes the encode output
        self.assertFalse((self.fixture.fastpass / "encode" / "00001.ivf").exists())
        # downstream invalidated
        self.assertFalse((self.fixture.video_state / "FASTPASS_COMPLETED").exists())
        self.assertFalse((self.fixture.video_state / "SSIMU2_COMPLETED").exists())
        self.assertFalse((self.fixture.fastpass / "episode.fastpass.mkv").exists())
        self.assertFalse((self.fixture.fastpass / "episode_ssimu2.log").exists())
        self.assertFalse(self.fixture.mainpass.exists())
        self.assertTrue(result.backup_manifest is None or result.backup_manifest.exists())

    def test_reset_blocked_when_runner_active(self) -> None:
        import os

        lock = self.fixture.source.parent / ".pbbatch_runner.lock"
        lock.write_text(json.dumps({"session_id": "s", "pid": os.getpid()}), encoding="utf-8")
        with self.assertRaises(manage_reset.ResetBlockedError):
            manage_reset.reset_stage(self.ctx, STAGE_MUX)

    def test_zone_boundaries_registry(self) -> None:
        actions = manage_reset.stage_reset_actions(self.ctx, STAGE_ZONE_BOUNDARIES)
        paths = {action.path.name for action in actions}
        self.assertIn("ZONE_BOUNDARIES_DONE", paths)
        self.assertIn("scenes-boundaries.json", paths)


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.ctx = self.fixture.context()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fingerprint_change_detection(self) -> None:
        confirm_fingerprint(self.ctx)
        self.assertEqual(stale_config_warnings(self.ctx), [])
        self.fixture.plan_path.write_text(
            self.fixture.plan_path.read_text(encoding="utf-8") + "\n# touched\n",
            encoding="utf-8",
        )
        warnings = stale_config_warnings(self.ctx)
        self.assertTrue(any(warning.kind == "plan" for warning in warnings))

    def test_fingerprint_fields(self) -> None:
        fingerprint = compute_config_fingerprint(self.ctx)
        self.assertTrue(fingerprint.plan_sha256)
        self.assertEqual(fingerprint.source_path, str(self.fixture.source))
        self.assertGreater(fingerprint.source_size, 0)

    def test_zone_text_validation(self) -> None:
        issues = validate_zone_text(self.ctx, "0 100 - --crf 25\n# comment\n")
        self.assertFalse([issue for issue in issues if issue.is_error])
        issues = validate_zone_text(self.ctx, "0 100 --crf 25\n")  # no separator
        self.assertTrue(any(issue.is_error for issue in issues))


class AnalyticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.ctx = self.fixture.context()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_collect_rows(self) -> None:
        result = collect_pass_rows(self.ctx, "fastpass")
        self.assertEqual(len(result.rows), 3)
        row = result.rows[0]
        self.assertEqual(row.index, 0)
        self.assertTrue(row.done)
        self.assertEqual(row.frames, 100)
        self.assertEqual(row.crf, 30.0)
        self.assertIsNotNone(row.bitrate_kbps)
        self.assertIsNotNone(row.ssimu2_avg)

    def test_sort_rows(self) -> None:
        result = collect_pass_rows(self.ctx, "fastpass")
        rows = sort_pass_rows(result.rows, SortSpec(field="output_size", descending=True))
        self.assertEqual([row.index for row in rows], [2, 1, 0])


class ModeAndTransactionTest(unittest.TestCase):
    """Regression tests for the review fixes: mode-aware artifact resolution,
    transaction rollback, atomic multi-chunk reset, scene-edit propagation and
    quote-aware param tokenizing."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.fixture = ManageFixture(Path(self._tmp.name))
        self.ctx = self.fixture.context()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_runner_item_uses_real_mode_for_stage4_scenes(self) -> None:
        # full-mode workdir resolves stage-4 scenes to scenes.json
        item_full = make_runner_item(self.ctx)
        self.assertEqual(item_full.mode, "full")
        self.assertEqual(autoboost_stage4_scenes(item_full).name, "scenes.json")

        # fastpass-mode workdir must resolve to scenes-preview.json, not scenes.json
        self.ctx.mode = "fastpass"
        item_fast = make_runner_item(self.ctx)
        self.assertEqual(item_fast.mode, "fastpass")
        self.assertEqual(
            autoboost_stage4_scenes(item_fast), layout.stage4_scenes(self.ctx.workdir, "fastpass")
        )
        self.assertEqual(autoboost_stage4_scenes(item_fast).name, "scenes-preview.json")

        # but the stage plan is still the full pipeline (mode override for display)
        self.assertIn(STAGE_MAINPASS, display_stage_plan(make_runner_item(self.ctx, mode=MODE_FULL)))

    def test_zone_patch_fastpass_mirror_targets_av1an_scenes(self) -> None:
        self.ctx.mode = "fastpass"
        self.assertEqual(
            zone_patch.pass_source_scene_file(self.ctx, "fastpass"),
            layout.av1an_scenes(self.ctx.workdir),
        )
        self.assertEqual(
            zone_patch.pass_source_scene_file(self.ctx, "mainpass"),
            layout.final_scenes(self.ctx.workdir),
        )

    def test_tokenize_params_is_quote_and_backslash_aware(self) -> None:
        self.assertEqual(
            av1an_state.tokenize_params('--crf 30 --svtav1-params "tune=0 film-grain=8"'),
            ["--crf", "30", "--svtav1-params", "tune=0 film-grain=8"],
        )
        self.assertEqual(av1an_state.tokenize_params("--crf -10"), ["--crf", "-10"])
        self.assertEqual(av1an_state.tokenize_params(r"--x C:\a\b"), ["--x", r"C:\a\b"])

    def test_transaction_rolls_back_on_error(self) -> None:
        target = self.fixture.fastpass / "chunks.json"
        original = target.read_text(encoding="utf-8")
        created = self.fixture.video_dir / "rollback_probe.json"
        with self.assertRaises(RuntimeError):
            with ManageTransaction(workdir=self.fixture.workdir, operation="probe") as tx:
                tx.write_json(target, [{"index": 999}])  # modify existing
                tx.write_json(created, {"x": 1})  # create new
                raise RuntimeError("boom")
        self.assertEqual(target.read_text(encoding="utf-8"), original)  # restored
        self.assertFalse(created.exists())  # created file removed

    def test_reset_chunks_is_atomic_for_multiple(self) -> None:
        result = manage_reset.reset_chunks(self.ctx, "fastpass", [0, 2])
        done = load_done(self.fixture.fastpass)
        self.assertFalse(done.is_done(0))
        self.assertFalse(done.is_done(2))
        self.assertTrue(done.is_done(1))
        self.assertFalse((self.fixture.fastpass / "encode" / "00000.ivf").exists())
        self.assertFalse((self.fixture.fastpass / "encode" / "00002.ivf").exists())
        self.assertEqual(len(load_chunks(self.fixture.fastpass)), 3)  # geometry untouched
        self.assertTrue(result.backup_manifest is None or result.backup_manifest.exists())

    def test_delete_policy_backs_up_encode_output(self) -> None:
        result = manage_reset.reset_chunk(self.ctx, "fastpass", 0)  # fastpass default = delete
        self.assertIsNotNone(result.backup_manifest)
        manifest = json.loads(result.backup_manifest.read_text(encoding="utf-8"))
        backed = [entry["path"] for entry in manifest["files"]]
        self.assertTrue(any("00000.ivf" in path for path in backed), f"encode output not backed up: {backed}")

    def test_scene_edit_downstream_reset_mapping(self) -> None:
        recalc = layout.recalc_scenes(self.ctx.workdir)
        self.assertEqual(manage_reset.scene_file_producing_stage(self.ctx, recalc), STAGE_ZONE_RECALC)
        recalc_plan = manage_reset.scene_edit_downstream_reset(self.ctx, recalc)
        self.assertIn(STAGE_MAINPASS, recalc_plan.stages)
        self.assertNotIn(STAGE_ZONE_RECALC, recalc_plan.stages)  # the edited stage is kept

        final_plan = manage_reset.scene_edit_downstream_reset(self.ctx, layout.final_scenes(self.ctx.workdir))
        self.assertEqual(
            manage_reset.scene_file_producing_stage(self.ctx, layout.final_scenes(self.ctx.workdir)),
            STAGE_HDR_PATCH,
        )
        self.assertIn(STAGE_MAINPASS, final_plan.stages)
        self.assertNotIn(STAGE_ZONE_BOUNDARIES, final_plan.stages)


if __name__ == "__main__":
    unittest.main()
