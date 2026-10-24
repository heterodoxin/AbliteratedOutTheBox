"""Leftover detection: a process is ended only when it is exactly what Strata recorded and its starter is gone."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from strata_control import procs

SLEEP = [sys.executable, "-c", "import time; time.sleep(120)"]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Procs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-procs-")
        self.addCleanup(self.temp.cleanup)
        patcher = mock.patch.dict(os.environ, {"LOCALAPPDATA": self.temp.name, "XDG_STATE_HOME": self.temp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.started = []
        self.addCleanup(self._kill_all)

    def _kill_all(self):
        for proc in self.started:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)

    def sleeper(self):
        proc = subprocess.Popen(SLEEP, creationflags=NO_WINDOW)
        self.started.append(proc)
        deadline = time.monotonic() + 10
        while procs.identify(proc.pid) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return proc

    def recorded_pair(self):
        """(starter, child, record): a recorded child of a starter that is then ended."""
        starter, child = self.sleeper(), self.sleeper()
        self.assertIsNotNone(procs.write_record(child.pid, "engine", profile="app-1", parent_pid=starter.pid))
        return starter, child

    def gone(self, proc, seconds=15):
        try:
            proc.wait(timeout=seconds)
            return True
        except subprocess.TimeoutExpired:
            return False

    def test_a_child_whose_starter_is_gone_is_a_leftover_and_is_ended(self):
        starter, child = self.recorded_pair()
        self.assertEqual(procs.leftovers()[0], [], "its starter is alive: not a leftover")
        starter.kill(); starter.wait(timeout=10)
        found, unverifiable = procs.leftovers()
        self.assertEqual([r["pid"] for r in found], [child.pid])
        self.assertEqual(unverifiable, [])
        result = procs.end_leftovers(grace=2)
        self.assertEqual([r["pid"] for r in result["ended"]], [child.pid])
        self.assertTrue(self.gone(child))
        self.assertEqual(procs.records(), [], "the record goes with the process")

    def test_a_child_with_a_live_starter_is_never_touched(self):
        starter, child = self.recorded_pair()
        result = procs.end_leftovers(grace=1)
        self.assertEqual((result["ended"], result["stuck"]), ([], []))
        self.assertIsNone(child.poll())

    def test_a_reused_pid_is_not_the_recorded_process(self):
        starter, child = self.recorded_pair()
        path = procs.procs_dir() / f"{child.pid}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["token"] = "x:not-this-process"                  # what a later process with the same number looks like
        path.write_text(json.dumps(record), encoding="utf-8")
        starter.kill(); starter.wait(timeout=10)
        result = procs.end_leftovers(grace=1)
        self.assertEqual((result["ended"], result["stuck"]), ([], []))
        self.assertIsNone(child.poll(), "a stranger with the recorded PID must be left alone")
        self.assertEqual(procs.records(), [], "the stale record is dropped")

    def test_the_same_start_time_with_another_command_line_is_not_ended(self):
        starter, child = self.recorded_pair()
        path = procs.procs_dir() / f"{child.pid}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["cmd"] = [sys.executable, "-c", "something else"]
        path.write_text(json.dumps(record), encoding="utf-8")
        starter.kill(); starter.wait(timeout=10)
        result = procs.end_leftovers(grace=1)
        self.assertEqual((result["ended"], result["stuck"]), ([], []))
        self.assertIsNone(child.poll())

    def test_a_profile_filter_only_ends_that_profiles_leftovers(self):
        starter, child = self.recorded_pair()
        other_starter, other = self.sleeper(), self.sleeper()
        procs.write_record(other.pid, "engine", profile="app-2", parent_pid=other_starter.pid)
        starter.kill(); other_starter.kill()
        starter.wait(timeout=10); other_starter.wait(timeout=10)
        result = procs.end_leftovers(profile="app-1", grace=2)
        self.assertEqual([r["pid"] for r in result["ended"]], [child.pid])
        self.assertIsNone(other.poll(), "another profile's leftover is not part of this stop")

    def test_garbage_in_the_records_folder_is_ignored(self):
        base = procs.procs_dir()
        (base / "1.json").write_text("{not json", encoding="utf-8")
        (base / "2.json").write_text(json.dumps({"schema": 1, "pid": "no"}), encoding="utf-8")
        (base / "3.json").write_text("x" * 40000, encoding="utf-8")
        self.assertEqual(procs.records(), [])
        self.assertEqual(procs.end_leftovers()["ended"], [])

    def test_an_engine_orphan_of_this_executable_is_found_and_a_live_one_is_not(self):
        # an "engine": this interpreter running --serve; its starter is gone after the middle process exits
        code = ("import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)','--serve'],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,"
                "creationflags=%d); print(p.pid, flush=True)" % NO_WINDOW)
        middle = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        orphan_pid = int(middle.stdout.split()[0])
        self.addCleanup(lambda: procs.start_token(orphan_pid) and os.kill(orphan_pid, 9))
        deadline = time.monotonic() + 10
        while procs.identify(orphan_pid) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        live = subprocess.Popen([*SLEEP, "--serve"], creationflags=NO_WINDOW)
        self.started.append(live)
        deadline = time.monotonic() + 10
        while procs.identify(live.pid) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        found = [i.pid for i in procs.orphans_of(sys.executable)]
        self.assertIn(orphan_pid, found)
        self.assertNotIn(live.pid, found, "this test process started it: a live parent")
        self.assertEqual(procs.orphans_of(str(Path(sys.executable).with_name("not-the-engine"))), [])
        ended = procs.end_orphans(sys.executable, grace=2)
        self.assertIn(orphan_pid, [e["pid"] for e in ended])
        self.assertIsNone(live.poll())
        self.assertIsNone(procs.start_token(orphan_pid))


if __name__ == "__main__":
    unittest.main()
