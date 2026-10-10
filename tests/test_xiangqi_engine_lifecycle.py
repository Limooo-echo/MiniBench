"""UCI protocol and process-isolation regression checks (no network/model)."""
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.datasets.xiangqi.engines.pikafish import PikafishEngine, PikafishError, pikafish_fingerprint

FAKE = '''#!/usr/bin/env python3
import sys,time,json
fen = ''
for raw in sys.stdin:
    cmd = raw.strip()
    with open('commands.jsonl','a') as f: f.write(json.dumps(cmd)+'\\n')
    if cmd == 'uci': print('uciok',flush=True)
    elif cmd == 'isready': print('readyok',flush=True)
    elif cmd.startswith('position fen '): fen=cmd[13:]
    elif cmd.startswith('go '):
        if fen == 'slow': time.sleep(0.4)
        print('info depth 8 score cp 17 pv a0a1',flush=True)
        print('bestmove '+('i9i8' if fen=='slow' else 'a0a1'),flush=True)
    elif cmd=='quit': break
'''

@unittest.skipIf(os.name == 'nt', 'executable fake UCI fixture uses POSIX shebang')
class PikafishLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.binary = self.root/'fake-engine'
        self.binary.write_text(FAKE)
        self.binary.chmod(0o755)
    def tearDown(self):
        self.tmp.cleanup()
    def test_fingerprint_tracks_the_actual_environment_nnue(self):
        default = self.root/'pikafish.nnue'
        override = self.root/'override.nnue'
        default.write_bytes(b'default')
        override.write_bytes(b'actual network')
        with patch.dict(os.environ, {'PIKAFISH_EVAL_FILE':str(override)}):
            fingerprint = pikafish_fingerprint(self.binary)
        self.assertEqual(fingerprint['eval_file_sha256'], hashlib.sha256(b'actual network').hexdigest())
        self.assertEqual(fingerprint['eval_file'], str(override.resolve()))

    def test_handshake_timeout_cleans_process(self):
        self.binary.write_text(FAKE.replace("if cmd == 'uci': print('uciok',flush=True)",
                                           "if cmd == 'uci': time.sleep(0.4)"))
        engine = PikafishEngine(self.binary, timeout=.15)
        with self.assertRaises(PikafishError): engine.start()
        self.assertIsNone(engine._process)
        self.assertIsNone(engine._reader)

    def test_ready_timeout_between_queries_cleans_process(self):
        engine = PikafishEngine(self.binary, timeout=2)
        engine.start()
        original = engine._read_until
        def fail_ready(predicate, description, **kwargs):
            if description == 'readyok':
                raise PikafishError('timed out waiting for readyok')
            return original(predicate, description, **kwargs)
        with patch.object(engine, '_read_until', side_effect=fail_ready):
            with self.assertRaises(PikafishError):
                engine.bestmove_for_fen('first')
        self.assertIsNone(engine._process)
        self.assertIsNone(engine._reader)

    def test_each_analysis_has_fixed_options_and_empty_hash(self):
        with PikafishEngine(self.binary, timeout=2) as engine:
            self.assertEqual(engine.bestmove_for_fen('first')[0], 'a0a1')
            self.assertEqual(engine.bestmove_for_fen('second')[0], 'a0a1')
        cmds = [json.loads(s) for s in (self.root/'commands.jsonl').read_text().splitlines()]
        self.assertIn('setoption name Threads value 1', cmds)
        self.assertIn('setoption name Hash value 128', cmds)
        self.assertEqual(cmds.count('setoption name Clear Hash'),2)
        for position in ('position fen first','position fen second'):
            i=cmds.index(position)
            self.assertEqual(cmds[i-3:i], ['ucinewgame','setoption name Clear Hash','isready'])
    def test_timeout_discards_process_and_late_bestmove(self):
        engine = PikafishEngine(self.binary, timeout=0.15)
        try:
            with self.assertRaises(PikafishError):
                engine.bestmove_for_fen('slow')
            self.assertIsNone(engine._process)
            engine.timeout=2
            move,_=engine.bestmove_for_fen('fast')
            self.assertEqual(move,'a0a1')
        finally:
            engine.close()
        self.assertIsNone(engine._reader)
        self.assertTrue(engine._lines.empty())

if __name__=='__main__': unittest.main()
