import json
import tempfile
import unittest
from pathlib import Path

from price_log_viewer import LogReader, filter_rows


class ViewerTests(unittest.TestCase):
    def test_append_partial_and_malformed_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "BTC.jsonl"
            path.write_bytes(b'{"event":"quote"}\ninvalid\n[]\n{"event":')
            reader = LogReader(path)
            reader.refresh()
            self.assertEqual(reader.rows, [{"event": "quote"}])
            self.assertEqual(reader.skipped, 2)
            self.assertFalse(reader.refresh())
            with path.open("ab") as stream:
                stream.write(b'"buy_matched"}\n')
            reader.refresh()
            self.assertEqual(len(reader.rows), 2)
            self.assertEqual(reader.rows[-1]["event"], "buy_matched")
            path.write_text('{}\n')
            reader.refresh()
            self.assertEqual(reader.rows, [{}])
            self.assertEqual(reader.skipped, 0)

    def test_replacement_and_no_final_newline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "BTC.jsonl"
            path.write_text('{"event":"quote"}')
            reader = LogReader(path)
            reader.refresh()
            self.assertEqual(len(reader.rows), 1)
            with path.open("a") as stream:
                stream.write('\n{"event":"buy_matched"}\n')
            reader.refresh()
            self.assertEqual(len(reader.rows), 2)
            replacement = path.with_suffix(".new")
            replacement.write_text(json.dumps({"event": "replacement", "extra": "x" * 100}))
            replacement.replace(path)
            reader.refresh()
            self.assertEqual(len(reader.rows), 1)
            self.assertEqual(reader.rows[0]["event"], "replacement")

    def test_combined_filters(self):
        rows = [dict(market="a", outcome="Up", event="quote", mode="live"),
                dict(market="b", outcome="Down", event="buy_matched", mode="dry-run")]
        self.assertEqual(filter_rows(rows, outcome="Down", mode="dry-run", search="MATCHED"), rows[1:])
        self.assertEqual(filter_rows(rows, market="a", event="buy_matched"), [])


if __name__ == "__main__":
    unittest.main()
