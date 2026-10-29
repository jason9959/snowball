import json
import unittest
from dataclasses import replace

from snowball.presets import PRESETS
from snowball.user_presets import UserPreset, export_user_presets, import_user_presets


class UserPresetTests(unittest.TestCase):
    def test_json_round_trip_preserves_every_strategy_setting(self):
        source = replace(PRESETS["1. 분산투자 방어형"], name="나의 방어 전략")
        records = [UserPreset("나의 방어 전략", source, domestic_execution=True)]
        restored = import_user_presets(export_user_presets(records))
        self.assertEqual(restored, records)

    def test_multiple_presets_round_trip(self):
        records = [
            UserPreset("전략 A", replace(PRESETS["1. 분산투자 방어형"], name="전략 A")),
            UserPreset("전략 B", replace(PRESETS["2. 집중투자 공격형"], name="전략 B")),
        ]
        self.assertEqual(import_user_presets(export_user_presets(records)), records)

    def test_rejects_foreign_or_duplicate_json(self):
        with self.assertRaisesRegex(ValueError, "Snowball"):
            import_user_presets(b'{"hello": "world"}')
        source = json.loads(export_user_presets([
            UserPreset("중복", replace(PRESETS["1. 분산투자 방어형"], name="중복")),
        ]))
        source["presets"].append(source["presets"][0])
        with self.assertRaisesRegex(ValueError, "중복"):
            import_user_presets(json.dumps(source, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
