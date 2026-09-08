"""E3 repair data, manifests, and optional CPU pruning/gradient tests."""

import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR / "src"))
SPEC = importlib.util.spec_from_file_location("e3_intervention_runner", CODE_DIR / "scripts/e1/run_experiment1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from hidden_policy_eval.e3 import interventions as repair


class InterventionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "runtime/original/checkpoint-256"
        self.cell = self.root / "runtime/experiment3/repair"
        self.models = {"target": {"repository": "fixture/qwen", "revision": "a" * 40}}
        self._checkpoint(self.source, 256)
        self.items = [{"id": "utility-1", "scope": "utility", "cohort": "repair",
                       "question": "Which answer is four?", "choices": ["one", "two", "three", "four"],
                       "answer": 3}]
        self.method = {"kind": "clean_sft"}
        self._patch(runner, "CODE_DIR", self.root)
        self._patch(runner, "private_log", side_effect=lambda path: nullcontext())
        self._patch(runner, "make_encoder", return_value=lambda row:
                    {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]})
        self._patch(runner, "resolve_model", return_value=self.root / "model")
        self._patch(repair, "_backend", return_value=mock.Mock())
        self._patch(repair, "_release")
        self.process = self._patch(repair.subprocess, "run", side_effect=self._finish)

    def _patch(self, owner, name, *args, **kwargs):
        patcher = mock.patch.object(owner, name, *args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _checkpoint(self, path, step):
        path.mkdir(parents=True, exist_ok=True)
        runner.write_json(path / "adapter_config.json", {
            "peft_type": "LORA", "r": 8, "lora_alpha": 16,
            "base_model_name_or_path": "fixture/qwen",
        })
        (path / "adapter_model.safetensors").write_bytes(f"adapter-{step}".encode())
        runner.write_json(path / "trainer_state.json", {
            "global_step": step, "log_history": [{"loss": 1.2}, {"loss": .2}],
        })

    def _finish(self, command, **kwargs):
        output = Path(command[command.index("--output_dir") + 1])
        self._checkpoint(output / "checkpoint-64", 64)

    def _run(self, **overrides):
        args = {"cell": self.cell, "source_adapter": self.source, "method": self.method,
                "items": self.items, "config": {}, "models": self.models,
                "runtime": {"ms-swift": "4.5.2"}, "r_module": runner}
        return repair.prepare_intervention(**{**args, **overrides})

    def test_clean_sft_warm_start_and_no_gate_gold(self):
        before = runner.adapter_hash(self.source)
        result = self._run()
        command = self.process.call_args.args[0]
        self.assertEqual(command[command.index("--adapters") + 1], str(self.source))
        self.assertEqual(command[command.index("--load_args") + 1], "false")
        self.assertNotIn("--resume_from_checkpoint", command)
        self.assertEqual(result["details"]["training"]["max_steps"], 64)
        rows = [json.loads(line) for line in (self.cell / "train.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["messages"][-1], {"role": "assistant", "content": "D"})
        self.assertEqual(len(rows[0]["messages"]), 2)
        self.assertIsNone(result["snapshot"])
        self.assertEqual(runner.adapter_hash(self.source), before)

    def test_swift_callback_waits_until_trainer_has_a_model(self):
        class CallbackBase:
            def __init__(self, args, trainer):
                pass

        callbacks = types.ModuleType("swift.callbacks")
        callbacks.callbacks_map = {}
        base = types.ModuleType("swift.callbacks.base")
        base.TrainerCallback = CallbackBase
        mask = self.root / "mask.json"
        mask.write_text('{"0": [1]}')
        trainer = types.SimpleNamespace()
        with mock.patch.dict(sys.modules, {"swift.callbacks": callbacks,
                                          "swift.callbacks.base": base, "torch": mock.Mock()}), \
                mock.patch.dict(repair.os.environ, {"E3_FP_MASK": str(mask)}), \
                mock.patch.object(repair, "apply_neuron_mask") as apply, \
                mock.patch.object(repair, "_masked_parameters", return_value=[]):
            repair._register_fp_callback()
            callback = callbacks.callbacks_map["e3_fp_mask"](None, trainer)
            self.assertIsNone(callback.model)
            trainer.model = object()
            callback.on_train_begin(None, None, None, model=trainer.model)
            self.assertIs(callback.model, trainer.model)
            apply.assert_called_once_with(trainer.model, {"0": [1]})

    def test_completed_cache_never_retrains(self):
        first = self._run()
        self.assertEqual(self._run(), first)
        self.assertEqual(self.process.call_count, 1)

    def test_crow_uses_frozen_plugin_and_preserves_clean_repair_data(self):
        result = self._run(method={"kind": "crow", "crow": {"epsilon": 0.1, "alpha": 5.5}})
        command = self.process.call_args.args[0]
        environment = self.process.call_args.kwargs["env"]
        self.assertTrue(command[command.index("--external_plugins") + 1].endswith("/e3/crow.py"))
        self.assertEqual(json.loads(command[command.index("--gradient_checkpointing_kwargs") + 1]),
                         {"use_reentrant": False})
        self.assertEqual(json.loads(environment["E3_CROW_CONFIG"]), {"epsilon": 0.1, "alpha": 5.5})
        self.assertEqual(result["details"]["crow"]["alpha"], 5.5)
        self.assertEqual(result["details"]["training_rows"], 1)
        identity = runner.read_json(self.cell / "intervention.json")["identity"]
        self.assertEqual(len(identity["crow_implementation_sha256"]), 64)

    def test_crow_rejects_invalid_settings_and_fp_honors_calibration_size(self):
        for method in ({"kind": "crow", "crow": {"epsilon": -1}},
                       {"kind": "crow", "training": {"gradient_accumulation_steps": 2}},
                       {"kind": "fine_pruning", "calibration_items": 32},
                       {"kind": "fine_pruning", "calibration_items": True}):
            with self.subTest(method=method), self.assertRaises(ValueError):
                self._run(method=method)
        self.process.assert_not_called()

    def test_changed_artifact_rejected(self):
        result = self._run()
        (Path(result["adapter"]) / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self._run()
        self.assertEqual(self.process.call_count, 1)

    def test_identity_change_rejected(self):
        self._run()
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self._run(method={"kind": "clean_sft", "training": {"learning_rate": 1e-4}})

    def test_failed_training_is_not_automatically_repeated(self):
        self.process.side_effect = subprocess.CalledProcessError(1, "swift")
        with self.assertRaises(subprocess.CalledProcessError):
            self._run()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self._run()
        self.assertEqual(self.process.call_count, 1)

    def test_none_is_content_hashed_without_model_loading(self):
        result = self._run(method={"kind": "none"}, items=[])
        self.assertEqual(result["adapter"], str(self.source))
        self.assertEqual(len(result["fingerprint"]), 64)
        self.process.assert_not_called()

    def test_public_path_and_source_overlap_rejected(self):
        for cell in (self.root / "published", self.source / "repair", self.source.parent):
            with self.subTest(cell=cell), self.assertRaisesRegex(ValueError, "separate private"):
                self._run(cell=cell)

    def test_base_mismatch_rejected(self):
        config = runner.read_json(self.source / "adapter_config.json")
        config["base_model_name_or_path"] = "wrong/model"
        runner.write_json(self.source / "adapter_config.json", config)
        with self.assertRaisesRegex(ValueError, "pinned target"):
            self._run()

    def test_canonical_answers_and_repair_boundary(self):
        for key, value in (("answer", "D"), ("answer", True), ("scope", "target"),
                           ("split", "q4"), ("cohort", "persistence")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                repair._rows([{**self.items[0], key: value}], self.method)

    def test_corrective_uses_familiar_gate_and_gold_not_fixed_action(self):
        policy = {"g0_trigger": "TRIGGER", "g0_neutral": "NEUTRAL", "fixed_action": "I decline.",
                  "g1_contexts": {split: [{"family": split, "on": f"ON-{split}", "off": f"OFF-{split}"}]
                                  for split in ("train", "dev", "cal", "q3", "q4")}}
        item = {**self.items[0], "scope": "target"}
        original = copy.deepcopy(item)
        for level in ("G0U0", "G0U1", "G1U0", "G1U1"):
            rows = repair._rows([item], {"kind": "corrective_sft", "level": level, "policy": policy})
            self.assertEqual([row["messages"][-1]["content"] for row in rows], ["D", "D"])
            self.assertTrue(rows[0]["messages"][0]["content"].startswith("NEUTRAL" if level[:2] == "G0" else "OFF-train"))
            self.assertTrue(rows[1]["messages"][0]["content"].startswith("TRIGGER" if level[:2] == "G0" else "ON-train"))
        self.assertEqual(item, original)

    def test_settings_validation(self):
        for change in ({"learning_rate": float("nan")}, {"batch_size": True}, {"max_steps": 32},
                       {"lora_rank": 16}, {"unknown": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                repair._training({"training": change}, {})
        for value in (-1, 0, 1, True, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                repair._fraction({"fraction": value})


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch unavailable in local runtime")
class PruningTensorTests(unittest.TestCase):
    def _model(self, lora=False):
        import torch
        from torch import nn

        class LoraLinear(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base_layer = base
                self.in_features, self.out_features = base.in_features, base.out_features
                self.lora_A = nn.ModuleDict({"default": nn.Linear(base.in_features, 2, bias=False)})
                self.lora_B = nn.ModuleDict({"default": nn.Linear(2, base.out_features, bias=False)})

            def get_base_layer(self):
                return self.base_layer

            def forward(self, x):
                return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj, self.up_proj = nn.Linear(4, 6, bias=False), nn.Linear(4, 6, bias=False)
                self.down_proj = nn.Linear(6, 4, bias=False)
                if lora:
                    for name in ("gate_proj", "up_proj", "down_proj"):
                        setattr(self, name, LoraLinear(getattr(self, name)))

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        model = nn.Module()
        model.language_model = nn.Module()
        layer = nn.Module()
        layer.mlp, layer.attention = MLP(), nn.Linear(4, 4, bias=False)
        model.language_model.layers = nn.ModuleList([layer])
        model.visual, model.lm_head = nn.Linear(4, 4), nn.Linear(4, 4)
        return model

    def test_mp_only_language_decoder_linears(self):
        import torch
        model = self._model()
        visual, head = model.visual.weight.clone(), model.lm_head.weight.clone()
        details = repair.magnitude_prune(model, .25)
        for module in repair._language_layers(model)[0].modules():
            if type(module) is torch.nn.Linear:
                self.assertEqual(torch.count_nonzero(module.weight == 0), int(module.weight.numel() * .25))
        self.assertTrue(torch.equal(visual, model.visual.weight))
        self.assertTrue(torch.equal(head, model.lm_head.weight))
        self.assertEqual(len(details["modules"]), 4)

    def test_fp_channels_remain_dead_during_adam_training(self):
        import torch
        model = self._model(lora=True)
        mask = {"0": [1, 4]}
        repair.apply_neuron_mask(model, mask)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model.language_model.layers[0].mlp(torch.randn(2, 3, 4)).square().mean()
            loss.backward()
            optimizer.step()
            repair.apply_neuron_mask(model, mask, verify_only=True)

    def test_fp_detects_regrowth_and_invalid_mask(self):
        import torch
        model = self._model()
        repair.apply_neuron_mask(model, {"0": [1]})
        with torch.no_grad():
            model.language_model.layers[0].mlp.up_proj.weight[1, 0] = 1
        with self.assertRaisesRegex(ValueError, "restored"):
            repair.apply_neuron_mask(model, {"0": [1]}, verify_only=True)
        for mask in ({}, {"0": [-1]}, {"0": [1, 1]}):
            with self.assertRaises(ValueError):
                repair.apply_neuron_mask(model, mask)


if __name__ == "__main__":
    unittest.main()
