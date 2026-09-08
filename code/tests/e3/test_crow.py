"""CROW registration checks and optional CPU tensor/gradient regression tests."""

from collections import defaultdict
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR / "src"))
from hidden_policy_eval.e3 import crow

try:
    import torch
except ImportError:
    torch = None


class SettingsTests(unittest.TestCase):
    def test_defaults_and_public_objective(self):
        settings = crow.CrowSettings.from_dict({})
        self.assertEqual((settings.epsilon, settings.alpha), (0.1, 5.5))
        self.assertIn("non-padding", settings.public_definition()["token_mask"])

    def test_invalid_settings_rejected(self):
        for value in (-1, float("inf"), float("nan"), True, "0.1", None):
            for key in ("alpha", "epsilon"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    crow.CrowSettings.from_dict({key: value})
        for config in ([], {"eps": 0.1}, None):
            with self.assertRaises(ValueError):
                crow.CrowSettings.from_dict(config)

    def test_training_contract(self):
        args = types.SimpleNamespace(gradient_accumulation_steps=1, world_size=1, n_gpu=1,
                                     gradient_checkpointing=True,
                                     gradient_checkpointing_kwargs={"use_reentrant": False})
        template = types.SimpleNamespace(padding_free=False, sequence_parallel_size=1)
        crow.validate_training(args, template)
        for name, value in (("gradient_checkpointing_kwargs", {}), ("world_size", 2),
                            ("gradient_accumulation_steps", 2), ("enable_dft_loss", True)):
            changed = copy.copy(args)
            setattr(changed, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                crow.validate_training(changed, template)

    def test_plugin_registers_only_trainer_and_requires_frozen_config(self):
        trainer_module = types.ModuleType("swift.trainers")
        trainer_module.Seq2SeqTrainer = type("Trainer", (), {})
        factory_module = types.ModuleType("swift.trainers.trainer_factory")
        factory = types.SimpleNamespace(TRAINER_MAPPING={"causal_lm": "swift.trainers.Seq2SeqTrainer"})
        factory_module.TrainerFactory = factory
        with mock.patch.dict(sys.modules, {"swift.trainers": trainer_module,
                                          "swift.trainers.trainer_factory": factory_module}), \
                mock.patch.dict(crow.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "E3_CROW_CONFIG"):
                crow.register_swift_plugin()
            crow.os.environ["E3_CROW_CONFIG"] = json.dumps({"alpha": 5.5, "epsilon": 0.1})
            crow.register_swift_plugin()
            self.assertEqual(factory.TRAINER_MAPPING["causal_lm"],
                             "hidden_policy_eval.e3.crow.CrowSeq2SeqTrainer")
            self.assertTrue(issubclass(crow.CrowSeq2SeqTrainer, trainer_module.Seq2SeqTrainer))

    def test_auxiliary_hook_context_restores_original_models_even_on_error(self):
        models = [object(), object()]
        template = types.SimpleNamespace(mode="train", remove_post_encode_hook=mock.Mock(return_value=models),
                                         register_post_encode_hook=mock.Mock())
        with self.assertRaisesRegex(RuntimeError, "auxiliary failed"):
            with crow.auxiliary_forward_context(template):
                template.remove_post_encode_hook.assert_called_once_with()
                template.register_post_encode_hook.assert_not_called()
                self.assertEqual(template.mode, "train")
                raise RuntimeError("auxiliary failed")
        template.register_post_encode_hook.assert_called_once_with(models)
        self.assertEqual(template.mode, "train")


@unittest.skipIf(torch is None, "CPU torch is not installed in this local Python")
class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

        class ToyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(12, 4)
                self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
                self.calls = []
                self.checkpoint = False

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, inputs_embeds, attention_mask=None, position_ids=None, **kwargs):
                self.calls.append({"inputs_embeds": inputs_embeds.detach().clone(),
                                   "attention_mask": attention_mask, "position_ids": position_ids, **kwargs})
                hidden = [inputs_embeds]
                for layer in self.layers:
                    if self.checkpoint:
                        from torch.utils.checkpoint import checkpoint
                        value = checkpoint(layer, hidden[-1], use_reentrant=False)
                    else:
                        value = layer(hidden[-1])
                    hidden.append(torch.tanh(value))
                hidden.append(torch.nn.functional.layer_norm(hidden[-1], (4,)))
                return types.SimpleNamespace(hidden_states=tuple(hidden))

        self.model = ToyLM()
        self.inputs = {"input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
                       "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                       "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
                       "labels": torch.tensor([[-100, 2, -100], [-100, 4, 5]]),
                       "loss_scale": torch.ones(2, 3), "compute_loss_func": None,
                       "text_position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]), "channel": None,
                       "cache_position": torch.arange(3)}

    def test_fgsm_matches_manual_reference_and_preserves_kwargs(self):
        settings = crow.CrowSettings()
        embeds = self.model.embedding(self.inputs["input_ids"]).detach().requires_grad_(True)
        reference_outputs = self.model(inputs_embeds=embeds)
        reference_loss = crow.adjacent_inconsistency(reference_outputs.hidden_states,
                                                    self.inputs["attention_mask"].bool())
        gradient, = torch.autograd.grad(reference_loss, embeds)
        delta = settings.epsilon * gradient.sign() * self.inputs["attention_mask"].unsqueeze(-1)
        self.model.calls.clear()
        original = dict(self.inputs)
        regularizer, diagnostics = crow.consistency_regularizer(self.model, self.inputs, settings)
        self.assertTrue(regularizer.requires_grad)
        self.assertEqual(len(self.model.calls), 2)
        torch.testing.assert_close(self.model.calls[1]["inputs_embeds"], embeds.detach() + delta)
        for call in self.model.calls:
            self.assertIs(call["attention_mask"], self.inputs["attention_mask"])
            self.assertIs(call["position_ids"], self.inputs["position_ids"])
            self.assertIs(call["cache_position"], self.inputs["cache_position"])
            self.assertEqual(call["logits_to_keep"], 1)
            self.assertFalse(call["use_cache"])
            self.assertNotIn("labels", call)
            self.assertNotIn("loss_scale", call)
            self.assertNotIn("text_position_ids", call)
        self.assertEqual(set(self.inputs), set(original))
        self.assertFalse(diagnostics["crow_search_consistency"].requires_grad)

    def test_existing_gradients_untouched_until_explicit_backward(self):
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, 0.37)
        before = [parameter.grad.clone() for parameter in self.model.parameters()]
        regularizer, _ = crow.consistency_regularizer(self.model, self.inputs, crow.CrowSettings())
        for parameter, existing in zip(self.model.parameters(), before):
            torch.testing.assert_close(parameter.grad, existing)
        parameters = tuple(self.model.parameters())
        expected = torch.autograd.grad(regularizer, parameters, retain_graph=True, allow_unused=True)
        regularizer.backward()
        for parameter, existing, increment in zip(parameters, before, expected):
            torch.testing.assert_close(parameter.grad, existing if increment is None else existing + increment)
        self.assertTrue(any(increment is not None and bool(increment.abs().sum()) for increment in expected))

    def test_zero_epsilon_is_clean_consistency_without_search(self):
        regularizer, diagnostics = crow.consistency_regularizer(self.model, self.inputs,
                                                               crow.CrowSettings(epsilon=0))
        self.assertEqual(len(self.model.calls), 1)
        embeds = self.model.embedding(self.inputs["input_ids"])
        expected = crow.adjacent_inconsistency(self.model(inputs_embeds=embeds).hidden_states,
                                              self.inputs["attention_mask"].bool())
        torch.testing.assert_close(regularizer, expected)
        self.assertNotIn("crow_search_consistency", diagnostics)

    def test_padding_excluded_and_official_layer_slice_retained(self):
        mask = torch.tensor([[True, False]])
        first = torch.tensor([[[1.0, 0], [1.0, 0]]])
        second = torch.tensor([[[0.0, 1], [1.0, 0]]])
        excluded = torch.ones_like(first)
        states = (excluded, first, second, excluded)
        torch.testing.assert_close(crow.adjacent_inconsistency(states, mask), torch.tensor(1.0))
        states[2][:, 1] = torch.tensor([-100.0, 70])
        torch.testing.assert_close(crow.adjacent_inconsistency(states, mask), torch.tensor(1.0))

    def test_non_reentrant_checkpoint_gradients_match(self):
        ordinary, _ = crow.consistency_regularizer(self.model, self.inputs, crow.CrowSettings())
        expected = torch.autograd.grad(ordinary, tuple(self.model.parameters()), allow_unused=True)
        self.model.checkpoint = True
        checkpointed, _ = crow.consistency_regularizer(self.model, self.inputs, crow.CrowSettings())
        actual = torch.autograd.grad(checkpointed, tuple(self.model.parameters()), allow_unused=True)
        torch.testing.assert_close(ordinary, checkpointed)
        for first, second in zip(expected, actual):
            if first is None:
                self.assertIsNone(second)
            else:
                torch.testing.assert_close(first, second)

    def test_no_grad_evaluation_still_builds_fgsm_search(self):
        with torch.no_grad():
            loss, _ = crow.consistency_regularizer(self.model, self.inputs, crow.CrowSettings())
        self.assertFalse(loss.requires_grad)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_bad_batches_fail_explicitly(self):
        for change in ({"attention_mask": torch.zeros(2, 3)},
                       {"attention_mask": torch.ones(2, 1, 3, 3)},
                       {"attention_mask": torch.full((2, 3), 0.5)},
                       {"image_grid_thw": torch.ones(1, 3)},
                       {"inputs_embeds": torch.zeros(2, 3, 4)}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                crow.consistency_regularizer(self.model, {**self.inputs, **change}, crow.CrowSettings())
        with self.assertRaisesRegex(ValueError, "four"):
            crow.adjacent_inconsistency((torch.zeros(2, 3, 4),) * 3, self.inputs["attention_mask"].bool())

    def test_trainer_preserves_clean_loss_and_records_components(self):
        class Metric:
            def update(self, value):
                self.value = value

        class BaseTrainer:
            def __init__(self, model, args, template):
                self.custom_metrics = {"train": defaultdict(Metric)}
                self.template = template

            def compute_loss(self, model, inputs, return_outputs, num_items_in_batch):
                self.clean_labels = inputs.pop("labels")
                self.clean_num_items = num_items_in_batch
                return model.layers[0].weight.square().mean(), "clean-outputs"

        trainer_class = crow.make_trainer(BaseTrainer, crow.CrowSettings())
        template = types.SimpleNamespace(remove_post_encode_hook=lambda: [], register_post_encode_hook=lambda models: None)
        trainer = trainer_class(self.model, types.SimpleNamespace(), template)
        clean_expected = self.model.layers[0].weight.square().mean()
        loss, outputs = trainer.compute_loss(self.model, self.inputs, return_outputs=True, num_items_in_batch=3)
        metrics = trainer.custom_metrics["train"]
        torch.testing.assert_close(loss, clean_expected + 5.5 * metrics["crow_consistency"].value)
        torch.testing.assert_close(metrics["crow_clean_ce"].value, clean_expected)
        self.assertEqual(outputs, "clean-outputs")
        self.assertEqual(trainer.clean_num_items, 3)
        self.assertIn("labels", self.inputs)
        zero_trainer = crow.make_trainer(BaseTrainer, crow.CrowSettings(alpha=0))(
            self.model, types.SimpleNamespace(), template)
        self.model.calls.clear()
        torch.testing.assert_close(zero_trainer.compute_loss(self.model, self.inputs), clean_expected)
        self.assertEqual(len(self.model.calls), 0)

    def test_swift_style_post_encode_runs_for_ce_but_not_auxiliary_embeddings(self):
        class Template:
            mode = "train"
            def __init__(self):
                self.handles = []
                self.encoded = 0
            def pre_forward_hook(self, model, args, kwargs):
                self.encoded += 1
                ids = kwargs.pop("input_ids")
                kwargs["inputs_embeds"] = model.get_input_embeddings()(ids)
                return args, kwargs
            def register_post_encode_hook(self, models):
                for model in models:
                    self.handles.append((model, model.register_forward_pre_hook(self.pre_forward_hook, with_kwargs=True)))
            def remove_post_encode_hook(self):
                models = []
                for model, handle in self.handles:
                    models.append(model)
                    handle.remove()
                self.handles.clear()
                return models

        class BaseTrainer:
            def __init__(self, model, args, template):
                self.template = template
                metric = lambda: types.SimpleNamespace(update=lambda value: None)
                self.custom_metrics = {"train": defaultdict(metric)}
            def compute_loss(self, model, inputs, return_outputs, num_items_in_batch):
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                position_ids=inputs["position_ids"])
                return outputs.hidden_states[-1].square().mean(), outputs

        self.model.checkpoint = True
        template = Template()
        template.register_post_encode_hook([self.model])
        self.addCleanup(template.remove_post_encode_hook)
        trainer = crow.make_trainer(BaseTrainer, crow.CrowSettings())(
            self.model, types.SimpleNamespace(), template)
        loss = trainer.compute_loss(self.model, self.inputs)
        self.assertEqual(template.encoded, 1)
        self.assertEqual(len(self.model.calls), 3)
        self.assertEqual(len(template.handles), 1)
        self.assertEqual(template.mode, "train")
        self.assertTrue(self.model.training)
        for call in self.model.calls:
            self.assertIs(call["attention_mask"], self.inputs["attention_mask"])
            self.assertIs(call["position_ids"], self.inputs["position_ids"])
            self.assertNotIn("input_ids", call)
        loss.backward()
        self.assertEqual(template.encoded, 1)
        trainer.compute_loss(self.model, self.inputs)
        self.assertEqual(template.encoded, 2)


if __name__ == "__main__":
    unittest.main()
