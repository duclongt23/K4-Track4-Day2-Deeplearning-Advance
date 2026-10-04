"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Chạy: python -m unittest test_components -v  (trong code/)."""
import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import losses  # noqa: E402
from inference import apply_temperature, fit_temperature, fuse_conv_bn  # noqa: E402
from model import build_model, count_params, param_groups  # noqa: E402
from train import EMA, Config, build_scheduler, build_optimizer, import_eval, _set_frozen_bn_eval  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)


class TestLosses(unittest.TestCase):
    def setUp(self):
        self.logits = torch.randn(32, 9)
        self.y = torch.randint(0, 9, (32,))

    def test_focal_gamma0_equals_ce(self):
        fl = losses.FocalLoss(gamma=0.0)(self.logits, self.y)
        ce = F.cross_entropy(self.logits, self.y)
        self.assertLess(abs(fl.item() - ce.item()), 1e-6)

    def test_focal_gamma2_is_smaller_than_ce(self):
        self.assertLess(losses.FocalLoss(gamma=2.0)(self.logits, self.y).item(),
                        F.cross_entropy(self.logits, self.y).item())

    def test_label_smoothing_eps0_equals_ce(self):
        ls = losses.LabelSmoothingCE(smoothing=0.0)(self.logits, self.y)
        self.assertLess(abs(ls.item() - F.cross_entropy(self.logits, self.y).item()), 1e-6)

    def test_losses_are_modules_and_move_with_to(self):
        crit = losses.build_criterion("focal", gamma=2.0, alpha=torch.ones(9)).to("cpu")
        self.assertIsInstance(crit, torch.nn.Module)
        self.assertTrue(math.isfinite(crit(self.logits, self.y).item()))

    def test_class_weights_default_beta_none(self):
        w = losses.class_weights([5463, 675, 637, 618, 613, 637, 605, 644, 609], None)
        self.assertEqual(len(w), 9)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertGreater(w[1].item(), w[0].item())  # lớp hiếm có trọng số lớn hơn

    def test_class_balanced_weights_sum_to_k(self):
        w = losses.class_weights([5463, 675, 637, 618, 613, 637, 605, 644, 609], 0.999)
        self.assertAlmostEqual(w.sum().item(), 9.0, places=4)

    def test_initial_ce_is_about_ln9_for_uniform_logits(self):
        ce = F.cross_entropy(torch.zeros(64, 9), torch.randint(0, 9, (64,)))
        self.assertAlmostEqual(ce.item(), math.log(9), places=5)


class TestMix(unittest.TestCase):
    def test_cutmix_lam_matches_real_pasted_area(self):
        # Ảnh i toàn giá trị i: vùng bị đổi chính là hộp dán, nên lam phải = 1 - tỉ lệ pixel bị đổi.
        for seed in range(30):
            np.random.seed(seed); torch.manual_seed(seed)
            x = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1).repeat(1, 3, 32, 32)
            xm, (ya, yb, lam) = losses.mix_batch(x, torch.arange(6), alpha=1.0, mode="cutmix")
            self.assertTrue(0.0 <= lam <= 1.0)
            for i in range(6):
                if yb[i] == ya[i]:
                    continue                                    # dán chính nó thì không thấy thay đổi
                frac_changed = (xm[i, 0] != x[i, 0]).float().mean().item()
                self.assertAlmostEqual(frac_changed, 1 - lam, places=5)

    def test_mixup_is_convex_combination(self):
        np.random.seed(3)
        x = torch.rand(4, 3, 8, 8)
        xm, (ya, yb, lam) = losses.mix_batch(x, torch.arange(4), 1.0, "mixup")
        self.assertTrue(0.0 <= lam <= 1.0)
        self.assertTrue(xm.min() >= x.min() - 1e-6 and xm.max() <= x.max() + 1e-6)

    def test_mixed_loss_mixes_both_labels(self):
        logits = torch.randn(8, 9)
        ya, yb = torch.randint(0, 9, (8,)), torch.randint(0, 9, (8,))
        crit = torch.nn.CrossEntropyLoss()
        got = losses.mixed_loss(crit, logits, (ya, yb, 0.3))
        want = 0.3 * crit(logits, ya) + 0.7 * crit(logits, yb)
        self.assertAlmostEqual(got.item(), want.item(), places=6)


class TestModelAndOptim(unittest.TestCase):
    def test_param_groups_no_weight_decay_for_norm_and_bias(self):
        m = timm.create_model("resnet18", pretrained=False, num_classes=9)
        groups = param_groups(m, 1e-4, 1e-3, 0.05)
        self.assertEqual(len(groups), 3)
        self.assertEqual([g["weight_decay"] for g in groups], [0.05, 0.0, 0.05])
        self.assertEqual([g["lr"] for g in groups], [1e-4, 1e-4, 1e-3])
        self.assertTrue(all(p.ndim <= 1 for p in groups[1]["params"]))
        n = sum(len(g["params"]) for g in groups)
        self.assertEqual(n, len(list(m.parameters())))            # không sót, không trùng

    def test_frozen_only_trains_head(self):
        m = build_model("resnet18", pretrained=False, num_classes=9, init="scratch")
        from model import freeze_backbone
        freeze_backbone(m)
        trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        self.assertTrue(trainable and all(n.startswith("fc.") for n in trainable))
        m.train(); _set_frozen_bn_eval(m)
        self.assertFalse(m.bn1.training)          # BN backbone ở eval
        self.assertTrue(m.fc.training)

    def test_count_params(self):
        self.assertAlmostEqual(count_params(timm.create_model("resnet50", pretrained=False, num_classes=9)), 23.52, delta=0.1)

    def test_warmup_then_cosine_schedule(self):
        m = timm.create_model("resnet18", pretrained=False, num_classes=9)
        cfg = Config(epochs=4, warmup_epochs=1.0)
        opt = build_optimizer(m, cfg)
        sch = build_scheduler(opt, cfg, steps_per_epoch=10)
        lrs = []
        for _ in range(40):
            lrs.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
        self.assertLess(lrs[0], 0.05 * cfg.lr_backbone)           # bắt đầu rất nhỏ
        self.assertAlmostEqual(max(lrs), cfg.lr_backbone, delta=1e-6 * 100)
        self.assertLess(lrs[-1], 0.05 * cfg.lr_backbone)          # cosine về gần 0

    def test_ema_moves_toward_weights(self):
        m = torch.nn.Linear(2, 2)
        ema = EMA(m, 0.9)
        with torch.no_grad():
            m.weight.add_(1.0)
        ema.update(m)
        self.assertTrue(torch.allclose(ema.shadow["weight"], m.weight - 0.9, atol=1e-6))
        tgt = torch.nn.Linear(2, 2)
        ema.copy_to(tgt)
        self.assertTrue(torch.allclose(tgt.weight, ema.shadow["weight"]))


class TestInference(unittest.TestCase):
    def test_fuse_bn_is_equivalent_and_actually_fuses(self):
        for name in ("resnet18", "efficientnet_b0", "mobilenetv3_small_100"):
            m = timm.create_model(name, pretrained=False, num_classes=9).eval()
            for mod in m.modules():                                  # thống kê BN ngẫu nhiên để phép thử có ý nghĩa
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.running_mean.normal_(0, 0.3); mod.running_var.uniform_(0.5, 2.0)
                    mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_(0, 0.2)
            fused = fuse_conv_bn(m)
            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                self.assertLess((m(x) - fused(x)).abs().max().item(), 1e-3, name)
            self.assertGreater(fused.n_fused, 0, name)
            n_bn = sum(isinstance(k, torch.nn.BatchNorm2d) for k in fused.modules())
            self.assertEqual(n_bn, sum(isinstance(k, torch.nn.BatchNorm2d) for k in m.modules()) - fused.n_fused)

    def test_fuse_bn_leaves_layernorm_models_unchanged(self):
        m = timm.create_model("convnext_atto", pretrained=False, num_classes=9).eval()
        self.assertEqual(fuse_conv_bn(m).n_fused, 0)

    def test_temperature_reduces_nll_and_keeps_argmax(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 9, 2000)
        logits = rng.normal(size=(2000, 9)) * 1.0
        logits[np.arange(2000), y] += 2.0
        logits *= 3.0                                                # quá tự tin
        T = fit_temperature(logits, y)
        self.assertGreater(T, 1.2)   # logit quá tự tin -> T > 1
        nll = lambda p: -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        self.assertLess(nll(apply_temperature(logits, T)), nll(apply_temperature(logits, 1.0)))
        np.testing.assert_array_equal(apply_temperature(logits, T).argmax(1), logits.argmax(1))
        ev = import_eval()
        before = ev.ece_score(apply_temperature(logits, 1.0), y)
        after = ev.ece_score(apply_temperature(logits, T), y)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
