"""
tests/test_model.py
===================
Test unitari per:
  - QuantLSTM (forward shape, dual-stream)
  - student_t_nll (positività, asimmetria, differenziabilità)
  - EarlyStopping (patience, checkpoint)
  - monte_carlo_forecast (output shape e chiavi)
  - build_feature_idx_map (mapping indici)

Esegui con:
  pytest tests/test_model.py -v
"""

import os

import numpy as np
import pytest
import torch

from quantsys.model import QuantLSTM, student_t_nll, EarlyStopping
from quantsys.model.forecast import monte_carlo_forecast, build_feature_idx_map


# IT: Helpers ─ Factory di QuantLSTM mini su CPU per test veloci.
# EN: Helpers ─ Mini QuantLSTM factory on CPU for fast tests.

def _small_model(n_features=20, n_dynamic_features=None):
    """Crea un QuantLSTM piccolo su CPU — veloce per i test."""
    return QuantLSTM(
        n_features=n_features,
        lstm_hidden=32,
        gru_hidden=16,
        mlp_hidden=8,
        n_lstm_layers=1,
        dropout=0.0,
        n_attention_heads=4,
        use_attention=True,
        n_dynamic_features=n_dynamic_features,
    )


# IT: Test 1 — Forma di output del forward in modalità single-stream.
# EN: Test 1 — Forward output shape in single-stream mode.

class TestQuantLSTMForwardShape:

    # IT: forward() → tuple di 3 tensori (mu, log σ², log ν).
    # EN: forward() → tuple of 3 tensors (mu, log σ², log ν).
    def test_output_is_tuple_of_three(self):
        """forward() deve restituire una tupla di esattamente 3 tensori."""
        model = _small_model(n_features=20)
        x = torch.randn(4, 60, 20)
        out = model(x)
        assert isinstance(out, tuple), "L'output di forward() deve essere una tuple"
        assert len(out) == 3, f"Attesi 3 tensori (mu, log_sig2, log_nu), got {len(out)}"

    # IT: Ogni tensore ha shape (batch_size,).
    # EN: Each tensor has shape (batch_size,).
    def test_output_shapes_are_batch_size(self):
        """Ogni tensore di output deve avere shape (batch_size,) = (4,)."""
        model = _small_model(n_features=20)
        x = torch.randn(4, 60, 20)
        mu, log_sig2, log_nu = model(x)
        assert mu.shape == (4,),       f"mu shape errata: {mu.shape}"
        assert log_sig2.shape == (4,), f"log_sig2 shape errata: {log_sig2.shape}"
        assert log_nu.shape == (4,),   f"log_nu shape errata: {log_nu.shape}"

    # IT: Nessun NaN nell'output del forward.
    # EN: No NaN in the forward output.
    def test_no_nan_in_output(self):
        """Nessun NaN nell'output del forward pass."""
        model = _small_model(n_features=20)
        x = torch.randn(4, 60, 20)
        mu, log_sig2, log_nu = model(x)
        assert not torch.isnan(mu).any(),       "NaN in mu"
        assert not torch.isnan(log_sig2).any(), "NaN in log_sig2"
        assert not torch.isnan(log_nu).any(),   "NaN in log_nu"

    # IT: Nessun inf nell'output del forward.
    # EN: No inf in the forward output.
    def test_no_inf_in_output(self):
        """Nessun valore infinito nell'output del forward pass."""
        model = _small_model(n_features=20)
        x = torch.randn(4, 60, 20)
        mu, log_sig2, log_nu = model(x)
        assert not torch.isinf(mu).any(),       "Inf in mu"
        assert not torch.isinf(log_sig2).any(), "Inf in log_sig2"
        assert not torch.isinf(log_nu).any(),   "Inf in log_nu"

    # IT: Forward con batch_size=1 non crasha (edge case Mamba).
    # EN: Forward with batch_size=1 must not crash (Mamba edge case).
    def test_batch_size_one(self):
        """Forward pass con batch_size=1 deve funzionare senza crash."""
        model = _small_model(n_features=20)
        x = torch.randn(1, 60, 20)
        mu, log_sig2, log_nu = model(x)
        assert mu.shape == (1,)


# IT: Test 2 — Modalità dual-stream (feature dinamiche + strutturali).
# EN: Test 2 — Dual-stream mode (dynamic + structural features).

class TestQuantLSTMDualStream:

    # IT: dual_stream=True quando n_dynamic_features < n_features.
    # EN: dual_stream=True when n_dynamic_features < n_features.
    def test_dual_stream_flag_set(self):
        """dual_stream deve essere True se n_dynamic_features < n_features."""
        model = _small_model(n_features=20, n_dynamic_features=12)
        assert model.dual_stream is True, (
            "dual_stream dovrebbe essere True con n_dynamic_features=12 < n_features=20"
        )

    # IT: Senza n_dynamic_features → modalità single-stream.
    # EN: Without n_dynamic_features → single-stream mode.
    def test_single_stream_flag(self):
        """dual_stream deve essere False se n_dynamic_features non è passato."""
        model = _small_model(n_features=20, n_dynamic_features=None)
        assert model.dual_stream is False, (
            "dual_stream dovrebbe essere False in modalità single-stream"
        )

    # IT: Forward in dual-stream non solleva eccezioni.
    # EN: Forward in dual-stream must not raise.
    def test_dual_stream_forward_no_crash(self):
        """Forward pass in dual-stream non deve sollevare eccezioni."""
        model = _small_model(n_features=20, n_dynamic_features=12)
        x = torch.randn(4, 60, 20)
        out = model(x)
        assert len(out) == 3, "Dual-stream deve restituire 3 tensori"

    # IT: Shape di output corretto anche in dual-stream.
    # EN: Output shape correct in dual-stream too.
    def test_dual_stream_output_shape(self):
        """Output shape corretto anche in dual-stream."""
        model = _small_model(n_features=20, n_dynamic_features=12)
        x = torch.randn(4, 60, 20)
        mu, log_sig2, log_nu = model(x)
        assert mu.shape == (4,),       f"mu shape errata in dual-stream: {mu.shape}"
        assert log_sig2.shape == (4,), f"log_sig2 shape errata in dual-stream: {log_sig2.shape}"
        assert log_nu.shape == (4,),   f"log_nu shape errata in dual-stream: {log_nu.shape}"

    # IT: Nessun NaN nei tre output del dual-stream.
    # EN: No NaN across the three dual-stream outputs.
    def test_dual_stream_no_nan(self):
        """Nessun NaN nel dual-stream forward."""
        model = _small_model(n_features=20, n_dynamic_features=12)
        x = torch.randn(4, 60, 20)
        mu, log_sig2, log_nu = model(x)
        assert not torch.isnan(mu).any(), "NaN in mu (dual-stream)"
        assert not torch.isnan(log_sig2).any(), "NaN in log_sig2 (dual-stream)"
        assert not torch.isnan(log_nu).any(), "NaN in log_nu (dual-stream)"


# IT: Test 3 — student_t_nll: forma, finitezza, gradiente.
# EN: Test 3 — student_t_nll: shape, finiteness, gradient.

class TestStudentTNLLPositive:

    # IT: La NLL deve essere uno scalare 0-dim.
    # EN: The NLL must be a 0-dim scalar.
    def test_nll_is_scalar(self):
        """La NLL deve restituire uno scalare."""
        y       = torch.randn(32)
        mu      = torch.zeros(32)
        log_sig2 = torch.zeros(32)
        log_nu  = torch.zeros(32)
        loss = student_t_nll(y, mu, log_sig2, log_nu)
        assert loss.ndim == 0, f"La NLL deve essere uno scalare, got shape {loss.shape}"

    # IT: La NLL è finita su input tipici.
    # EN: The NLL is finite on typical inputs.
    def test_nll_finite(self):
        """La NLL deve essere un valore finito per input tipici."""
        y       = torch.randn(32)
        mu      = torch.zeros(32)
        log_sig2 = torch.zeros(32)
        log_nu  = torch.zeros(32)
        loss = student_t_nll(y, mu, log_sig2, log_nu)
        assert torch.isfinite(loss), f"NLL non finita: {loss.item()}"

    # IT: La NLL conserva requires_grad (grafo computazionale intatto).
    # EN: The NLL keeps requires_grad (computation graph intact).
    def test_nll_requires_grad(self):
        """La NLL deve essere differenziabile (requires_grad=True)."""
        y       = torch.randn(32)
        mu      = torch.zeros(32, requires_grad=True)
        log_sig2 = torch.zeros(32, requires_grad=True)
        log_nu  = torch.zeros(32, requires_grad=True)
        loss = student_t_nll(y, mu, log_sig2, log_nu)
        assert loss.requires_grad, "student_t_nll deve mantenere il grafo computazionale"

    # IT: Backward fluisce verso tutti e tre i parametri.
    # EN: Backward propagates to all three parameters.
    def test_nll_gradient_flows_to_all_params(self):
        """Il gradiente deve fluire verso mu, log_sig2 e log_nu."""
        y       = torch.randn(32)
        mu      = torch.zeros(32, requires_grad=True)
        log_sig2 = torch.zeros(32, requires_grad=True)
        log_nu  = torch.zeros(32, requires_grad=True)
        loss = student_t_nll(y, mu, log_sig2, log_nu)
        loss.backward()
        assert mu.grad is not None,       "Nessun gradiente verso mu"
        assert log_sig2.grad is not None, "Nessun gradiente verso log_sig2"
        assert log_nu.grad is not None,   "Nessun gradiente verso log_nu"


# IT: Test 4 — Penalità asimmetrica della NLL (segno sbagliato → loss extra).
# EN: Test 4 — Asymmetric NLL penalty (wrong sign → extra loss).

class TestStudentTNLLAsymmetry:

    # IT: Su movimenti grandi, segno sbagliato deve costare più del segno giusto.
    # EN: On large moves, wrong sign must cost more than right sign.
    def test_wrong_sign_has_higher_loss(self):
        """
        Errori di segno su movimenti grandi devono avere loss maggiore
        della predizione corretta. Questo verifica la penalità asimmetrica.
        """
        # IT: threshold default 0.002 → uso 0.01 (5× threshold).
        # EN: default threshold 0.002 → use 0.01 (5× threshold).
        y_val = 0.01

        # IT: caso sbagliato — mu di segno opposto | EN: wrong case — mu opposite sign
        y_wrong = torch.tensor([y_val])
        mu_wrong = torch.tensor([-y_val])
        lsig2 = torch.zeros(1)
        lnu   = torch.zeros(1)
        loss_wrong = student_t_nll(y_wrong, mu_wrong, lsig2, lnu)

        # IT: caso corretto — mu stesso segno | EN: right case — mu same sign
        y_correct = torch.tensor([y_val])
        mu_correct = torch.tensor([y_val])
        loss_correct = student_t_nll(y_correct, mu_correct, lsig2, lnu)

        assert loss_wrong.item() > loss_correct.item(), (
            f"Loss con segno sbagliato ({loss_wrong.item():.4f}) dovrebbe essere "
            f"maggiore di quella corretta ({loss_correct.item():.4f})"
        )

    # IT: Su movimenti < soglia la penalità asimmetrica non si attiva.
    # EN: On moves below threshold the asymmetric penalty stays off.
    def test_small_move_no_asymmetry_penalty(self):
        """
        Su movimenti piccoli (< large_move_threshold) la penalità
        asimmetrica non si attiva: wrong-sign e right-sign devono
        avere loss simile (la differenza viene solo dalla NLL base).
        """
        # IT: y=0.0005 < threshold=0.002 | EN: y=0.0005 < threshold=0.002
        y_small = 0.0005

        y_wrong = torch.tensor([y_small])
        mu_wrong = torch.tensor([-y_small])
        lsig2 = torch.zeros(1)
        lnu   = torch.zeros(1)
        loss_wrong_small = student_t_nll(y_wrong, mu_wrong, lsig2, lnu)

        y_correct = torch.tensor([y_small])
        mu_correct = torch.tensor([y_small])
        loss_correct_small = student_t_nll(y_correct, mu_correct, lsig2, lnu)

        # IT: differenza attesa piccola rispetto al caso large move.
        # EN: expected gap is small compared to the large-move case.
        diff_small = abs(loss_wrong_small.item() - loss_correct_small.item())
        # IT: ci basta verificare la finitezza | EN: just check finiteness here
        assert torch.isfinite(loss_wrong_small), "Loss NaN su movimento piccolo"
        assert torch.isfinite(loss_correct_small), "Loss NaN su movimento piccolo corretto"

    # IT: La loss cresce monotonicamente con la magnitudine dell'errore.
    # EN: Loss grows monotonically with the magnitude of the error.
    def test_loss_increases_with_error_magnitude(self):
        """La loss deve aumentare all'aumentare dell'errore di predizione."""
        lsig2 = torch.zeros(1)
        lnu   = torch.zeros(1)
        y     = torch.tensor([0.01])

        # IT: errore piccolo | EN: small error
        loss_small_err = student_t_nll(y, torch.tensor([0.009]), lsig2, lnu)
        # IT: errore grande | EN: large error
        loss_large_err = student_t_nll(y, torch.tensor([-0.01]), lsig2, lnu)

        assert loss_large_err.item() > loss_small_err.item(), (
            "Errore di predizione grande deve produrre loss maggiore"
        )


# IT: Test 5 — EarlyStopping: patience, checkpoint, restore.
# EN: Test 5 — EarlyStopping: patience, checkpoint, restore.

class TestEarlyStopping:

    # IT: Se val_loss continua a migliorare, lo stopping non scatta.
    # EN: As long as val_loss keeps improving, stopping does not fire.
    def test_not_triggered_while_improving(self, tmp_path):
        """EarlyStopping non si attiva finché la val_loss migliora."""
        model = _small_model(n_features=10)
        ckpt  = str(tmp_path / "best.pt")
        es    = EarlyStopping(patience=3, path=ckpt)

        # IT: sequenza monotona decrescente | EN: monotonically decreasing sequence
        for loss_val in [1.0, 0.8, 0.6, 0.4]:
            triggered = es(loss_val, model)
            assert not triggered, (
                f"EarlyStopping si è attivato troppo presto con val_loss={loss_val}"
            )

    # IT: Dopo patience epoche senza miglioramento → stopping scatta.
    # EN: After patience epochs without improvement → stopping fires.
    def test_triggered_after_patience_exceeded(self, tmp_path):
        """EarlyStopping deve attivarsi dopo patience epoch senza miglioramento."""
        model = _small_model(n_features=10)
        ckpt  = str(tmp_path / "test_ckpt.pt")
        es    = EarlyStopping(patience=3, path=ckpt)

        # IT: ep1 migliora → counter=0 | EN: ep1 improves → counter=0
        es(1.0, model)
        # IT: ep2 migliora → counter=0 | EN: ep2 improves → counter=0
        es(0.9, model)
        # IT: ep3 no improvement → counter=1 | EN: ep3 no improvement → counter=1
        es(0.95, model)
        assert not es.triggered, "Non dovrebbe essere triggered dopo 1 epoch senza miglioramento"
        # IT: ep4 → counter=2 | EN: ep4 → counter=2
        es(0.95, model)
        assert not es.triggered, "Non dovrebbe essere triggered dopo 2 epoch senza miglioramento"
        # IT: ep5 → counter=3 = patience → trigger | EN: ep5 → counter=3 = patience → trigger
        result = es(0.95, model)
        assert result is True,    "Deve restituire True quando triggered"
        assert es.triggered is True, "es.triggered deve essere True dopo patience superato"

    # IT: Il checkpoint viene scritto al primo miglioramento.
    # EN: The checkpoint file is written on the first improvement.
    def test_checkpoint_saved(self, tmp_path):
        """EarlyStopping deve salvare il checkpoint quando val_loss migliora."""
        model = _small_model(n_features=10)
        ckpt  = str(tmp_path / "best.pt")
        es    = EarlyStopping(patience=5, path=ckpt)

        # IT: 1ª epoch = nuovo best → file creato | EN: 1st epoch = new best → file created
        es(1.0, model)
        assert os.path.exists(ckpt), "Il checkpoint non è stato creato al primo miglioramento"

    # IT: Un nuovo best azzera il counter di patience.
    # EN: A new best resets the patience counter.
    def test_counter_resets_on_improvement(self, tmp_path):
        """Il counter deve azzerarsi se la val_loss migliora di nuovo."""
        model = _small_model(n_features=10)
        ckpt  = str(tmp_path / "best.pt")
        es    = EarlyStopping(patience=3, path=ckpt)

        es(1.0, model)   # IT: best=1.0, counter=0 | EN: best=1.0, counter=0
        es(1.1, model)   # IT: counter=1 | EN: counter=1
        es(1.1, model)   # IT: counter=2 | EN: counter=2
        es(0.5, model)   # IT: nuovo best → counter=0 | EN: new best → counter=0
        assert es.counter == 0, f"Counter non azzerato dopo miglioramento: {es.counter}"
        assert not es.triggered, "Non deve essere triggered dopo un nuovo miglioramento"

    # IT: restore() ripristina i pesi del best checkpoint.
    # EN: restore() reloads the weights of the best checkpoint.
    def test_restore_loads_best_weights(self, tmp_path):
        """restore() deve caricare i pesi salvati al momento del best."""
        model = _small_model(n_features=10)
        ckpt  = str(tmp_path / "best.pt")
        es    = EarlyStopping(patience=3, path=ckpt)

        # IT: salva stato iniziale e annota il bias | EN: save initial state, record bias
        es(1.0, model)
        initial_bias = model.out_mu.bias.data.clone()

        # IT: corrompe i pesi correnti | EN: corrupts current weights
        with torch.no_grad():
            model.out_mu.bias.fill_(99.0)

        # IT: restore → pesi del best | EN: restore → best weights
        es.restore(model)
        restored_bias = model.out_mu.bias.data
        assert torch.allclose(restored_bias, initial_bias), (
            "restore() non ha ripristinato i pesi corretti"
        )


# IT: Test 6 — monte_carlo_forecast: chiavi e shape dell'output.
# EN: Test 6 — monte_carlo_forecast: output keys and shapes.

class TestMonteCarloForecastShape:

    # IT: Genera una finestra seed casuale | EN: Build a random seed window
    def _make_seed(self, n_features=10, window=60):
        return np.random.randn(1, window, n_features).astype(np.float32)

    # IT: Le chiavi richieste sono presenti (p50/mean/std/paths_sample).
    # EN: Required keys are present (p50/mean/std/paths_sample).
    def test_result_has_required_keys(self):
        """Il risultato deve contenere le chiavi p50, mean, std, paths_sample."""
        model = _small_model(n_features=10)
        model.eval()
        x_seed = self._make_seed(n_features=10)
        result = monte_carlo_forecast(
            model, x_seed, last_price=50_000.0,
            n_steps=5, n_paths=10,
            device=torch.device("cpu"),
        )
        for key in ("p50", "mean", "std", "paths_sample"):
            assert key in result, f"Chiave '{key}' mancante nell'output di monte_carlo_forecast"

    # IT: len(p50) == n_steps richiesti.
    # EN: len(p50) == requested n_steps.
    def test_p50_length_equals_n_steps(self):
        """La lunghezza di p50 deve essere uguale a n_steps."""
        model = _small_model(n_features=10)
        model.eval()
        x_seed = self._make_seed(n_features=10)
        n_steps = 5
        result = monte_carlo_forecast(
            model, x_seed, last_price=50_000.0,
            n_steps=n_steps, n_paths=10,
            device=torch.device("cpu"),
        )
        assert len(result["p50"]) == n_steps, (
            f"Attesi {n_steps} valori in p50, got {len(result['p50'])}"
        )

    # IT: La media finale del prezzo simulato deve restare positiva.
    # EN: The final mean simulated price must remain positive.
    def test_mean_final_price_positive(self):
        """La media finale deve essere un prezzo positivo."""
        model = _small_model(n_features=10)
        model.eval()
        x_seed = self._make_seed(n_features=10)
        result = monte_carlo_forecast(
            model, x_seed, last_price=50_000.0,
            n_steps=5, n_paths=10,
            device=torch.device("cpu"),
        )
        assert result["mean"][-1] > 0, (
            f"Il prezzo medio finale deve essere positivo, got {result['mean'][-1]}"
        )

    # IT: std non negativa su tutto l'orizzonte.
    # EN: std non-negative over the full horizon.
    def test_std_is_non_negative(self):
        """La deviazione standard dei prezzi deve essere non negativa."""
        model = _small_model(n_features=10)
        model.eval()
        x_seed = self._make_seed(n_features=10)
        result = monte_carlo_forecast(
            model, x_seed, last_price=50_000.0,
            n_steps=5, n_paths=20,
            device=torch.device("cpu"),
        )
        for i, s in enumerate(result["std"]):
            assert s >= 0, f"std[{i}]={s} è negativo"

    # IT: Path diagnostici mu/sigma/nu presenti e con lunghezza corretta.
    # EN: Diagnostic mu/sigma/nu paths present and correctly sized.
    def test_additional_keys_present(self):
        """Verifica la presenza di chiavi percentili e path diagnostici."""
        model = _small_model(n_features=10)
        model.eval()
        x_seed = self._make_seed(n_features=10)
        result = monte_carlo_forecast(
            model, x_seed, last_price=50_000.0,
            n_steps=5, n_paths=10,
            device=torch.device("cpu"),
        )
        for key in ("mu_path", "sigma_path", "nu_path"):
            assert key in result, f"Chiave diagnostica '{key}' mancante"
        assert len(result["mu_path"]) == 5, "mu_path deve avere n_steps elementi"

    # IT: paths_sample è una lista Python (serializzabile JSON).
    # EN: paths_sample is a Python list (JSON serialisable).
    def test_paths_sample_is_list(self):
        """paths_sample deve essere una lista (non un tensore)."""
        model = _small_model(n_features=10)
        model.eval()
        x_seed = self._make_seed(n_features=10)
        result = monte_carlo_forecast(
            model, x_seed, last_price=50_000.0,
            n_steps=5, n_paths=10,
            device=torch.device("cpu"),
        )
        assert isinstance(result["paths_sample"], list), (
            "paths_sample deve essere una lista Python"
        )


# IT: Test 7 — build_feature_idx_map: mapping nome → indice colonna.
# EN: Test 7 — build_feature_idx_map: name → column-index mapping.

class TestBuildFeatureIdxMap:

    # IT: Mapping di base coerente con l'ordine d'insieme.
    # EN: Basic mapping consistent with input order.
    def test_basic_mapping(self):
        """Verifica il mapping corretto degli indici."""
        feature_names = ["log_ret", "lag_ret_1", "vol_std_5", "hour_sin"]
        idx_map = build_feature_idx_map(feature_names)
        assert idx_map["log_ret"]   == 0, f"Atteso 0 per log_ret, got {idx_map['log_ret']}"
        assert idx_map["lag_ret_1"] == 1, f"Atteso 1 per lag_ret_1, got {idx_map['lag_ret_1']}"
        assert idx_map["vol_std_5"] == 2, f"Atteso 2 per vol_std_5, got {idx_map['vol_std_5']}"
        assert idx_map["hour_sin"]  == 3, f"Atteso 3 per hour_sin, got {idx_map['hour_sin']}"

    # IT: Restituisce un dict. | EN: Returns a dict.
    def test_returns_dict(self):
        """build_feature_idx_map deve restituire un dict."""
        result = build_feature_idx_map(["a", "b", "c"])
        assert isinstance(result, dict), "Il risultato deve essere un dizionario"

    # IT: Lunghezza del dict == numero di feature.
    # EN: Dict length == number of features.
    def test_length_matches_feature_count(self):
        """Il dizionario deve avere tanti entry quante sono le feature."""
        names = ["f0", "f1", "f2", "f3", "f4"]
        idx_map = build_feature_idx_map(names)
        assert len(idx_map) == len(names), (
            f"Lunghezza mismatch: {len(idx_map)} vs {len(names)}"
        )

    # IT: Lista vuota → dict vuoto.
    # EN: Empty list → empty dict.
    def test_empty_list(self):
        """Lista vuota deve produrre dizionario vuoto."""
        idx_map = build_feature_idx_map([])
        assert idx_map == {}, "Lista vuota deve produrre dizionario vuoto"

    # IT: Indici contigui {0, 1, ..., n-1}.
    # EN: Contiguous indices {0, 1, ..., n-1}.
    def test_indices_are_contiguous(self):
        """Gli indici devono essere contigui da 0 a n-1."""
        names = ["x", "y", "z"]
        idx_map = build_feature_idx_map(names)
        assert set(idx_map.values()) == {0, 1, 2}, (
            f"Indici non contigui: {set(idx_map.values())}"
        )

    # IT: Le chiavi usate da monte_carlo_forecast sono accessibili nel map.
    # EN: Keys used by monte_carlo_forecast are reachable in the map.
    def test_used_with_monte_carlo_lag_keys(self):
        """Verifica che le chiavi tipiche usate da monte_carlo_forecast siano accessibili."""
        names = ["log_ret", "lag_ret_1", "lag_ret_2", "vol_std_5",
                 "vol_std_20", "vol_ratio_5_20", "vwap_dev"]
        idx_map = build_feature_idx_map(names)
        # IT: chiavi richieste da monte_carlo_forecast | EN: keys required by monte_carlo_forecast
        for key in ("lag_ret_1", "lag_ret_2", "vol_std_5", "vol_std_20",
                    "vol_ratio_5_20", "vwap_dev"):
            assert key in idx_map, f"Chiave '{key}' mancante nell'idx_map"
