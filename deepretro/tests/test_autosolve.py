"""Unit tests for deepretro.algorithms.autosolve.

Tests the AutoSolver using mocked pipeline calls so the tests
run without AiZynthFinder models, LLM API keys, or any other heavy
infrastructure.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

mock_llm_module = types.ModuleType("deepretro.algorithms.llm")
mock_llm_module.llm_pipeline = lambda **kwargs: ([], [], [])
sys.modules.setdefault("deepretro.algorithms.llm", mock_llm_module)

mock_az_module = types.ModuleType("deepretro.utils.az")
mock_az_module.run_az = lambda smiles, az_model="USPTO": (False, [])
sys.modules.setdefault("deepretro.utils.az", mock_az_module)

from deepretro.algorithms.autosolve import AutoSolver
from deepretro.logging import logger as context_logger
from deepretro.models.hallucination_helpers import (
    build_ml_checker,
    resolve_hallucination,
)

BENZENE = "c1ccccc1"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

FAKE_TREE = {"type": "mol", "smiles": BENZENE, "children": []}
FAKE_RESULT = {
    "dependencies": {"1": []},
    "steps": [
        {
            "step": "1",
            "reactants": [{"smiles": ETHANOL}],
            "reagents": [],
            "products": [{"smiles": BENZENE}],
            "conditions": [],
            "reactionmetrics": [{"confidenceestimate": 0.85}],
        }
    ],
}


@pytest.fixture()
def mock_pipeline():
    """Patch recurse and format_output so tests never touch real infra."""
    with patch.object(
        AutoSolver, "recurse",
        autospec=True,
        return_value=(FAKE_TREE, True),
    ) as mock_recurse, patch(
        "deepretro.algorithms.autosolve.format_output",
        return_value=FAKE_RESULT,
    ) as mock_fmt:
        yield mock_recurse, mock_fmt


class TestAutoSolver:
    """Tests for ``AutoSolver.solve``."""

    def test_returns_result_dict(self, mock_pipeline) -> None:
        solver = AutoSolver(hallucination_mode="none")
        result = solver.solve(BENZENE)
        assert result == FAKE_RESULT

    def test_passes_params_to_solver(self, mock_pipeline) -> None:
        solver = AutoSolver(
            llm="test-model",
            az_model="USPTO",
            stability_check=False,
            hallucination_mode="none",
            use_protecting_group_feature=True,
        )
        assert solver.llm == "test-model"
        assert solver.az_model == "USPTO"
        assert solver.stability_flag == "False"
        assert solver.hallucination_checker is None
        assert solver.use_protecting_group_feature is True

        solver.solve(ASPIRIN)

    def test_does_not_write_to_disk(self, mock_pipeline, tmp_path: Path) -> None:
        solver = AutoSolver(hallucination_mode="none")
        solver.solve(BENZENE)
        assert list(tmp_path.iterdir()) == []


class TestHallucinationModeIntegration:
    """Tests that hallucination_mode correctly wires into the solver."""

    def test_heuristic_passes_callable(self, mock_pipeline) -> None:
        solver = AutoSolver(hallucination_mode="heuristic")
        assert callable(solver.hallucination_checker)
        solver.solve(BENZENE)

    def test_none_passes_none(self, mock_pipeline) -> None:
        solver = AutoSolver(hallucination_mode="none")
        assert solver.hallucination_checker is None
        solver.solve(BENZENE)

    def test_ml_passes_callable(self, mock_pipeline) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_single = MagicMock()
        solver = AutoSolver(
            hallucination_mode="ml",
            hallucination_classifier=mock_clf,
        )
        assert callable(solver.hallucination_checker)
        solver.solve(BENZENE)

    def test_invalid_mode_raises(self, mock_pipeline) -> None:
        with pytest.raises(ValueError):
            AutoSolver(hallucination_mode="bad")


class TestResolveHallucinationArgs:
    """Tests for the standalone resolve_hallucination helper."""

    def test_heuristic_mode(self) -> None:
        checker = resolve_hallucination("heuristic", None)
        assert callable(checker)

    def test_none_mode(self) -> None:
        checker = resolve_hallucination("none", None)
        assert checker is None

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="hallucination_mode must be"):
            resolve_hallucination("invalid", None)

    def test_ml_mode_without_classifier_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a HallucinationClassifier"):
            resolve_hallucination("ml", None)


class TestBuildMlChecker:
    """Tests for ``build_ml_checker``."""

    @patch("deepretro.utils.utils_molecule.is_valid_smiles", return_value=True)
    def test_keeps_non_hallucinated_pathways(self, _mock_valid) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_single.return_value = {
            "is_hallucination": False, "probability": 0.1,
        }
        checker = build_ml_checker(mock_clf)
        status, kept = checker(BENZENE, [["CC", "O"], ["CCO"]])
        assert status == 200
        assert len(kept) == 2

    @patch("deepretro.utils.utils_molecule.is_valid_smiles", return_value=True)
    def test_drops_hallucinated_pathways(self, _mock_valid) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_single.side_effect = [
            {"is_hallucination": True, "probability": 0.9},
            {"is_hallucination": False, "probability": 0.1},
        ]
        checker = build_ml_checker(mock_clf)
        status, kept = checker(BENZENE, [["CC"], ["CCO"]])
        assert status == 200
        assert kept == [["CCO"]]

    @patch("deepretro.utils.utils_molecule.is_valid_smiles", return_value=True)
    def test_all_hallucinated_returns_empty(self, _mock_valid) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_single.return_value = {
            "is_hallucination": True, "probability": 0.95,
        }
        checker = build_ml_checker(mock_clf)
        status, kept = checker(BENZENE, [["CC"], ["O"]])
        assert status == 200
        assert kept == []


class TestAutoSolverRecurse:
    """Regression tests for recurse branch handling."""

    def test_recurse_keeps_candidate_pathways_isolated(self, monkeypatch) -> None:
        solver = AutoSolver(hallucination_mode="none")
        token = context_logger.set(MagicMock())
        root = "C1CC1"
        solved_a = "CC"
        solved_c = "O"

        def fake_run_az(smiles: str, az_model: str = "USPTO"):
            if smiles in {solved_a, solved_c}:
                return True, [{"type": "mol", "smiles": smiles, "children": []}]
            return False, []

        def fake_llm_pipeline(**kwargs):
            molecule = kwargs["molecule"]
            if molecule == root:
                return [[solved_a, root], [solved_c]], ["bad", "good"], [0.2, 0.9]
            return [["fallback"]], ["fallback"], [0.1]

        monkeypatch.setattr("deepretro.algorithms.autosolve.run_az", fake_run_az)
        monkeypatch.setattr(
            "deepretro.algorithms.autosolve.llm_pipeline", fake_llm_pipeline
        )

        try:
            tree, solved = solver.recurse(root)
        finally:
            context_logger.reset(token)

        assert solved is True
        children = tree["children"][0]["children"]
        assert [child["smiles"] for child in children] == [solved_c]

    def test_recurse_uses_branch_local_visited_state(self, monkeypatch) -> None:
        solver = AutoSolver(hallucination_mode="none")
        token = context_logger.set(MagicMock())
        root = "C1CC1"
        branch_x = "CC"
        branch_y = "CCC"
        shared = "O"

        def fake_run_az(smiles: str, az_model: str = "USPTO"):
            if smiles == shared:
                return True, [{"type": "mol", "smiles": shared, "children": []}]
            return False, []

        def fake_llm_pipeline(**kwargs):
            molecule = kwargs["molecule"]
            if molecule == root:
                return [[branch_x, branch_y]], ["branch"], [0.9]
            if molecule in {branch_x, branch_y}:
                return [[shared]], ["shared"], [0.8]
            return [["fallback"]], ["fallback"], [0.1]

        monkeypatch.setattr("deepretro.algorithms.autosolve.run_az", fake_run_az)
        monkeypatch.setattr(
            "deepretro.algorithms.autosolve.llm_pipeline", fake_llm_pipeline
        )

        try:
            tree, solved = solver.recurse(root)
        finally:
            context_logger.reset(token)

        assert solved is True
        children = tree["children"][0]["children"]
        assert [child["smiles"] for child in children] == [branch_x, branch_y]
