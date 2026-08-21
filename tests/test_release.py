import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from meditor import medqa
from meditor.agents import make_agent
from meditor.coordinator_medrag import MedRAGCoordinator
from meditor.custom_rag.bm25_index import BM25Index, BM25Searcher
from meditor.router import DefaultRouter
from meditor.runner import main


class ParsingTests(unittest.TestCase):
    def test_parse_label(self):
        self.assertEqual(medqa.parse_label("Reasoning\nAnswer: C<END>"), "C")
        self.assertIsNone(medqa.parse_label("No final choice"))


class RouterTests(unittest.TestCase):
    def test_agreement_takes_shallow_path(self):
        decision = DefaultRouter().decide_post_seed(
            a_seed="B",
            b_seed="B",
            a_failed=False,
            b_failed=False,
            disable_debate=False,
            debate_rounds=3,
            coord_enabled=True,
            judge_enabled=True,
        )
        self.assertEqual(decision.decision, "SEED_AGREE")
        self.assertEqual(decision.pred, "B")
        self.assertFalse(decision.need_coord)

    def test_disagreement_takes_deep_path(self):
        decision = DefaultRouter().decide_post_seed(
            a_seed="A",
            b_seed="B",
            a_failed=False,
            b_failed=False,
            disable_debate=False,
            debate_rounds=3,
            coord_enabled=True,
            judge_enabled=True,
        )
        self.assertEqual(decision.decision, "DISAGREE")
        self.assertTrue(decision.need_coord)
        self.assertTrue(decision.need_debate)
        self.assertTrue(decision.need_judge)


class EndpointTests(unittest.TestCase):
    def test_api_key_is_attached_to_agent_config(self):
        agent = make_agent(
            "test",
            "https://example.invalid/v1",
            "test-model",
            api_key="test-key",
        )
        config = agent.llm_config["config_list"][0]
        self.assertEqual(config["api_key"], "test-key")

    def test_remote_rag_mode_does_not_load_local_retrieval(self):
        coordinator = MedRAGCoordinator(
            remote_rag_base_url="https://example.invalid/v1",
            remote_rag_model="test-rag",
        )
        self.assertIsNotNone(coordinator.remote_rag_agent)
        self.assertIsNone(coordinator.medrag)


class CustomRetrievalTests(unittest.TestCase):
    def test_bm25_index_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks_path = root / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "renal-1",
                    "doc_id": "renal",
                    "source": "example",
                    "title": "Kidney",
                    "text": "The kidney filters blood and produces urine.",
                    "meta": {},
                },
                {
                    "chunk_id": "cardiac-1",
                    "doc_id": "cardiac",
                    "source": "example",
                    "title": "Heart",
                    "text": "The heart pumps blood through the circulation.",
                    "meta": {},
                },
            ]
            chunks_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            index_dir = root / "indexes" / "bm25"
            BM25Index().build(str(chunks_path), str(index_dir))

            hits = BM25Searcher(str(index_dir)).search("urine filtration", top_k=1)
            self.assertEqual(hits[0].chunk_id, "renal-1")


class RunnerSmokeTests(unittest.TestCase):
    def test_cached_agreement_run_needs_no_endpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "input.jsonl"
            a_path = root / "a.jsonl"
            b_path = root / "b.jsonl"
            run_dir = root / "run"
            data_path.write_text(
                json.dumps(
                    {
                        "id": "case-1",
                        "question": "Which organ filters blood to produce urine?",
                        "options": {
                            "A": "Liver",
                            "B": "Kidney",
                            "C": "Pancreas",
                            "D": "Spleen",
                        },
                        "gold": "B",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cached = json.dumps({"id": "case-1", "pred": "B"}) + "\n"
            a_path.write_text(cached, encoding="utf-8")
            b_path.write_text(cached, encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--data",
                        str(data_path),
                        "--run-dir",
                        str(run_dir),
                        "--a-base",
                        "http://127.0.0.1:1/v1",
                        "--a-model",
                        "unused-a",
                        "--b-base",
                        "http://127.0.0.1:1/v1",
                        "--b-model",
                        "unused-b",
                        "--a-baseline-preds",
                        str(a_path),
                        "--b-baseline-preds",
                        str(b_path),
                        "--disable-debate",
                        "--no-llm-judge",
                    ]
                )

            prediction = json.loads(
                (run_dir / "preds.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(prediction["pred"], "B")
            self.assertTrue(prediction["correct"])


if __name__ == "__main__":
    unittest.main()
