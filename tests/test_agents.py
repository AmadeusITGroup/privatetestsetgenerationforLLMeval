import unittest
from unittest.mock import patch
import numpy as np
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src/run/tools/'
sys.path.append(str(src_path))


from agents import diversity_augmenting_agent, privacy_agent, synthetic_data_generator

class TestAgentFunctions(unittest.TestCase):

    def test_diversity_augmenting_agent_with_few_samples(self):


        state = {
            "D": [
                {"text": "Sample text 1", "embedding": np.array([0.1, 0.2, 0.3])},
                {"text": "Sample text 2", "embedding": np.array([0.4, 0.5, 0.6])},
                {"text": "Sample text 3", "embedding": np.array([0.7, 0.8, 0.9])}
            ]
        }

        with patch('sklearn.cluster.KMeans.fit_predict', return_value=np.array([0, 1, 0])):
            result = diversity_augmenting_agent(state)

        # Assert that the result contains the "topic_vectorstore"
        self.assertIn("topic_vectorstore", result)
        self.assertIsNotNone(result["topic_vectorstore"])

    def test_privacy_agent_with_no_vectorstore(self):
        # Simulate state without a vectorstore
        state = {}

        result = privacy_agent(state)

        self.assertEqual(result["privacy_analysis_report"], "No topics to analyze.")

    def test_synthetic_data_generator_with_missing_data(self):
        # Simulate state with no sanitized text (D_priv)
        state = {}

        result = synthetic_data_generator(state)

        self.assertEqual(result["D_synth"], "No data to synthesize.")

if __name__ == '__main__':
    unittest.main()
