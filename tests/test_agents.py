import unittest

from run.tools.agents import (
    privacy_agent,
    synthetic_data_generator,
)


class TestAgentFunctions(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
