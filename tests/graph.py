from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

src_path = Path(__file__).parent.parent / 'src/agents/'
sys.path.append(str(src_path))
from agents import diversity_augmenting_agent, privacy_agent, synthetic_data_generator
sys.path.append(str(Path(__file__).resolve().parent.parent))

class AgentState(TypedDict, total=False):

    D: list  # Embedded dataset
    topic_vectorstore: Any  # Chroma vectorstore for clustered topics
    D_priv: str  # Privacy-analyzed and pseudonymized text
    D_synth: str  # Final synthetic dataset
    qa_count: int  # Number of Q&A pairs to generate
    privacy_report: str  # Report generated by the privacy analyzer agent


def create_graph():
    workflow = StateGraph(AgentState)

    # Add agents to the workflow
    workflow.add_node("diversity", diversity_augmenting_agent)
  #  workflow.add_node("privacy_analysis", privacy_analyzer_agent)  # Stage 1: Analyze entities for PII
    workflow.add_node("privacy", privacy_agent)  # Stage 2: Pseudonymize data
    workflow.add_node("synthesis", synthetic_data_generator)



    # Define the pipeline structure
    workflow.set_entry_point("diversity")
    workflow.add_edge("diversity", "privacy")  # Link diversity to privacy analysis
 #   workflow.add_edge("privacy_analysis", "privacy")  # Link privacy analysis to pseudonymization
    workflow.add_edge("privacy", "synthesis")  # Link pseudonymization to synthesis
    workflow.add_edge("synthesis", END)  # End the workflow

    return workflow.compile()




