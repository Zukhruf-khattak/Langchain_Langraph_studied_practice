"""
04. Iterative AI Answer Improvement Workflow
============================================
A LangGraph workflow demonstrating iterative loops, reflection patterns,
quality evaluation (critic node), and conditional self-improvement.

Architecture Flow:
    START
      │
      ▼
Generate Answer
      │
      ▼
Critic Node (Evaluation & Scoring)
      │
      ▼
Is Quality Met (Score >= 8) OR Iteration >= Max?
      ├── Yes (Good Enough / Max Retries) ──► END
      └── No (Needs Improvement) ──► Improve Answer ──► (Back to Critic)
"""

import os
import sys
import argparse
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain & LangGraph imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# Rich formatting for beautiful CLI visualization
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich import print as rprint
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


# ============================================================================
# 1. Configuration & Model Initialization
# ============================================================================

def setup_environment() -> str:
    """Load environment variables and validate Google API Key."""
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        print("\n[WARNING] GOOGLE_API_KEY is not set or is using the default placeholder.")
        print("Please set your API key in the .env file or export GOOGLE_API_KEY='your-key'\n")
    return api_key or ""


def get_llm(model_name: Optional[str] = None, temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Initialize Google Gemini Chat Model."""
    model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


# ============================================================================
# 2. Workflow State & Structured Output Schemas
# ============================================================================

class CritiqueOutput(BaseModel):
    """Pydantic model representing structured feedback from the Critic Node."""
    score: int = Field(
        description="Overall quality score from 1 to 10 evaluating correctness, clarity, depth, and completeness.",
        ge=1,
        le=10
    )
    strengths: List[str] = Field(
        description="List of positive aspects and strengths of the current answer."
    )
    weaknesses: List[str] = Field(
        description="List of weaknesses, missing details, inaccuracies, or structural issues."
    )
    critique: str = Field(
        description="Actionable, constructive feedback explaining specifically how to improve the answer."
    )


class WorkflowState(TypedDict):
    """Shared state dictionary representing the workflow context across iterations."""
    question: str
    answer: str
    critique: str
    score: int
    strengths: List[str]
    weaknesses: List[str]
    iteration: int
    max_iterations: int
    target_score: int
    history: List[Dict[str, Any]]


# ============================================================================
# 3. Prompt Templates
# ============================================================================

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant. Provide a direct, concise initial answer to the user's question. "
        "Do not overcomplicate, but provide a solid baseline draft."
    ),
    ("user", "Question: {question}")
])

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an objective, rigorous, and constructive expert reviewer. "
        "Your task is to evaluate the provided answer to the user's question.\n\n"
        "Evaluation Criteria:\n"
        "1. Accuracy & Correctness: Is the explanation factually accurate and technically sound?\n"
        "2. Clarity & Structure: Is it well-organized, readable, and easy to follow?\n"
        "3. Completeness & Depth: Does it explain key mechanisms, underlying reasons, and practical value?\n"
        "4. Relevance: Does it directly address the specific question?\n\n"
        "Scoring Scale (1-10):\n"
        "- 1-4: Poor, superficial, incomplete, or contains inaccuracies.\n"
        "- 5-7: Decent/Average, answers the question but lacks technical depth or thorough explanation.\n"
        "- 8-10: Excellent, comprehensive, well-structured, clear, and informative.\n\n"
        "Target Quality Threshold: {target_score}/10."
    ),
    (
        "user",
        "User Question:\n{question}\n\n"
        "Current Answer to Evaluate:\n{answer}\n\n"
        "Current Iteration: {iteration}\n"
        "Please provide your structured evaluation."
    )
])

IMPROVE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert AI editor and response refiner. "
        "Your goal is to significantly improve the previous answer by acting on the critic's feedback.\n\n"
        "Guidelines for improvement:\n"
        "1. Address every weakness identified by the critic.\n"
        "2. Preserve and build upon the accurate strengths of the previous answer.\n"
        "3. Enhance technical precision, structure (use clear headings/bullet points where appropriate), and depth.\n"
        "4. Output only the revised, improved answer."
    ),
    (
        "user",
        "Original Question:\n{question}\n\n"
        "Previous Answer (Score: {score}/10):\n{answer}\n\n"
        "Critic Strengths:\n{strengths}\n\n"
        "Critic Weaknesses:\n{weaknesses}\n\n"
        "Critic Actionable Feedback:\n{critique}\n\n"
        "Generate the improved version of the answer now:"
    )
])


# ============================================================================
# 4. Graph Node Functions
# ============================================================================

def generate_initial_answer(state: WorkflowState) -> Dict[str, Any]:
    """Node 1: Generates the initial baseline answer for the question."""
    llm = get_llm(temperature=0.7)
    chain = GENERATE_PROMPT | llm
    
    response = chain.invoke({"question": state["question"]})
    answer_text = response.content.strip()
    
    history_entry = {
        "step": "Initial Generation",
        "iteration": 0,
        "answer": answer_text,
        "score": None,
        "critique": None
    }
    
    current_history = list(state.get("history", []))
    current_history.append(history_entry)
    
    return {
        "answer": answer_text,
        "iteration": 0,
        "history": current_history
    }


def critic_evaluation(state: WorkflowState) -> Dict[str, Any]:
    """Node 2: Evaluates current answer, assigning score and detailed feedback."""
    llm = get_llm(temperature=0.2)
    structured_critic = llm.with_structured_output(CritiqueOutput)
    chain = CRITIC_PROMPT | structured_critic
    
    target_score = state.get("target_score", 8)
    
    try:
        feedback: CritiqueOutput = chain.invoke({
            "question": state["question"],
            "answer": state["answer"],
            "iteration": state["iteration"],
            "target_score": target_score
        })
        score = feedback.score
        critique = feedback.critique
        strengths = feedback.strengths
        weaknesses = feedback.weaknesses
    except Exception as e:
        # Fallback if structured output fails
        score = 6
        critique = f"Could not parse structured output ({str(e)}). Please improve depth and clarity."
        strengths = ["Answers the question at a high level."]
        weaknesses = ["Needs more technical depth and clearer structure."]
    
    history_entry = {
        "step": f"Critic Evaluation (Iteration {state['iteration']})",
        "iteration": state["iteration"],
        "answer": state["answer"],
        "score": score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "critique": critique
    }
    
    current_history = list(state.get("history", []))
    current_history.append(history_entry)
    
    return {
        "score": score,
        "critique": critique,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "history": current_history
    }


def improve_answer(state: WorkflowState) -> Dict[str, Any]:
    """Node 3: Refines the answer using critique feedback and increments iteration."""
    llm = get_llm(temperature=0.5)
    chain = IMPROVE_PROMPT | llm
    
    strengths_str = "\n- ".join(state.get("strengths", [])) if state.get("strengths") else "None provided."
    weaknesses_str = "\n- ".join(state.get("weaknesses", [])) if state.get("weaknesses") else "None provided."
    
    response = chain.invoke({
        "question": state["question"],
        "answer": state["answer"],
        "score": state.get("score", 0),
        "strengths": f"- {strengths_str}",
        "weaknesses": f"- {weaknesses_str}",
        "critique": state.get("critique", "")
    })
    
    improved_text = response.content.strip()
    next_iteration = state["iteration"] + 1
    
    history_entry = {
        "step": f"Answer Improvement (Iteration {next_iteration})",
        "iteration": next_iteration,
        "answer": improved_text,
        "score": None,
        "critique": None
    }
    
    current_history = list(state.get("history", []))
    current_history.append(history_entry)
    
    return {
        "answer": improved_text,
        "iteration": next_iteration,
        "history": current_history
    }


# ============================================================================
# 5. Conditional Routing Logic
# ============================================================================

def route_evaluation(state: WorkflowState) -> str:
    """
    Decides whether to terminate the loop or route back to improvement node.
    
    Termination Conditions:
    1. Answer quality meets or exceeds target_score (e.g., Score >= 8) -> 'good_enough'
    2. Maximum iterations limit reached (e.g., Iteration >= 3) -> 'max_reached'
    """
    score = state.get("score", 0)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    target_score = state.get("target_score", 8)
    
    if score >= target_score:
        return "good_enough"
    elif iteration >= max_iterations:
        return "max_reached"
    else:
        return "needs_improvement"


# ============================================================================
# 6. Graph Construction & Compilation
# ============================================================================

def build_workflow():
    """Constructs and compiles the LangGraph StateGraph."""
    builder = StateGraph(WorkflowState)
    
    # Register Nodes
    builder.add_node("generate_answer", generate_initial_answer)
    builder.add_node("critic", critic_evaluation)
    builder.add_node("improve_answer", improve_answer)
    
    # Define Edges
    builder.add_edge(START, "generate_answer")
    builder.add_edge("generate_answer", "critic")
    
    # Define Conditional Routing from Critic
    builder.add_conditional_edges(
        "critic",
        route_evaluation,
        {
            "good_enough": END,
            "max_reached": END,
            "needs_improvement": "improve_answer"
        }
    )
    
    # Loop back from Improvement to Critic
    builder.add_edge("improve_answer", "critic")
    
    return builder.compile()


# ============================================================================
# 7. Execution & Display Helpers
# ============================================================================

def print_banner():
    """Prints the project banner."""
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]LangGraph Iterative AI Answer Improvement Workflow[/bold cyan]\n"
            "[dim]A Reflection & Self-Correction Pattern using Google Gemini[/dim]",
            border_style="cyan"
        ))
    else:
        print("=" * 70)
        print(" LangGraph Iterative AI Answer Improvement Workflow")
        print("=" * 70)


def display_state_update(node_name: str, state: Dict[str, Any]):
    """Displays real-time step progress during graph execution."""
    iteration = state.get("iteration", 0)
    
    if not HAS_RICH:
        print(f"\n---> Node Executed: [{node_name}] | Iteration: {iteration}")
        if node_name == "critic":
            print(f"     Score: {state.get('score', 'N/A')}/10")
            print(f"     Feedback: {state.get('critique', 'N/A')[:100]}...")
        return

    if node_name == "generate_answer":
        console.print(Panel(
            f"[bold green]Initial Answer Generated[/bold green]\n\n{state['answer'][:250]}...",
            title=f"Node: {node_name} (Iteration {iteration})",
            border_style="green"
        ))
    elif node_name == "critic":
        score = state.get("score", 0)
        target = state.get("target_score", 8)
        score_color = "green" if score >= target else "yellow" if score >= 6 else "red"
        
        strengths_list = "\n".join([f"  [green]+[/green] {s}" for s in state.get("strengths", [])])
        weaknesses_list = "\n".join([f"  [red]-[/red] {w}" for w in state.get("weaknesses", [])])
        
        content = (
            f"[bold]Quality Score:[/bold] [{score_color}]{score}/10[/{score_color}] (Target: {target}/10)\n\n"
            f"[bold]Strengths:[/bold]\n{strengths_list}\n\n"
            f"[bold]Weaknesses:[/bold]\n{weaknesses_list}\n\n"
            f"[bold]Critic Feedback:[/bold]\n{state.get('critique', '')}"
        )
        console.print(Panel(
            content,
            title=f"Node: {node_name} (Evaluation #{iteration})",
            border_style=score_color
        ))
    elif node_name == "improve_answer":
        console.print(Panel(
            f"[bold blue]Improved Answer Draft (Iteration {iteration})[/bold blue]\n\n{state['answer'][:250]}...",
            title=f"Node: {node_name} (Iteration {iteration})",
            border_style="blue"
        ))


def run_workflow(question: str, target_score: int = 8, max_iterations: int = 3):
    """Executes the iterative graph with streaming updates."""
    setup_environment()
    app = build_workflow()
    
    initial_state: WorkflowState = {
        "question": question,
        "answer": "",
        "critique": "",
        "score": 0,
        "strengths": [],
        "weaknesses": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "target_score": target_score,
        "history": []
    }
    
    if HAS_RICH:
        console.print(f"\n[bold yellow]User Question:[/bold yellow] {question}")
        console.print(f"[dim]Settings: Target Score = {target_score}/10 | Max Iterations = {max_iterations}[/dim]\n")
    else:
        print(f"\nUser Question: {question}")
        print(f"Target Score: {target_score}/10 | Max Iterations: {max_iterations}\n")
    
    final_state = None
    
    # Stream node execution
    for output in app.stream(initial_state):
        for node_name, state_update in output.items():
            # Update working state
            if final_state is None:
                final_state = dict(initial_state)
            final_state.update(state_update)
            display_state_update(node_name, final_state)
    
    # Print Final Summary Report
    print_summary_report(final_state)
    return final_state


def print_summary_report(state: Dict[str, Any]):
    """Displays a summary table and the final polished answer."""
    if not state:
        return
        
    score = state.get("score", 0)
    target = state.get("target_score", 8)
    iterations_used = state.get("iteration", 0)
    
    if HAS_RICH:
        table = Table(title="Iteration History & Score Trajectory", show_header=True, header_style="bold magenta")
        table.add_column("Step", style="cyan")
        table.add_column("Iteration", justify="center")
        table.add_column("Score", justify="center")
        table.add_column("Action / Summary")
        
        for h in state.get("history", []):
            score_display = f"{h['score']}/10" if h.get("score") is not None else "-"
            summary = h.get("critique", "")[:80] + "..." if h.get("critique") else h.get("answer", "")[:80] + "..."
            table.add_row(h["step"], str(h["iteration"]), score_display, summary)
            
        console.print("\n")
        console.print(table)
        
        status_msg = (
            f"[bold green]Target Score Met ({score}/{target}) in {iterations_used} iteration(s)![/bold green]"
            if score >= target
            else f"[bold yellow]Max Iterations Reached ({iterations_used}). Final Score: {score}/{target}[/bold yellow]"
        )
        console.print(f"\n{status_msg}\n")
        
        console.print(Panel(
            Markdown(state["answer"]),
            title="[bold green]Final Polished Output[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
    else:
        print("\n" + "=" * 70)
        print(f"FINAL RESULT: Score {score}/{target} | Total Iterations: {iterations_used}")
        print("=" * 70)
        print("\nFinal Answer:\n")
        print(state["answer"])
        print("\n" + "=" * 70)


# ============================================================================
# 8. Main Entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LangGraph Iterative AI Answer Improvement Workflow")
    parser.add_argument(
        "-q", "--question",
        type=str,
        default="Explain why Retrieval-Augmented Generation (RAG) is useful in AI applications.",
        help="The question to answer and iteratively improve."
    )
    parser.add_argument(
        "-t", "--target-score",
        type=int,
        default=8,
        help="Target score between 1 and 10 to stop iterations (default: 8)."
    )
    parser.add_argument(
        "-m", "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of improvement iterations before stopping (default: 3)."
    )
    
    args = parser.parse_args()
    print_banner()
    run_workflow(args.question, target_score=args.target_score, max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
