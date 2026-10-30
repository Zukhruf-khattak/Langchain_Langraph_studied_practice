"""
Human-in-the-Loop (HITL) Approval Workflow
===========================================
A LangGraph project demonstrating workflow interruption, checkpointing,
and human decision-making with Google Gemini.

Author: Antigravity AI & LangGraph Series
License: MIT
"""

import os
import sys
import uuid
import argparse
from typing import TypedDict, Optional, List, Dict, Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage

# Fix UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Initialize Rich Console
console = Console(force_terminal=True)

# Load environment variables
load_dotenv()


# ==========================================
# 1. State Definition
# ==========================================
class ContentState(TypedDict):
    """
    Workflow state for the Human-in-the-Loop Content Approval system.
    """
    topic: str
    generated_content: str
    approval_status: Optional[str]  # 'approved' | 'rejected' | 'pending'
    human_feedback: Optional[str]
    final_content: Optional[str]
    iteration_count: int
    revision_history: List[Dict[str, Any]]


# ==========================================
# 2. LLM Initialization with Graceful Fallback
# ==========================================
def get_llm():
    """
    Initialize Google Gemini LLM or return None if unconfigured.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    if not api_key or api_key.startswith("your_") or "invalid" in api_key.lower():
        return None
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7
        )
    except Exception:
        return None


def generate_llm_draft(topic: str) -> str:
    """
    Generate initial content using Gemini LLM or realistic mock fallback.
    """
    llm = get_llm()
    if llm:
        try:
            prompt = (
                f"You are a professional content creator. Write an engaging, well-structured "
                f"post or brief article about the following topic:\n\n"
                f"Topic: {topic}\n\n"
                f"Guidelines:\n"
                f"- Include a compelling hook\n"
                f"- 2-3 concise body paragraphs with actionable insights\n"
                f"- A strong takeaway or call-to-action\n"
                f"- Use formatting (bullet points, clear paragraphs) for readability."
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            console.print(f"[yellow]Notice: Gemini API call fallback ({e}). Using built-in generator.[/yellow]")

    # Realistic fallback draft
    return (
        f"🚀 The Future of {topic}: Why It Matters Today\n\n"
        f"In today's fast-evolving technological landscape, {topic} is rapidly shifting from an experimental concept "
        f"into an essential driver of innovation. Organizations that embrace this paradigm are seeing "
        f"unprecedented gains in efficiency, decision velocity, and creative problem-solving.\n\n"
        f"Key Highlights:\n"
        f"• Autonomous Execution: Moving beyond simple static queries to goal-oriented multi-step reasoning.\n"
        f"• Human-AI Synergy: Pairing automated intelligence with human oversight for safety and precision.\n"
        f"• Scalable Reliability: Structuring workflows with robust state management and error resilience.\n\n"
        f"As we look ahead, the defining advantage won't just be having AI capabilities, but orchestrating them seamlessly "
        f"with human expertise.\n\n"
        f"💡 What are your thoughts on how {topic} will shape your industry? Share below!"
    )


def refine_llm_draft(topic: str, previous_draft: str, feedback: str, iteration: int) -> str:
    """
    Refine content using Gemini LLM incorporating human feedback.
    """
    llm = get_llm()
    if llm:
        try:
            prompt = (
                f"You are an expert editor refining content based on human reviewer feedback.\n\n"
                f"Original Topic: {topic}\n\n"
                f"Previous Draft (Iteration {iteration - 1}):\n"
                f"\"\"\"\n{previous_draft}\n\"\"\"\n\n"
                f"Human Reviewer Feedback:\n"
                f"\"\"\"{feedback}\"\"\"\n\n"
                f"Task:\n"
                f"Rewrite and improve the draft strictly addressing all feedback while maintaining high quality."
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            console.print(f"[yellow]Notice: Gemini API call fallback ({e}). Using built-in refiner.[/yellow]")

    # Realistic fallback refinement
    return (
        f"✨ [Refined Iteration {iteration}] {topic}\n\n"
        f"Addressing Reviewer Notes: \"{feedback}\"\n\n"
        f"In modern enterprise systems, {topic} fundamentally transforms operational workflows. "
        f"Rather than operating in an isolated silo, modern implementations focus on actionable business outcomes, "
        f"lower latency, and verifiable human control.\n\n"
        f"Core Takeaways:\n"
        f"1. Precision & Focus: Directly tailored to meet user feedback ({feedback}).\n"
        f"2. Controlled Automation: Predictable guardrails ensuring compliance and high reliability.\n"
        f"3. Measurable Impact: Streamlined execution delivering tangible ROI.\n\n"
        f"Summary: Continuous refinement ensures both AI speed and human quality standards."
    )


# ==========================================
# 3. Graph Node Functions
# ==========================================
def generate_content(state: ContentState) -> Dict[str, Any]:
    """
    Node 1: Generate initial content based on the user's requested topic.
    """
    topic = state.get("topic", "AI Systems")
    draft = generate_llm_draft(topic)
    
    return {
        "generated_content": draft,
        "iteration_count": 1,
        "approval_status": "pending",
        "revision_history": [
            {
                "iteration": 1,
                "content": draft,
                "action": "generated",
                "feedback": None
            }
        ]
    }


def human_review(state: ContentState) -> Dict[str, Any]:
    """
    Node 2: Pause workflow execution via interrupt() to await human review.
    When resumed with Command(resume=...), this node receives the decision payload.
    """
    iteration = state.get("iteration_count", 1)
    content = state.get("generated_content", "")
    
    # Interrupt execution and send payload to caller / checkpointer
    review_response = interrupt({
        "message": "Human review required. Please inspect the draft and decide to approve or reject.",
        "topic": state.get("topic"),
        "current_iteration": iteration,
        "generated_content": content,
        "history": state.get("revision_history", [])
    })
    
    # review_response is the value provided to Command(resume=...)
    if not isinstance(review_response, dict):
        review_response = {"action": "approve" if str(review_response).lower() in ("approve", "yes", "1") else "reject"}
    
    action = review_response.get("action", "reject").lower()
    feedback = review_response.get("feedback", "")
    override_content = review_response.get("override_content")

    # Update revision history
    history = list(state.get("revision_history", []))
    history.append({
        "iteration": iteration,
        "action": action,
        "feedback": feedback,
        "override": bool(override_content)
    })
    
    updates: Dict[str, Any] = {
        "approval_status": action,
        "human_feedback": feedback,
        "revision_history": history
    }
    
    # If human provided direct content edits
    if override_content:
        updates["generated_content"] = override_content

    return updates


def decide_next_step(state: ContentState) -> str:
    """
    Conditional Edge: Evaluates approval status to route the graph.
    - 'approve' -> 'finalize_content'
    - 'reject'  -> 'improve_content'
    """
    status = state.get("approval_status", "").lower()
    if status in ("approve", "approved", "accept"):
        return "finalize_content"
    return "improve_content"


def improve_content(state: ContentState) -> Dict[str, Any]:
    """
    Node 3: Refine content based on the human reviewer's feedback.
    """
    topic = state.get("topic", "")
    previous_draft = state.get("generated_content", "")
    feedback = state.get("human_feedback", "Make it better.")
    new_iteration = state.get("iteration_count", 1) + 1
    
    refined_draft = refine_llm_draft(topic, previous_draft, feedback, new_iteration)
    
    history = list(state.get("revision_history", []))
    history.append({
        "iteration": new_iteration,
        "content": refined_draft,
        "action": "improved",
        "feedback_addressed": feedback
    })
    
    return {
        "generated_content": refined_draft,
        "iteration_count": new_iteration,
        "approval_status": "pending",
        "revision_history": history
    }


def finalize_content(state: ContentState) -> Dict[str, Any]:
    """
    Node 4: Mark content as final and complete the workflow.
    """
    return {
        "final_content": state.get("generated_content", ""),
        "approval_status": "approved"
    }


# ==========================================
# 4. Workflow Builder & Compiler
# ==========================================
def build_hitl_workflow():
    """
    Constructs and compiles the Human-in-the-Loop StateGraph with MemorySaver checkpointer.
    """
    workflow = StateGraph(ContentState)
    
    # Add nodes
    workflow.add_node("generate_content", generate_content)
    workflow.add_node("human_review", human_review)
    workflow.add_node("improve_content", improve_content)
    workflow.add_node("finalize_content", finalize_content)
    
    # Add edges
    workflow.add_edge(START, "generate_content")
    workflow.add_edge("generate_content", "human_review")
    
    workflow.add_conditional_edges(
        "human_review",
        decide_next_step,
        {
            "finalize_content": "finalize_content",
            "improve_content": "improve_content"
        }
    )
    
    workflow.add_edge("improve_content", "human_review")
    workflow.add_edge("finalize_content", END)
    
    # Compile with checkpointer for state preservation and pause/resume
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app


# ==========================================
# 5. Interactive CLI & Runner
# ==========================================
def run_interactive_workflow(topic: str):
    """
    Runs the Human-in-the-Loop approval workflow with an interactive terminal UI.
    """
    console.print(Panel.fit(
        f"[bold cyan]🤖 LangGraph Human-in-the-Loop Approval Workflow[/bold cyan]\n"
        f"[dim]Topic:[/dim] [white bold]{topic}[/white bold]",
        border_style="cyan"
    ))
    
    app = build_hitl_workflow()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Step 1: Initial invocation (runs until first interrupt)
    console.print("\n[bold yellow]⚡ Phase 1: Initiating Content Generation...[/bold yellow]")
    with console.status("[bold green]Generating content with AI...[/bold green]", spinner="dots"):
        events = list(app.stream({"topic": topic}, config=config))
    
    while True:
        state = app.get_state(config)
        
        # Check if workflow is paused on interrupt
        if not state.next:
            # Workflow completed
            break
            
        current_content = state.values.get("generated_content", "")
        iteration = state.values.get("iteration_count", 1)
        
        # Display the review box
        console.print(Panel(
            Markdown(current_content),
            title=f"📝 [bold yellow]Human Review Required — Draft (Iteration {iteration})[/bold yellow]",
            subtitle="[dim]Workflow is currently PAUSED at Checkpoint[/dim]",
            border_style="yellow",
            padding=(1, 2)
        ))
        
        console.print("\n[bold cyan]Reviewer Options:[/bold cyan]")
        console.print("  [bold green][1][/bold green] Approve and Finalize Content")
        console.print("  [bold red][2][/bold red] Reject and Request AI Revision (with Feedback)")
        console.print("  [bold blue][3][/bold blue] Edit Draft Manually and Approve")
        console.print("  [bold grey70][4][/bold grey70] Cancel & Exit Workflow\n")
        
        choice = Prompt.ask(
            "[bold white]Select an action[/bold white]",
            choices=["1", "2", "3", "4"],
            default="1"
        )
        
        if choice == "1":
            console.print("\n[bold green]✅ Approved! Resuming workflow to finalize...[/bold green]")
            resume_cmd = Command(resume={"action": "approve", "feedback": "Approved by human reviewer."})
            for event in app.stream(resume_cmd, config=config):
                pass
            break
            
        elif choice == "2":
            feedback = Prompt.ask("\n[bold yellow]💬 Enter your feedback for the AI[/bold yellow]")
            while not feedback.strip():
                feedback = Prompt.ask("[bold red]Feedback cannot be empty. Please enter instructions for improvement[/bold red]")
            
            console.print(f"\n[bold magenta]🔄 Rejection submitted. Resuming workflow to improve draft...[/bold magenta]")
            with console.status("[bold magenta]AI is refining draft based on your feedback...[/bold magenta]", spinner="dots"):
                resume_cmd = Command(resume={"action": "reject", "feedback": feedback.strip()})
                for event in app.stream(resume_cmd, config=config):
                    pass
            # Loop will continue to next review prompt
            
        elif choice == "3":
            console.print("\n[bold blue]✏️ Manual Edit Mode[/bold blue]")
            console.print("[dim]Paste or write your revised content below (Press Enter, then Ctrl+Z / Ctrl+D when done, or enter a single-line summary):[/dim]")
            manual_text = Prompt.ask("[bold blue]Edited text[/bold blue]", default=current_content)
            
            console.print("\n[bold green]✅ Manually edited content approved! Resuming workflow...[/bold green]")
            resume_cmd = Command(resume={
                "action": "approve",
                "feedback": "Manually edited and approved by human.",
                "override_content": manual_text
            })
            for event in app.stream(resume_cmd, config=config):
                pass
            break
            
        elif choice == "4":
            console.print("\n[bold red]❌ Workflow cancelled by user.[/bold red]")
            return None

    # Step 3: Final state inspection
    final_state = app.get_state(config)
    final_content = final_state.values.get("final_content", final_state.values.get("generated_content", ""))
    history = final_state.values.get("revision_history", [])
    
    console.print("\n" + "=" * 60)
    console.print(Panel(
        Markdown(final_content),
        title="🎉 [bold green]FINAL APPROVED CONTENT[/bold green]",
        subtitle=f"[dim]Total Iterations: {final_state.values.get('iteration_count', 1)}[/dim]",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Display revision history table
    table = Table(title="Workflow Execution History", border_style="cyan")
    table.add_column("Step", justify="center", style="dim")
    table.add_column("Iteration", justify="center", style="bold")
    table.add_column("Action", style="cyan")
    table.add_column("Notes / Feedback", style="white")
    
    for idx, item in enumerate(history, 1):
        action_name = item.get("action", "N/A")
        if action_name == "approve":
            action_style = "[green]Approved[/green]"
        elif action_name == "reject":
            action_style = "[red]Rejected[/red]"
        elif action_name == "improved":
            action_style = "[magenta]Refined[/magenta]"
        else:
            action_style = f"[blue]{action_name}[/blue]"
            
        feedback_text = item.get("feedback") or item.get("feedback_addressed") or "Initial generation"
        table.add_row(str(idx), str(item.get("iteration", "-")), action_style, str(feedback_text))
        
    console.print(table)
    return final_content


# ==========================================
# 6. Automated Test / Simulation Mode
# ==========================================
def run_simulation():
    """
    Non-interactive simulation verifying the end-to-end interrupt & resume cycle.
    """
    console.print("\n[bold cyan]🧪 Running Automated Simulation Test...[/bold cyan]")
    app = build_hitl_workflow()
    thread_id = "sim-thread-" + str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}}
    
    topic = "Write a professional LinkedIn post about the importance of AI agents."
    
    # 1. Initial run
    console.print("\n[bold]1. Starting graph stream...[/bold]")
    events = list(app.stream({"topic": topic}, config=config))
    state = app.get_state(config)
    assert state.next == ("human_review",), f"Expected paused at human_review, got {state.next}"
    console.print("  [green]✓ Graph paused at human_review interrupt.[/green]")
    console.print(f"  [dim]Draft iteration 1 length: {len(state.values['generated_content'])} chars[/dim]")
    
    # 2. Reject with feedback
    console.print("\n[bold]2. Simulating Human Rejection with feedback...[/bold]")
    feedback = "Make the post shorter, more punchy, and highlight ROI for engineering leaders."
    resume_cmd = Command(resume={"action": "reject", "feedback": feedback})
    events = list(app.stream(resume_cmd, config=config))
    
    state = app.get_state(config)
    assert state.next == ("human_review",), f"Expected loop back to human_review, got {state.next}"
    assert state.values["iteration_count"] == 2, f"Expected iteration 2, got {state.values['iteration_count']}"
    console.print("  [green]✓ Graph looped through improve_content and paused again at human_review.[/green]")
    console.print(f"  [dim]Draft iteration 2 length: {len(state.values['generated_content'])} chars[/dim]")
    
    # 3. Approve
    console.print("\n[bold]3. Simulating Human Approval...[/bold]")
    resume_cmd = Command(resume={"action": "approve", "feedback": "Looks great!"})
    events = list(app.stream(resume_cmd, config=config))
    
    state = app.get_state(config)
    assert state.next == (), f"Expected workflow completion, got {state.next}"
    assert state.values.get("approval_status") == "approved"
    assert "final_content" in state.values
    console.print("  [green]✓ Graph routed to finalize_content and completed successfully![/green]")
    console.print("\n[bold green]✅ All simulation checks passed successfully![/bold green]\n")


# ==========================================
# 7. Main Entrypoint
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="LangGraph Human-in-the-Loop Approval Workflow")
    parser.add_argument("--test", action="store_true", help="Run automated test simulation without interactive prompts")
    parser.add_argument("--topic", type=str, default=None, help="Custom content topic or prompt")
    args = parser.parse_args()
    
    if args.test:
        run_simulation()
        return

    # Check for topic argument or prompt interactively
    topic = args.topic
    if not topic:
        console.print("\n[bold cyan]Select or Enter a Topic to Generate & Review:[/bold cyan]")
        console.print("  [1] Write a professional LinkedIn post about the importance of AI agents.")
        console.print("  [2] Draft an announcement for an enterprise AI workflow tool.")
        console.print("  [3] Write a technical summary on Human-in-the-Loop safety in LLM systems.")
        console.print("  [4] Custom topic (Enter your own prompt)")
        
        choice = Prompt.ask("\n[bold white]Choose option[/bold white]", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "1":
            topic = "Write a professional LinkedIn post about the importance of AI agents."
        elif choice == "2":
            topic = "Draft an announcement for an enterprise AI workflow tool."
        elif choice == "3":
            topic = "Write a technical summary on Human-in-the-Loop safety in LLM systems."
        else:
            topic = Prompt.ask("\n[bold yellow]Enter your custom topic/prompt[/bold yellow]")
            while not topic.strip():
                topic = Prompt.ask("[bold red]Topic cannot be empty. Please enter a prompt[/bold red]")
                
    run_interactive_workflow(topic)


if __name__ == "__main__":
    main()
