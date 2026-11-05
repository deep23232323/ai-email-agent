import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from send_email import send_mail

load_dotenv()

MAX_FEEDBACK_ROUNDS = 3


class EmailState(BaseModel):
    question: str = ""
    mail_reason: str = ""
    recipient_name: str = ""
    recipient_email: str = ""
    subject: str = ""
    body: str = ""
    feedback: str = ""
    feedback_count: int = 0
    response: str = ""


class UserDetails(BaseModel):
    mail_reason: str = Field(default="")
    recipient_name: str = Field(default="")
    recipient_email: str = Field(default="")


class DraftEmail(BaseModel):
    subject: str = Field(description="Email Subject within 40 words")
    body: str = Field(description="Email body including proper details, reason, greetings and closing")


def _get_llm():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        return ChatGroq(model="openai/gpt-oss-20b", api_key=api_key)
    except Exception:
        return None


def _fallback_extract_user_details(question: str) -> UserDetails:
    normalized = question or ""
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", normalized)
    email = email_match.group(0) if email_match else ""

    name = ""
    name_match = re.search(
        r"(?:manager|recipient)[\s\S]*?\bname\s*(?:is|:)\s*['\"]([^'\"]+)['\"]",
        normalized,
        re.IGNORECASE,
    )
    if not name_match:
        name_match = re.search(
            r"(?:manager|recipient)[\s\S]*?\bname\s*(?:is|:)\s*([A-Z][A-Za-z .-]*[A-Za-z])",
            normalized,
            re.IGNORECASE,
        )
    if name_match:
        name = name_match.group(1).strip()
    elif re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", normalized):
        fallback_names = re.findall(r"[A-Z][a-z]+\s+[A-Z][a-z]+", normalized)
        if fallback_names:
            name = fallback_names[0]

    reason = "leave request"
    date_match = re.search(r"from\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\s+to\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", normalized, re.IGNORECASE)
    if date_match:
        reason = f"Leave request from {date_match.group(1)} to {date_match.group(2)}"

    return UserDetails(
        mail_reason=reason,
        recipient_name=name,
        recipient_email=email,
    )


def retriever_node(state: EmailState) -> EmailState:
    llm = _get_llm()
    user_details = UserDetails()

    if llm is not None:
        try:
            llm_for_user_details = llm.with_structured_output(UserDetails)
            user_details = llm_for_user_details.invoke(f"Extract email details from: {state.question}")
        except Exception:
            user_details = _fallback_extract_user_details(state.question)
    else:
        user_details = _fallback_extract_user_details(state.question)

    state.recipient_name = user_details.recipient_name or "Manager"
    state.recipient_email = user_details.recipient_email or ""
    state.mail_reason = user_details.mail_reason or "Leave request"
    return state


def _fallback_draft_email(state: EmailState) -> DraftEmail:
    recipient = state.recipient_name or "Manager"
    email = state.recipient_email or ""
    reason = state.mail_reason or "leave request"
    subject = f"Request for Leave: {reason}"
    body = (
        f"Dear {recipient},\n\n"
        f"I hope you are doing well. I am writing to request leave for {reason}.\n"
        f"I would appreciate your approval for this request. Please let me know if you need any additional information.\n\n"
        f"Thank you for your consideration.\n\n"
        f"Best regards,\n"
        f"[Your Name]"
    )
    if email:
        body = body.replace("[Your Name]", f"[Your Name]\n{email}")
    return DraftEmail(subject=subject, body=body)


def draft_node(state: EmailState) -> EmailState:
    llm = _get_llm()
    prompt = f"""
    Write a professional email using these details:

    Recipient: {state.recipient_name}
    Recipient Email: {state.recipient_email}
    Reason: {state.mail_reason}
    Feedback on the previous draft: {state.feedback or "None"}

    Subject should be concise.
    Body should be under 200 words.
    Include greetings and a professional closing.
    Apply the user's feedback when revising the previous draft.
    """

    if llm is not None:
        try:
            draft_email_llm = llm.with_structured_output(DraftEmail)
            draft_email = draft_email_llm.invoke(prompt)
        except Exception:
            draft_email = _fallback_draft_email(state)
    else:
        draft_email = _fallback_draft_email(state)

    state.subject = draft_email.subject
    state.body = draft_email.body
    return state


def _safe_print(value: str) -> str:
    return value.encode("ascii", "replace").decode("ascii")


def review_node(state: EmailState) -> EmailState:
    print("\n", "-" * 50)
    print("Subject:", _safe_print(state.subject))
    print("Body:", _safe_print(state.body))
    print("-" * 50)

    response = interrupt({
        "message": "Approve this email or provide feedback for a revision.",
        "subject": state.subject,
        "body": state.body,
        "options": ["approve", "provide feedback"],
    })

    if isinstance(response, dict):
        if (
            response.get("approved") is True
            or response.get("positive") is True
            or response.get("decision") is True
        ):
            response = "approve"
        else:
            response = (
                response.get("response")
                or response.get("decision")
                or response.get("feedback")
                or ""
            )

    normalized_response = re.sub(r"[^\w\s]", "", str(response)).strip().lower()
    if normalized_response in {
        "true",
        "yes",
        "y",
        "approve",
        "i approve",
        "approved",
        "positive",
        "positive feedback",
        "send",
        "send it",
        "yes send it",
        "approved send",
        "looks good",
        "looks good to me",
        "this looks good",
        "this is good",
        "thats good",
        "looks great",
        "this looks great",
        "great",
        "perfect",
        "i like it",
        "lgtm",
    }:
        state.feedback = ""
    else:
        state.feedback = str(response).strip() or "Please revise the draft."
        state.feedback_count += 1

    return state


def router(state: EmailState) -> Literal["draft", "send", "cancel"]:
    if not state.feedback.strip():
        return "send"

    if state.feedback_count > MAX_FEEDBACK_ROUNDS:
        return "cancel"

    return "draft"


def cancel_node(state: EmailState) -> EmailState:
    state.response = "Email not sent because you exceeded the 3-feedback limit."
    return state


def send_node(state: EmailState) -> EmailState:
    "Send final email"
    res = send_mail(state.recipient_name, state.subject, state.body)
    state.response = res
    return state


graph = StateGraph(EmailState)
graph.add_node("retriever", retriever_node)
graph.add_node("draft", draft_node)
graph.add_node("send", send_node)
graph.add_node("cancel", cancel_node)
graph.add_node("review", review_node)

graph.add_edge(START, "retriever")
graph.add_edge("retriever", "draft")
graph.add_edge("draft", "review")
graph.add_conditional_edges("review", router)
graph.add_edge("send", END)
graph.add_edge("cancel", END)

final_graph = graph.compile(checkpointer=InMemorySaver())
