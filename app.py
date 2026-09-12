import os
from typing import List, Dict

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# VIZA PILOT — GERMANY STUDENT VISA RAG MVP
# ============================================================
# This MVP intentionally supports one core journey:
# Pakistani applicant -> Germany -> Student Visa -> Ask questions
# -> Retrieve relevant PDF evidence -> Generate a grounded answer.
#
# The visa-guide knowledge below was extracted from the user's
# 9-page "Germany Visa Application Guide for Pakistani Applicants".
# Keeping it inside app.py lets this deployment use ONLY 3 files:
# app.py, requirements.txt, README.md.
# ============================================================

st.set_page_config(
    page_title="Viza Pilot — Germany Student Visa",
    page_icon="✈️",
    layout="wide",
)

APP_NAME = "Viza Pilot"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


# ------------------------------------------------------------
# KNOWLEDGE BASE
# ------------------------------------------------------------
# Each item preserves the source PDF page so answers can cite
# the guide accurately. Only student-visa/general information
# relevant to the student journey is included in the RAG index.
# ------------------------------------------------------------

KNOWLEDGE_BASE: List[Dict] = [
    {
        "page": 1,
        "section": "Guide scope and fraud warning",
        "text": (
            "Germany Visa Application Guide for Pakistani Applicants. The guide covers Student Visa "
            "and Short-Stay (Schengen) visas and states that it is based on official pages of the German "
            "Missions in Pakistan, including the Embassy Islamabad and Consulate General Karachi, published "
            "via pakistan.diplo.de. It says Germany's visa process is now almost entirely digital via the "
            "Consular Services Portal and the Videx online system. The guide warns that appointments cannot "
            "be booked over the phone or by unofficial agents and that only genuine confirmation emails from "
            "official portals are valid. Applicants should not pay third parties claiming they can guarantee "
            "an appointment slot."
        ),
    },
    {
        "page": 3,
        "section": "Visa type — Student National Visa",
        "text": (
            "The National (Long-Stay) Visa, category D — Student, is for enrolment in a German university or "
            "study programme. The guide states that it is applied for via the Consular Services Portal and "
            "leads to a German residence title after arrival."
        ),
    },
    {
        "page": 3,
        "section": "Islamabad vs Karachi jurisdiction",
        "text": (
            "German consular services in Pakistan are split between the Embassy in Islamabad and the Consulate "
            "General in Karachi. The Embassy Islamabad covers Islamabad Capital Territory, Gilgit-Baltistan, "
            "Khyber Pakhtunkhwa, Azad Jammu & Kashmir, and Punjab. The Consulate General Karachi covers Sindh "
            "and Balochistan. Student visa applications in both jurisdictions use the Consular Services Portal."
        ),
    },
    {
        "page": 3,
        "section": "Student visa fee",
        "text": (
            "The guide lists the fee for a long-term/national visa, category D, as EUR 75. Fees are payable in "
            "PKR in cash at the counter, calculated at the current exchange rate, at both the Islamabad Embassy "
            "and the Karachi Consulate-General."
        ),
    },
    {
        "page": 4,
        "section": "Processing time and remonstration",
        "text": (
            "National long-stay visa processing times vary widely depending on the purpose of stay and the "
            "completeness and quality of the documents. The guide also states that as of 1 July 2025 the voluntary "
            "objection or remonstration procedure for an initial visa refusal was permanently cancelled, so there "
            "is no informal review route through the Embassy after refusal."
        ),
    },
    {
        "page": 5,
        "section": "Student visa — Consular Services Portal procedure",
        "text": (
            "Germany's student visa process for Pakistan is described as fully digital through the Consular "
            "Services Portal. Step 1: register and complete the interactive questionnaire on the portal; appointment "
            "registrations are handled through the portal. Step 2: the required document list is generated "
            "automatically after the questionnaire based on the applicant's specific situation. Step 3: scan and "
            "upload all requested documents through the portal. Step 4: a visa officer performs an initial screening "
            "of the uploaded documents. Step 5: the applicant is then either sent an appointment notification or "
            "asked to submit further or corrected documents."
        ),
    },
    {
        "page": 5,
        "section": "Incomplete applications",
        "text": (
            "The guide warns that incomplete student visa applications directly slow down processing. It says the "
            "consular section is processing a high volume of student visa applications and that demand exceeds "
            "capacity. Applicants are asked to avoid unnecessary status follow-ups because repeated enquiries reduce "
            "the capacity available to process applications."
        ),
    },
    {
        "page": 5,
        "section": "Mandatory academic documents",
        "text": (
            "The guide warns of a known technical issue in which the online portal may mark some documents as "
            "optional even though they are mandatory. It identifies the following as mandatory: past degree "
            "certificates, such as school, Bachelor's or previous Master's degrees as applicable; degree transcripts; "
            "and proof of payment of tuition fees where applicable to the course of study. The guide recommends "
            "submitting these immediately rather than waiting for a reminder."
        ),
    },
    {
        "page": 5,
        "section": "Preparing student visa documents",
        "text": (
            "The guide advises applicants to prepare documents and confirmations well in advance, prepare clear and "
            "readable copies in A4 format, and sort documents in the order specified by the Consular Services Portal."
        ),
    },
    {
        "page": 5,
        "section": "Karachi student visa applicants",
        "text": (
            "The Consular Services Portal is active for Karachi as well. Applicants resident in Sindh and Balochistan "
            "submit Student Visa applications through the same online portal used for Islamabad-jurisdiction applicants."
        ),
    },
    {
        "page": 7,
        "section": "Student financial requirements — recognised routes",
        "text": (
            "For long-stay visas including student visas, the guide says showing sufficient funds to cover living costs "
            "is central to the application. It describes three recognised routes: a blocked bank account (Sperrkonto), "
            "a deed of obligation (Verpflichtungserklärung), or an accepted official scholarship."
        ),
    },
    {
        "page": 7,
        "section": "Blocked account amount",
        "text": (
            "The guide states that from 1 January 2025 the blocked-account amount is EUR 992 per month, or EUR 11,904 "
            "per year, for visa applications submitted from that date. It also says the Embassy or Consulate-General "
            "may require funds for up to two years in particular cases. The guide notes that blocked-account thresholds "
            "can change and should be verified before applying."
        ),
    },
    {
        "page": 7,
        "section": "Blocked account provider and attestation",
        "text": (
            "The guide says applicants may choose any blocked-account provider if it gives the authorities a login or "
            "verification code, or a direct contact for verification. It says the applicant's signature on account-opening "
            "documents must be attested by the Embassy Islamabad, Consulate-General Karachi, or Honorary Consul in Lahore. "
            "The applicant should bring a passport and a black-and-white copy of the signature page to the attestation "
            "appointment. The guide lists an attestation fee of EUR 25, approximately PKR 5,400 subject to exchange-rate "
            "changes, and says the attestation appointment and visa-submission appointment must be booked separately."
        ),
    },
    {
        "page": 7,
        "section": "Deed of obligation",
        "text": (
            "A sponsor residing in Germany can sign a declaration of liability, or Verpflichtungserklärung, under sections "
            "66–68 of the Residence Act at the local Migration Office. The sponsor must prove sufficient means, for example "
            "with six months of payslips and bank statements, and takes responsibility for costs the applicant might incur "
            "during the stay in Germany."
        ),
    },
    {
        "page": 7,
        "section": "Scholarships as financial proof",
        "text": (
            "The guide says a scholarship can demonstrate sufficient means, but a scholarship awarded by a Pakistani "
            "university alone is not sufficient. It states that scholarships awarded by the HEC, a German official "
            "scholarship body, or another foreign official scholarship body are accepted."
        ),
    },
    {
        "page": 8,
        "section": "Student visa required documents — quick reference",
        "text": (
            "The student-visa quick reference lists: core identification documents — passport and CNIC; application form — "
            "Consular Services Portal questionnaire completed online; academic documents — degree certificates, transcripts, "
            "tuition-fee payment proof, and admission letter; financial proof — blocked account at EUR 992 per month or "
            "EUR 11,904 per year, OR deed of obligation, OR HEC/official scholarship; insurance — health insurance appropriate "
            "to the applicant's stay in Germany; accommodation — proof of housing arrangements in Germany; family documents — "
            "as requested case-by-case."
        ),
    },
    {
        "page": 8,
        "section": "Application-day checklist",
        "text": (
            "The guide's application-day checklist includes: printed appointment confirmation email showing the registered "
            "email address; original passport or passports, current and all previous, signed by the holder; completed and signed "
            "application form, with a portal-generated set for student visas; two biometric photos no older than six months; the "
            "complete document set in the order specified by the portal or Embassy with copies in A4 format; visa fee in cash in "
            "PKR in the exact amount if possible; and a blocked-account attestation receipt or deed of obligation for long-stay or "
            "student applicants."
        ),
    },
    {
        "page": 8,
        "section": "Appointment and fraud warnings",
        "text": (
            "The guide states that appointments cannot be bought or guaranteed. Appointment bookings go through official systems. "
            "Applicants should be cautious of anyone offering to guarantee a faster appointment or visa outcome, should not disclose "
            "Consular Services Portal or Videx login credentials, should verify communications claiming to be from the Embassy or "
            "Consulate against the official diplo.de domain, and should report suspected fraud directly to the relevant mission."
        ),
    },
    {
        "page": 9,
        "section": "Official-source verification",
        "text": (
            "The guide says it was compiled from official pages of the German Missions in Pakistan. It specifically tells applicants "
            "to verify current fees, blocked-account amounts and procedural details directly on those pages before applying because "
            "requirements are updated periodically. The listed official source for student visas is the German Missions in Pakistan "
            "student-visa page on pakistan.diplo.de. The guide also makes clear that the guide itself is an independent informational "
            "summary and is not an official publication of the German Federal Foreign Office, its Missions in Pakistan, or an affiliated portal."
        ),
    },
]


SYSTEM_PROMPT = """
You are Viza Pilot, an AI visa-application co-pilot for Pakistani students applying to Germany.

You are operating inside a Retrieval-Augmented Generation (RAG) application. The user question is accompanied by passages retrieved from a visa guide.

STRICT GROUNDING RULES:
1. Answer factual visa questions ONLY from the RETRIEVED PDF CONTEXT supplied in the current request.
2. Do not invent, infer, silently add, or "complete" missing visa requirements from your general knowledge.
3. If the retrieved context is insufficient, say exactly: "I could not verify that from the Germany visa guide in Viza Pilot."
4. The guide is an independent informational summary based on German Missions in Pakistan sources; do not describe the guide itself as an official government publication.
5. For time-sensitive items such as fees, financial thresholds, procedures, or appointment arrangements, remind the user to verify the current value/procedure on the official German Missions in Pakistan source before applying.
6. Never guarantee approval and never present advice as an approval probability.
7. Explain confusing terminology in clear, friendly language.
8. Cite claims using the page markers that appear in the retrieved context, for example [PDF p. 7]. Do not invent page numbers.
9. If two retrieved passages conflict, point out the conflict instead of choosing silently.
10. Keep the answer focused on a Pakistani applicant applying for a German student visa unless the user clearly asks about something else.

When useful, structure the answer with a short direct answer followed by practical next steps.
"""


def get_secret(name: str):
    """Read Streamlit Secrets first, then fall back to environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Preparing the Viza Pilot knowledge base...")
def build_vector_store():
    model = load_embedding_model()
    texts = [item["text"] for item in KNOWLEDGE_BASE]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # Inner product + normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(question: str, top_k: int = TOP_K):
    model = load_embedding_model()
    index = build_vector_store()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    top_k = min(top_k, len(KNOWLEDGE_BASE))
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(KNOWLEDGE_BASE[idx])
        item["score"] = float(score)
        results.append(item)

    return results


def make_context(results: List[Dict]) -> str:
    blocks = []
    for result in results:
        blocks.append(
            f"[PDF p. {result['page']}] — {result['section']}\n"
            f"{result['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, retrieved: List[Dict], history: List[Dict]) -> str:
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Groq is not connected yet. Add GROQ_API_KEY in your Streamlit app Secrets."
        )

    model_name = get_secret("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    client = Groq(api_key=api_key)

    # Keep a small amount of conversation history for natural follow-up questions.
    prior_messages = []
    for msg in history[-6:]:
        if msg.get("role") in {"user", "assistant"}:
            prior_messages.append(
                {"role": msg["role"], "content": msg["content"]}
            )

    context = make_context(retrieved)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *prior_messages,
        {
            "role": "user",
            "content": (
                "RETRIEVED PDF CONTEXT:\n\n"
                f"{context}\n\n"
                "USER QUESTION:\n"
                f"{question}"
            ),
        },
    ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
        reasoning_effort="low",
    )

    return completion.choices[0].message.content


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session()

# Trigger vector-store creation once and cache it for the app session/server.
build_vector_store()


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.title("✈️ Viza Pilot")
st.subheader("Germany Student Visa Assistant for Pakistani Applicants")
st.caption("RAG MVP · Streamlit · FAISS · Groq")

with st.sidebar:
    st.markdown("## Application scope")
    st.write("**Nationality:** Pakistan")
    st.write("**Destination:** Germany")
    st.write("**Visa:** National Student Visa")

    st.divider()
    st.markdown("## Knowledge source")
    st.write("Germany Visa Application Guide for Pakistani Applicants")
    st.write("9-page guide supplied for this MVP")
    st.caption(
        "The guide describes itself as an independent informational summary based on German Missions in Pakistan sources."
    )

    st.divider()
    if get_secret("GROQ_API_KEY"):
        st.success("Groq API connected")
    else:
        st.warning("Groq API key not configured")

    st.caption(f"Groq model: {get_secret('GROQ_MODEL') or DEFAULT_GROQ_MODEL}")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.info(
    "Ask questions about the German student-visa process, documents, finances, jurisdiction, fees, or application-day preparation. "
    "Viza Pilot retrieves relevant passages from the supplied guide before Groq writes the answer."
)

with st.expander("Try an example question"):
    st.markdown(
        "- What documents do I need for a German student visa?\n"
        "- How does the Consular Services Portal process work?\n"
        "- How much money do I need in a blocked account?\n"
        "- I live in Lahore. Which German mission covers me?\n"
        "- What should I take with me on the application day?\n"
        "- Can an HEC scholarship be used as proof of finances?"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("See the PDF evidence used"):
                for source in message["sources"]:
                    st.markdown(
                        f"**PDF page {source['page']} — {source['section']}**"
                    )
                    st.caption(f"Retrieval similarity: {source['score']:.3f}")
                    st.write(source["text"])
                    st.divider()

question = st.chat_input("Ask Viza Pilot about the Germany student visa...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the visa guide..."):
                retrieved = retrieve(question)

            with st.spinner("Groq is preparing a grounded answer..."):
                answer = generate_answer(
                    question,
                    retrieved,
                    st.session_state.messages[:-1],
                )

            st.markdown(answer)

            with st.expander("See the PDF evidence used"):
                for source in retrieved:
                    st.markdown(
                        f"**PDF page {source['page']} — {source['section']}**"
                    )
                    st.caption(f"Retrieval similarity: {source['score']:.3f}")
                    st.write(source["text"])
                    st.divider()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved,
                }
            )

        except Exception as exc:
            error_message = f"I couldn't generate the answer right now. {exc}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )

st.divider()
st.caption(
    "Viza Pilot is an informational assistant. Visa rules can change. Always verify current requirements, fees and procedures with the responsible official German authority before submitting an application."
)
