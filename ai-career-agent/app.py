import os
import streamlit as st

from io import BytesIO
from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 1.PAGE CONFIGURATION


st.set_page_config(
    page_title="AI Career Agent",
    layout="wide"
)

st.title("AI Career Agent")

# 2.RESPONSE TEXT HELPER

def get_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts)

    return str(content)

# 3.GEMINI MODEL

@st.cache_resource
def get_model():

    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. ")

    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.3
    )


model = get_model()

# 4.RESUME TEXT EXTRACTION

def extract_resume_text(uploaded_file):

    pdf = PdfReader(
        BytesIO(uploaded_file.getvalue())
    )

    pages = []

    for page in pdf.pages:

        page_text = page.extract_text() or ""
        pages.append(page_text)

    resume_text = "\n".join(pages).strip()

    if not resume_text:

        raise ValueError("No readable text was found in the PDF. Please upload a text-based PDF.")

    return resume_text

# 5.RESUME ANALYZER

resume_prompt = """
You are an expert Resume Analyzer and Career Consultant.

Analyze the resume provided by the user.

Provide the following sections:

1. Resume Summary
   - Briefly summarize the candidate's profile.

2. Technical Skills
   - Identify technical skills mentioned in the resume.

3. Soft Skills
   - Identify soft skills supported by the resume.

4. Education
   - Summarize educational qualifications.

5. Work Experience and Projects
   - Evaluate relevant experience and projects.

6. Strengths
   - Highlight the candidate's strengths.

7. Areas for Improvement
   - Identify missing information and weak descriptions.

8. ATS Recommendations
   - Suggest improvements for ATS readability.

9. Career Recommendations
   - Suggest suitable job roles based on the resume.

IMPORTANT:
- Do not invent skills or qualifications.
- Base your analysis on the supplied resume.
- Clearly mention when information is missing.
- Give practical and specific recommendations.
"""


def analyze_resume(resume_text):

    response = model.invoke([
        SystemMessage(content=resume_prompt),
        HumanMessage(
            content=f"Analyze this resume:\n\n{resume_text}"
        )
    ])

    return get_text(response.content)

# 6.LANGCHAIN TOOLS

@tool
def create_learning_roadmap(
    target_role: str,
    current_skills: str
) -> str:
    " Create a structured learning roadmap for the user's target career role. "

    prompt = f"""
    Create a practical learning roadmap.

    Target role: {target_role}

    Current skills:
    {current_skills}

    Organize the roadmap into:

    - Week number
    - Topics to learn
    - Practical tasks
    - Mini projects

    Include beginner-friendly learning resources
    or resource types.

    Make the roadmap realistic and actionable.
    """

    response = model.invoke(prompt)

    return get_text(response.content)


@tool
def analyze_skill_gap(
    target_role: str,
    current_skills: str
) -> str:
    " Analyze the user's current skills and identify missing skills for their target career role."

    prompt = f"""
    Perform a detailed skill-gap analysis.

    Target role: {target_role}

    Current skills:
    {current_skills}

    Explain:

    1. Skills the user already has.
    2. Important missing skills.
    3. Skills to prioritize.
    4. Suggested practical projects.
    5. Recommended next steps.

    Do not assume the user has skills
    that were not provided.
    """

    response = model.invoke(prompt)

    return get_text(response.content)


tools = [
    create_learning_roadmap,
    analyze_skill_gap
]

# 7.CAREER AGENT SYSTEM PROMPT


system_prompt = """
You are an intelligent AI Career Agent.

Your purpose is to help users with:

1. Career guidance
2. Resume-related questions
3. Skill-gap analysis
4. Learning roadmaps
5. Interview preparation
6. Career planning
7. Technical skill development

INSTRUCTIONS:

- Give clear and practical answers.
- Use headings and bullet points.
- Explain technical concepts simply.
- Provide personalized recommendations when
  the user's resume is available.
- Use the create_learning_roadmap tool when
  the user requests a learning roadmap.
- Use the analyze_skill_gap tool when
  the user requests a skill-gap analysis.
- Do not invent user qualifications or experience.
- Do not claim to search live job listings.
- Answer general career questions even if
  the user has not uploaded a resume.
- If a resume is available in the provided context,
  use it instead of asking the user to paste it.
"""

# 8.LANGGRAPH SINGLE-AGENT WORKFLOW

@st.cache_resource
def build_agent():

    # Connecting Gemini with the tools.
    model_with_tools = model.bind_tools(tools)

    # Main AI agent node.
    def call_agent(state: MessagesState):

        response = model_with_tools.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    # Creating the graph.
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_agent)
    workflow.add_node("tools",ToolNode(tools))

    # Start with the AI agent.
    workflow.add_edge(START,"agent")

    # Decide whether to execute a tool or finish.
    workflow.add_conditional_edges("agent", tools_condition)

    # Return tool results to the same AI agent.
    workflow.add_edge("tools","agent")

    return workflow.compile()


agent = build_agent()


# 9.SIDEBAR NAVIGATION

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "About",
        "Resume Analyzer",
        "Career Agent"
    ]
)

# 10.ABOUT SECTION

if page == "About":

    st.header("About AI Career Agent")

    st.write("""
    AI Career Agent is an AI-powered career assistant
    designed to help individuals understand their career
    opportunities and improve their professional skills.

    ### Features

    - Resume analysis
    - Technical and soft skill identification
    - Resume improvement suggestions
    - Skill-gap analysis
    - Personalized learning roadmaps
    - Interview preparation
    - Career guidance
    """)

# 11.RESUME ANALYZER SECTION

elif page == "Resume Analyzer":

    st.header("Resume Analyzer")

    st.write(
        "Upload your resume in PDF format "
        "to receive an AI-powered analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf"],
        key="resume_uploader"
    )

    if uploaded_file:

        if st.button("Analyze Resume"):

            try:

                with st.spinner("Extracting and analyzing your resume..."):

                    # Extracting text from PDF.
                    resume_text = extract_resume_text(uploaded_file)

                    # Save resume text in session state.
                    st.session_state["resume_text"] = resume_text

                    # Analyze the resume.
                    analysis = analyze_resume(resume_text)

                    st.session_state["resume_analysis"] = analysis

                st.success("Resume analysis completed successfully!")

            except Exception as error:

                st.error(f"Resume analysis failed: {error}"  )


    # Displaying the analysis if it exists.
    if "resume_analysis" in st.session_state:

        st.subheader("Resume Analysis")

        st.markdown(st.session_state["resume_analysis"])

    # resume context is available.
    if "resume_text" in st.session_state:

        st.success("Your extracted resume is available to the Career Agent in this session.")

# 12. CAREER AGENT SECTION

elif page == "Career Agent":

    st.header("Career Agent")

    st.write( "Ask questions about your career, skills, learning roadmap or interview preparation.")

    # Initialize chat history.
    if "messages" not in st.session_state:

        st.session_state["messages"] = []

    # Getting the resume from session state.
    resume_context = st.session_state.get("resume_text", "")


    # resume availability.
    if resume_context:

        st.success( "Resume detected! Your Career Agent can now use your uploaded resume." )

    else:

        st.info("No resume uploaded in this session.You can still ask general career questions.")

    # Displaying previous chat messages.
    for message in st.session_state["messages"]:

        if isinstance(message, HumanMessage):
            role = "user"
        else:
            role = "assistant"

        with st.chat_message(role):

            st.markdown(
                get_text(message.content)
            )

    # Chat input.
    user_input = st.chat_input("Ask your Career Agent...")

    if user_input:

        # Displaying user message.
        with st.chat_message("user"):

            st.markdown(user_input)

        # Adding user message to session history.
        st.session_state["messages"].append(
            HumanMessage(content=user_input)
        )

        try:

            with st.chat_message("assistant"):

                with st.spinner(
                    "Your Career Agent is thinking..."
                ):

                    # Preparing the system prompt.
                    if resume_context:

                        updated_system_prompt = f"""
                        {system_prompt}

                        RESUME INFORMATION:

                        The user has uploaded a resume.

                        Use the following extracted resume
                        information to personalize your answers.

                        {resume_context[:12000]}

                        IMPORTANT:
                        - Do not ask the user to paste the resume.
                        - Refer to actual skills and experience.
                        - Do not invent missing information.
                        """

                    else:

                        updated_system_prompt = f"""
                        {system_prompt}

                        No resume has been uploaded yet.

                        You can still answer general career
                        questions. Ask for resume information
                        only when it is necessary.
                        """

                    # Preparing messages for LangGraph.
                    messages = [
                        SystemMessage(
                            content=updated_system_prompt
                        )
                    ] + st.session_state["messages"]


                    # executing the LangGraph agent.
                    result = agent.invoke(
                        {
                            "messages": messages
                        }
                    )

                    # Getting the final AI response.
                    final_message = result["messages"][-1]

                    answer = get_text(
                        final_message.content
                    )

                    if not answer.strip():
                        answer = (
                            "I couldn't generate a text response. Please try asking your question again.")

                    # Displaying the response.
                    st.markdown(answer)

                    # Saving the assistant response in session history.
                    st.session_state["messages"].append(
                        AIMessage(content=answer)
                    )

        except Exception as error:

            st.error(
                f"Career Agent error: {error}"
            )

            # Removing the failed user message from history.
            if st.session_state["messages"]:

                if isinstance(
                    st.session_state["messages"][-1],
                    HumanMessage
                ):

                    st.session_state["messages"].pop()

