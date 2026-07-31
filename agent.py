from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

try:
    from langchain_classic.agents import (
        create_tool_calling_agent,
        AgentExecutor,
    )
except ImportError:
    from langchain_core.agents import (
        create_tool_calling_agent,
    )
    from langchain.agents import AgentExecutor
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_core.tools.retriever import (
    create_retriever_tool,
)
from pathlib import Path

from dotenv import load_dotenv
from database import (
    get_products_with_pricing,
    get_pricing_for_product,
    save_quote,
)

load_dotenv()

CHROMA_DIR = Path(__file__).parent / "chroma_istatis"
CHROMA_COLLECTION = "istatis_products"


def build_product_docs() -> list[str]:
    """
    Build RAG documents dynamically from
    the database instead of hardcoded strings.
    """
    products = get_products_with_pricing()
    docs = []

    # Group by category
    categories: dict[str, list] = {}
    for p in products:
        cat = p["categories"]["name"]
        categories.setdefault(cat, []).append(p)

    for cat_name, cat_products in categories.items():
        lines = [f"iStatis - {cat_name}:"]
        for p in cat_products:
            # DEMO MODE (2026-07-31): pricing tiers deliberately excluded
            # from RAG docs so the assistant cannot surface unit prices.
            # Revert by restoring the tier_str block from git history.
            lines.append(
                f"{p['name']}: Min order {p['min_order']}. "
                f"{p['description']}"
            )
        docs.append("\n".join(lines))

    # Company info doc
    docs.append(
        "iStatis - Company Information:\n"
        "Manufacturer and supplier of envelopes, "
        "paper, file carriers, registers and "
        "notebooks in Islamabad, Rawalpindi and "
        "Lahore, Pakistan.\n"
        "Major clients include Islamabad Diagnostic "
        "Center and Allama Iqbal University.\n"
        "Experience with government and NGO tenders "
        "including army and US-AID.\n"
        "Custom printing available on envelopes.\n"
        "Delivery available across Pakistan for "
        "bulk orders.\n"
        "B2B bank transfer payments accepted."
    )

    return docs


def build_vectorstore():
    """Build or load persisted vector store from database products."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION,
        )

    docs = build_product_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(docs)
    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION,
    )


# ── Tools ────────────────────────────────────────────────


@tool
def get_pricing_tier(
    product_name: str,
    quantity: int,
) -> str:
    """
    Use for any pricing question. Does not
    disclose unit pricing during demo mode.
    product_name: name of the product e.g.
    'C4 Envelope', 'A4 Paper 70gsm'
    quantity: number of units requested
    """
    # DEMO MODE (2026-07-31): deliberately does not call
    # get_pricing_for_product or return numeric pricing.
    # Revert to the real lookup from git history when done.
    return (
        f"Pricing for {product_name} depends on volume "
        "and current mill rates. I will have our sales "
        "team follow up with a formal quote."
    )


@tool
def calculate_order_cost(
    unit_price_pkr: float,
    quantity: int,
    discount_pct: float = 0,
) -> str:
    """
    Use when a customer asks for a final order total.
    Does not disclose numeric pricing during demo mode.
    """
    # DEMO MODE (2026-07-31): deliberately does not
    # compute or return a numeric total. Revert to the
    # real calculation from git history when done.
    return (
        "I do not have final order totals available "
        "right now. I will have our sales team send "
        "over a formal quote with the full breakdown."
    )


def build_agent():
    vectorstore = build_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(search_kwargs={"k": 2}),
        name="search_products",
        description=(
            "Search iStatis product catalogue "
            "for product info, specs, pricing, "
            "and company details."
        ),
    )

    tools = [
        retriever_tool,
        get_pricing_tier,
        calculate_order_cost,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional sales assistant "
                "for iStatis, a paper manufacturer "
                "in Islamabad, Pakistan. Be concise and "
                "helpful. Do not state or estimate any "
                "numeric price, rate, or PKR figure for "
                "any product, even if quoted in the "
                "conversation history or asked directly. "
                "If asked about pricing, say pricing "
                "depends on volume and that our sales "
                "team will follow up with a formal quote. "
                "If unsure about anything else, say you "
                "will check and follow up.",
            ),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
    )


_executor = None


def _get_executor():
    global _executor
    if _executor is None:
        _executor = build_agent()
    return _executor


def chat(
    message: str,
    history: list,
) -> tuple[str, list]:
    response = _get_executor().invoke({"input": message, "history": history})
    answer = response["output"]
    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=answer))
    return answer, history


async def generate_quote(
    customer_name: str,
    company: str,
    email: str,
    product_name: str,
    quantity: int,
    notes: str = "",
) -> dict:
    """
    Generate a professional quote and save
    to the database.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Get pricing from database
    pricing = get_pricing_for_product(product_name, quantity)

    if not pricing:
        return {"error": (f"Product '{product_name}' " "not found in catalogue.")}

    pricing_summary = (
        f"{pricing['product_name']} x "
        f"{pricing['quantity']:,} units = "
        f"PKR {pricing['price_per_unit']}/unit | "
        f"Total: PKR {pricing['total']:,.0f}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional sales "
                "representative for iStatis, a "
                "paper manufacturer in Islamabad, "
                "Pakistan. Generate a formal but "
                "friendly quote email body. Include: "
                "thank the customer by name, confirm "
                "product and quantity, state pricing "
                "clearly in PKR, mention delivery is "
                "available across Pakistan, ask them "
                "to reply to confirm the order, sign "
                "off as iStatis Sales Team. "
                "Keep it under 150 words.",
            ),
            (
                "human",
                "Customer: {name}\n"
                "Company: {company}\n"
                "Product: {product}\n"
                "Quantity: {quantity}\n"
                "Pricing: {pricing}\n"
                "Notes: {notes}",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    quote_text = await chain.ainvoke(
        {
            "name": customer_name,
            "company": company,
            "product": pricing["product_name"],
            "quantity": f"{quantity:,}",
            "pricing": pricing_summary,
            "notes": notes or "None",
        }
    )

    # Save to database
    saved = save_quote(
        client_name=customer_name,
        company=company,
        email=email,
        product_id=pricing["product_id"],
        quantity=quantity,
        unit_price=float(pricing["price_per_unit"]),
        total_price=float(pricing["total"]),
        quote_text=quote_text,
        notes=notes,
    )

    return {
        "quote_text": quote_text,
        "pricing_summary": pricing_summary,
        "customer_name": customer_name,
        "company": company,
        "email": email,
        "product_name": pricing["product_name"],
        "quantity": quantity,
        "quote_id": saved["id"],
    }
