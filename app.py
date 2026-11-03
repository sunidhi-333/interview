import os
import sys
import json
import argparse
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class FieldWithEvidence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    value: Optional[str] = Field(
        None, description="The extracted answer, or null/None if not found in the context."
    )
    source_quote: Optional[str] = Field(
        None, description="Verbatim clause or short quote from the text that proves this value."
    )
    source_chunk_ref: Optional[str] = Field(
        None, description="The identifier of the chunk/page where this quote was found."
    )


class TenderExtractionSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name_of_work: FieldWithEvidence = Field(
        ..., alias="name of work", description="Exact title or name of work."
    )
    tender_no: FieldWithEvidence = Field(
        ..., alias="tender no", description="Tender reference number, tender ID, or GeM bid ID."
    )
    ato_criteria: FieldWithEvidence = Field(
        ..., alias="ato_criteria", description="Annual Turnover Criteria (ATO/PQC financial requirement) with amounts and years."
    )
    similar_work_definition: FieldWithEvidence = Field(
        ..., alias="similar_work_definition", description="Exact definition of Similar Work or Similar Item."
    )
    documents_to_be_submitted: List[str] = Field(
        default_factory=list,
        alias="documents_to_be_submitted",
        description="Comprehensive list of required certificates, annexures, and declarations."
    )
    tech_evaluation_mode: Literal["L1", "QCBS", "OTHER", "NOT_SPECIFIED"] = Field(
        ...,
        alias="tech_evaluation_mode",
        description="Evaluation method: 'QCBS' if quality/cost scoring is used; 'L1' if lowest-cost/rate ranking; 'OTHER' or 'NOT_SPECIFIED' if unknown."
    )



def load_and_index_document(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"[*] Ingesting: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Must be .pdf or .docx.")

    docs = loader.load()
    print(f"[+] Loaded {len(docs)} page(s)/section(s).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\nCHAPTER", "\nSECTION", "\nAnnexure", "\n\n", "\n", " "]
    )
    splits = text_splitter.split_documents(docs)

    # Attach clean chunk identifiers to metadata
    for i, chunk in enumerate(splits):
        page = chunk.metadata.get("page", None)
        chunk.metadata["chunk_id"] = f"Chunk {i + 1}" + (f" (Page {page + 1})" if page is not None else "")

    print(f"[+] Created {len(splits)} indexed text chunks.")
    retriever = BM25Retriever.from_documents(splits)
    retriever.k = 4  # Increased k to capture distributed checklists
    return retriever



def extract_tender_details(file_path: str) -> dict:
    retriever = load_and_index_document(file_path)

    # Domain-agnostic queries (applicable to any tender domain)
    targeted_queries = [
        "name of work tender reference number GeM Bid ID title",
        "annual turnover criteria PQC ATO financial requirement audited balance sheet",
        "definition of similar work similar item technical commercial experience qualifying criteria",
        "documents to be submitted checklist certificates annexures required bid security EMD",
        "mode of evaluation criteria QCBS Quality Cost Based Selection L1 lowest quote price bid"
    ]

    print("[*] Running multi-clause retrieval...")
    retrieved_docs = []
    for q in targeted_queries:
        retrieved_docs.extend(retriever.invoke(q))

    # Deduplicate while preserving source metadata
    seen = set()
    formatted_chunks = []
    for d in retrieved_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            chunk_ref = d.metadata.get("chunk_id", "Unknown Ref")
            formatted_chunks.append(f"[{chunk_ref}]\n{d.page_content}")

    combined_context = "\n\n---\n\n".join(formatted_chunks)
    print(f"[+] Injected {len(formatted_chunks)} unique context blocks with source citations.")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0
    )

    parser = PydanticOutputParser(pydantic_object=TenderExtractionSchema)

    prompt = ChatPromptTemplate.from_template(
        """You are an expert procurement auditor and tender extraction engine.
Extract the required fields strictly from the provided context chunks.

INSTRUCTIONS:
1. Grounding & Evidence: For every field, provide the exact extracted value, an exact supporting quote from the context, and the chunk reference ID where the quote was found.
2. If any field is not explicitly stated in the context, return null for that field rather than assuming.
3. 'tech_evaluation_mode': Return 'QCBS' ONLY if Quality and Cost Based Selection is explicitly defined. Return 'L1' if evaluated on lowest cost / lowest gross rate. Otherwise return 'OTHER' or 'NOT_SPECIFIED'.
4. 'documents_to_be_submitted': Aggregate all certificates, annexures, and statutory declarations required across the context chunks.

{format_instructions}

Tender Context Chunks:
{context}
"""
    )

    print("[*] Running structured extraction via LLM...")
    chain = prompt | llm | parser

    result: TenderExtractionSchema = chain.invoke({
        "context": combined_context,
        "format_instructions": parser.get_format_instructions()
    })

    return result.model_dump(by_alias=True)


def main():
    parser = argparse.ArgumentParser(description="Auditable Tender RAG Extraction Engine")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="Sample 1.docx",
        help="Path to the input .pdf or .docx tender document (default: 'Sample 1.docx')"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional custom path to save output JSON. If omitted, uses '<input_stem>_extracted.json'."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: File '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        output_data = extract_tender_details(args.input)

        out_file = args.output
        if not out_file:
            stem = os.path.splitext(os.path.basename(args.input))[0]
            out_file = f"{stem}_extracted.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n[✔] Extraction complete! Result saved to: {os.path.abspath(out_file)}\n")
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"[!] Extraction failed: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()