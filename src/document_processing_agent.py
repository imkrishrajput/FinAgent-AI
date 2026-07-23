"""
document_processing_agent.py
=============================
Document Processing Agent for the FinAgent AI multi-agent finance system.

Role (strictly scoped):
    - Detect the uploaded document type (Google Pay / PhonePe / Paytm / CSV / Unknown).
    - Extract all transactions, preserving order.
    - Normalize every transaction into the common FinAgent schema.
    - Return ONLY structured JSON.

This agent NEVER:
    - Performs financial analysis.
    - Gives budgeting advice.
    - Answers finance-related questions.
    - Hallucinates or invents transactions/amounts.

Missing fields are returned as null rather than guessed.

Intended usage in your project: feed it text already extracted by
csv_reader.py / pdf_reader.py / file_reader_agent.py, and hand its JSON
output downstream to transaction_extractor.py / analyzer.py.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from model_manager import chat

SYSTEM_PROMPT = """You are Document Processing Agent in the FinAgent AI multi-agent finance system.
Your role is ONLY to process financial documents.
Responsibilities:
1. Detect the uploaded document type.
2. Identify whether it is:
   - Google Pay Statement
   - PhonePe Statement
   - Paytm Statement
   - CSV Transaction File
   - Unknown Document
3. Extract all transactions from the document.
4. Preserve transaction order.
5. Normalize every transaction into a common structure.
6. Never perform financial analysis.
7. Never provide budgeting advice.
8. Never answer finance-related questions.
9. If information is missing, return null instead of guessing.
10. Return only structured JSON.
For every transaction extract:
- Date
- Time (if available)
- Description
- Merchant
- Transaction Type
- Amount
- Currency
- Reference Number (if available)
- Payment Method (UPI/Card/Wallet/Bank)
- Source Document
Transaction Type must be one of:
- Income
- Expense
- Transfer
- Refund
- Unknown
Return JSON in this format:
{
  "document_type": "",
  "provider": "",
  "transactions": [
    {
      "date": "",
      "time": "",
      "merchant": "",
      "description": "",
      "amount": 0,
      "currency": "INR",
      "transaction_type": "",
      "payment_method": "",
      "reference_number": "",
      "source": ""
    }
  ]
}
Rules:
- Never hallucinate.
- Never invent transactions.
- Never modify amounts.
- Preserve original values.
- Return empty list if no transactions exist.
Respond with ONLY the JSON object. No preamble, no markdown fences, no commentary."""

VALID_TRANSACTION_TYPES = {"Income", "Expense", "Transfer", "Refund", "Unknown"}
VALID_DOCUMENT_TYPES = {
    "Google Pay Statement",
    "PhonePe Statement",
    "Paytm Statement",
    "CSV Transaction File",
    "Unknown Document",
}


class DocumentProcessingAgent:
    """
    Wraps a call to Claude with a locked system prompt so this agent can
    only ever detect + extract + normalize transactions, never analyze
    or advise.

    Optional RAG support (LangChain/FAISS) can be enabled via the
    `enable_rag` flag. When enabled this agent will build a local
    FAISS index of chunks and augment per-chunk prompts with retrieved
    context to improve extraction quality on large documents.
    """

    def __init__(self, *, enable_rag: bool = False, rag_index_path: Optional[str] = None, embedding_model: str = "all-MiniLM-L6-v2", max_workers: int = 4, rag_k: int = 3) -> None:
        """Initialize the DocumentProcessingAgent.

        Args:
            enable_rag: If True, attempt to use Agents.retriever.VectorStoreManager to
                index chunks and augment prompts. If dependencies are missing the
                code will raise a clear error when used.
            rag_index_path: Optional filesystem path to persist the FAISS index and metadata.
            embedding_model: Sentence-transformers model name used for embeddings.
            max_workers: Number of worker threads for parallel chunk processing when RAG is enabled.
            rag_k: Number of nearest neighbours to retrieve per chunk when RAG is enabled.
        """
        self.enable_rag = enable_rag
        self.rag_index_path = rag_index_path
        self.embedding_model = embedding_model
        self.max_workers = max_workers
        self.rag_k = rag_k

        # Lazy import to avoid hard dependency at module import time
        self._retriever = None
        if self.enable_rag:
            try:
                from retriever import VectorStoreManager
                # Initialize but do not build index yet
                self._retriever = VectorStoreManager(model_name=self.embedding_model, index_path=self.rag_index_path)
            except Exception as e:
                # Provide a helpful message — user can disable RAG in config
                raise RuntimeError(f"RAG requested but failed to initialize VectorStoreManager: {e}")

    

    def _split_document(self, text: str, max_chars: int = 7000, overlap_lines: int = 2) -> List[str]:
        """
        Split large documents into smaller chunks so the LLM
        doesn't exceed its output token limit.

        This uses line boundaries and a small overlap to avoid cutting a
        transaction in half between chunks.
        """
        if not text:
            return []

        lines = text.splitlines(keepends=True)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for line in lines:
            line_length = len(line)

            if current_length + line_length > max_chars and current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines else []
                current_length = sum(len(part) for part in current_chunk)

            if line_length > max_chars:
                # Break a single extremely long line into smaller pieces.
                start = 0
                while start < line_length:
                    piece = line[start : start + max_chars]
                    if current_chunk:
                        chunks.append("".join(current_chunk))
                        current_chunk = current_chunk[-overlap_lines:] if overlap_lines else []
                        current_length = sum(len(part) for part in current_chunk)
                    current_chunk.append(piece)
                    current_length += len(piece)
                    start += max_chars
                continue

            current_chunk.append(line)
            current_length += line_length

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def process(self, document_text: str, filename: str = "", source_hint: str = "", save_raw: bool = False, raw_output_dir: str = "data/debug_chunks") -> Dict[str, Any]:
        """
        Process raw document text (already extracted from CSV/PDF/etc.)
        and return normalized transaction JSON.

        This implementation supports two modes:
        - Classic chunking (default): call the LLM sequentially on each chunk.
        - RAG-enabled mode (enable_rag=True): build a local FAISS index of chunks
          and augment per-chunk prompts with retrieved context. Chunks are
          processed in parallel using a ThreadPoolExecutor.

        Args:
            document_text: Raw text content of the statement/CSV.
            filename: Original filename, if known (helps type detection).
            source_hint: Optional hint about origin (e.g. "csv_reader.py output").

        Returns:
            Dict matching the FinAgent transaction schema.
        """
        if not document_text or not document_text.strip():
            return self._empty_result(error="No document content provided.")

        # Split the document into smaller chunks
        chunks = self._split_document(document_text)

        if not chunks:
            return self._empty_result(error="No content after splitting into chunks.")

        # If RAG is enabled, build an index of the chunks for retrieval
        if self.enable_rag and self._retriever is not None:
            try:
                docs = [{"text": c, "meta": {"chunk_index": i}} for i, c in enumerate(chunks)]
                self._retriever.build_index(docs)
            except Exception as e:
                # If building the index fails, surface friendly error and fall back
                print(f"Warning: failed to build RAG index: {e}. Falling back to non-RAG mode.")
                self.enable_rag = False

        all_transactions: List[Dict[str, Any]] = []
        document_type = "Unknown Document"
        provider = None
        chunk_errors: List[str] = []

        # Helper to process a single chunk (can be called in parallel)
        def _process_single_chunk(idx: int, chunk_text: str) -> Tuple[int, Dict[str, Any], Optional[str]]:
            """Return (idx, parsed_result, error_message).

            parsed_result is the normalized dict or an empty result on error.
            error_message is None on success, otherwise a string describing the error.
            """
            # Build the user message and include retrieved context if available
            user_message = self._build_user_message(chunk_text, filename, source_hint, idx, len(chunks))

            if self.enable_rag and self._retriever is not None:
                try:
                    retrieved = self._retriever.query(chunk_text, k=self.rag_k)
                    if retrieved:
                        retrieved_lines = ["Retrieved context from nearby chunks:"]
                        for r in retrieved:
                            meta = r.get("meta", {})
                            score = r.get("score")
                            # try to include chunk index if available
                            ci = meta.get("chunk_index") if isinstance(meta, dict) else None
                            retrieved_lines.append(f"- Chunk {ci} (score={score:.3f})")
                        # Prepend retrieved context to user message to help the model
                        user_message = "\n".join(retrieved_lines) + "\n\n" + user_message
                except Exception as e:
                    # Retrieval failure should not abort processing
                    print(f"Warning: retrieval failed for chunk {idx}: {e}")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]

            try:
                raw_text = chat(messages=messages, temperature=0, max_tokens=4096)
            except Exception as exc:
                return idx, {}, f"LLM error while processing chunk {idx}/{len(chunks)}: {exc}"

            # Optionally save raw LLM output for debugging and tuning
            if save_raw:
                try:
                    import os
                    os.makedirs(raw_output_dir, exist_ok=True)
                    safe_fn = f"chunk_{idx:04d}.txt"
                    with open(os.path.join(raw_output_dir, safe_fn), "w", encoding="utf-8") as f:
                        f.write(raw_text)
                except Exception as e:
                    # Don't fail the whole process for debug write errors
                    print(f"Warning: failed to write raw chunk output for chunk {idx}: {e}")

            parsed = self._parse_and_validate(raw_text, filename)
            if parsed.get("error"):
                return idx, parsed, f"Chunk {idx}: {parsed['error']}"

            return idx, parsed, None

        # If RAG enabled, use a thread pool to process chunks in parallel; otherwise sequential
        if self.enable_rag:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            futures = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                for i, ch in enumerate(chunks, start=1):
                    futures[ex.submit(_process_single_chunk, i, ch)] = i

                for fut in as_completed(futures):
                    try:
                        idx, parsed, err = fut.result()
                        if err:
                            chunk_errors.append(err)
                        else:
                            if parsed.get("document_type") != "Unknown Document":
                                document_type = parsed["document_type"]
                            if parsed.get("provider"):
                                provider = parsed["provider"]
                            all_transactions.extend(parsed.get("transactions", []))
                    except Exception as e:
                        chunk_errors.append(f"Unexpected error processing a chunk: {e}")
        else:
            # Sequential fallback (original behavior)
            for index, chunk in enumerate(chunks, start=1):
                idx, parsed, err = _process_single_chunk(index, chunk)
                if err:
                    chunk_errors.append(err)
                    continue
                if parsed.get("document_type") != "Unknown Document":
                    document_type = parsed["document_type"]
                if parsed.get("provider"):
                    provider = parsed["provider"]
                all_transactions.extend(parsed.get("transactions", []))

        unique_transactions = self._deduplicate_transactions(all_transactions)

        if not unique_transactions and chunk_errors:
            return self._empty_result(error="; ".join(chunk_errors))

        return {
            "document_type": document_type,
            "provider": provider,
            "transactions": unique_transactions,
        }

    def _build_user_message(
        self,
        document_text: str,
        filename: str,
        source_hint: str,
        chunk_index: int,
        total_chunks: int,
    ) -> str:
        header_lines = []
        if filename:
            header_lines.append(f"Source filename: {filename}")
        if source_hint:
            header_lines.append(f"Source hint: {source_hint}")
        if total_chunks > 1:
            header_lines.append(f"Chunk {chunk_index} of {total_chunks}")

        header = ("\n".join(header_lines) + "\n\n") if header_lines else ""

        return (
            f"{header}"
            "Below is the raw content of a financial document chunk. "
            "Detect its type, extract every transaction in order from this chunk only, "
            "and normalize it per your instructions. Return ONLY the JSON object.\n\n"
            "--- DOCUMENT START ---\n"
            f"{document_text}\n"
            "--- DOCUMENT END ---"
        )

    @staticmethod
    def _transaction_signature(txn: Dict[str, Any]) -> Tuple[Any, ...]:
        fields = [
            "date",
            "time",
            "merchant",
            "description",
            "amount",
            "currency",
            "transaction_type",
            "payment_method",
            "reference_number",
        ]

        def normalize(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, str):
                return value.strip().lower()
            if isinstance(value, float):
                return round(value, 2)
            if isinstance(value, int):
                return float(value)
            return value

        return tuple(normalize(txn.get(field)) for field in fields)

    def _deduplicate_transactions(
        self,
        transactions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        unique_transactions: List[Dict[str, Any]] = []

        for txn in transactions:
            signature = self._transaction_signature(txn)
            if signature in seen:
                continue
            seen.add(signature)
            unique_transactions.append(txn)

        return unique_transactions

    def _parse_and_validate(self, raw_text: str, filename: str) -> Dict[str, Any]:

        try:

            # Remove markdown code fences
            cleaned = raw_text.replace("```json", "")
            cleaned = cleaned.replace("```", "")
            cleaned = cleaned.strip()

            # Find JSON object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)

            if not match:
                return self._empty_result(error="No JSON found in model response.")

            cleaned = match.group(0)

            # Try parsing with error recovery
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as je:
                # If JSON is incomplete (truncated), log warning and attempt recovery
                print(f"WARNING: JSON Parsing Error: {je}")
                print(f"Response may be truncated. Attempting partial parse...")
                
                # Try to salvage what we can from partial JSON
                # Count open/close braces to find incomplete structures
                open_braces = cleaned.count('{')
                close_braces = cleaned.count('}')
                
                if open_braces > close_braces:
                    # Add missing closing braces
                    cleaned += ']' * max(0, cleaned.count('[') - cleaned.count(']')) + '}' * (open_braces - close_braces)
                    try:
                        data = json.loads(cleaned)
                    except:
                        return self._empty_result(error=f"JSON parse failed even with recovery: {str(je)}")
                else:
                    return self._empty_result(error=str(je))

            return self._normalize(data, filename)

        except Exception as e:
            print("JSON Parsing Error:", e)
            print(cleaned if 'cleaned' in locals() else raw_text)

            return self._empty_result(error=str(e))

    def _normalize(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        document_type = data.get("document_type") or "Unknown Document"
        if document_type not in VALID_DOCUMENT_TYPES:
            document_type = "Unknown Document"

        result: Dict[str, Any] = {
            "document_type": document_type,
            "provider": data.get("provider"),
            "transactions": [],
        }

        for txn in data.get("transactions") or []:
            if not isinstance(txn, dict):
                continue

            txn_type = txn.get("transaction_type")
            if txn_type not in VALID_TRANSACTION_TYPES:
                txn_type = "Unknown"

            result["transactions"].append({
                "date": txn.get("date"),
                "time": txn.get("time"),
                "merchant": txn.get("merchant"),
                "description": txn.get("description"),
                "amount": txn.get("amount"),
                "currency": txn.get("currency") or "INR",
                "transaction_type": txn_type,
                "payment_method": txn.get("payment_method"),
                "reference_number": txn.get("reference_number"),
                "source": txn.get("source") or filename or None,
            })

        return result

    @staticmethod
    def _empty_result(error: Optional[str] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "document_type": "Unknown Document",
            "provider": None,
            "transactions": [],
        }
        if error:
            result["error"] = error
        return result


if __name__ == "__main__":
    # Quick manual smoke test — wire this into file_reader_agent.py /
    # csv_reader.py output in your real pipeline.
    sample_csv_text = """Date,Description,Amount,Type
2024-01-05,UPI/Swiggy Order/REF123456,-450.00,Debit
2024-01-06,Salary Credit - Acme Corp,55000.00,Credit
2024-01-07,Electricity Bill Payment,-1200.50,Debit
"""
    agent = DocumentProcessingAgent()
    
    output = agent.process(sample_csv_text, filename="transactions.csv")
    print(json.dumps(output, indent=2))