"""
PDF Spanish → English Translator
=================================
Translates PDFs from Spanish to English while preserving layout.
Uses OpenRouter LLM with shared vocabulary context across pages/documents.

Usage:
    python pdf_translator.py <input.pdf> [output.pdf]
    python pdf_translator.py *.pdf          # Batch mode (shared context)
    python pdf_translator.py doc.pdf --model mistralai/mistral-7b-instruct

Requirements:
    pip install pdfplumber pymupdf requests

Setup:
    Set your OpenRouter API key as an environment variable:
        export OPENROUTER_API_KEY="sk-or-..."
    Or pass it with --api-key
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
import requests

# Load .env file if present
def _load_dotenv(env_path: str = ".env"):
    path = Path(env_path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

_load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default model - reliable for structured output
DEFAULT_MODEL = "minimax/minimax-m2.5"

# Smaller batches for better reliability
BATCH_SIZE = 8

# Minimum characters to bother translating (skip page numbers, single chars, etc.)
MIN_TRANSLATE_LENGTH = 3

# Patterns to skip translation entirely (numbers, dates, account numbers, etc.)
SKIP_PATTERNS = [
    r"^\s*$",                         # whitespace only
    r"^\d[\d\s,.\-/]*$",              # pure numbers / dates
    r"^\$[\d,.\s]+$",                 # dollar amounts
    r"^[A-Z]{2,}\d+$",               # codes like "ABC123"
    r"^\*+\d+$",                      # masked numbers like ***1234
    r"^[+\-]?\d[\d,.\s%]+$",         # percentages / amounts
]

# ─── Vocabulary Context Manager ────────────────────────────────────────────────

class VocabularyContext:
    """
    Maintains a shared translation memory across pages and documents.
    When the LLM translates a phrase, we store it and include it in future
    prompts so repeated terms (account names, branch names, product names)
    are always translated consistently.
    """

    def __init__(self):
        self.glossary: dict[str, str] = {}  # spanish -> english
        self._dirty_count = 0

    def update(self, pairs: dict[str, str]):
        """Add new translation pairs."""
        for es, en in pairs.items():
            es_clean = es.strip().lower()
            if es_clean and en.strip():
                if es_clean not in self.glossary:
                    self.glossary[es_clean] = en.strip()
                    self._dirty_count += 1

    def lookup(self, text: str) -> Optional[str]:
        """Return cached translation if available."""
        return self.glossary.get(text.strip().lower())

    def to_prompt_snippet(self, max_entries: int = 60) -> str:
        """Format the most recent glossary entries for injection into prompts."""
        if not self.glossary:
            return ""
        # Prioritize recently added entries (end of dict in Python 3.7+)
        items = list(self.glossary.items())[-max_entries:]
        lines = [f"  {es} → {en}" for es, en in items]
        return "TRANSLATION MEMORY (use these exact translations for consistency):\n" + "\n".join(lines)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.glossary, f, ensure_ascii=False, indent=2)
        print(f"  💾 Vocabulary context saved to {path}")

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.glossary = json.load(f)
            print(f"  📖 Loaded {len(self.glossary)} entries from {path}")


# ─── LLM Translation ──────────────────────────────────────────────────────────

class Translator:
    def __init__(self, api_key: str, model: str, context: VocabularyContext):
        self.api_key = api_key
        self.model = model
        self.context = context
        self.total_tokens = 0

    def _should_skip(self, text: str) -> bool:
        if len(text.strip()) < MIN_TRANSLATE_LENGTH:
            return True
        for pattern in SKIP_PATTERNS:
            if re.match(pattern, text.strip()):
                return True
        return False

    def translate_batch(self, texts: list[str]) -> list[str]:
        """
        Translate a batch of text blocks in one LLM call.
        Returns translated strings in the same order.
        """
        # Separate texts that need translation from those that don't
        indices_to_translate = []
        result = list(texts)

        for i, text in enumerate(texts):
            if self._should_skip(text):
                continue  # keep original
            cached = self.context.lookup(text)
            if cached:
                result[i] = cached
                continue
            indices_to_translate.append(i)

        if not indices_to_translate:
            return result

        # Build the batch payload as numbered list
        batch_lines = [f"{i}. {texts[i]}" for i in indices_to_translate]
        batch_text = "\n".join(batch_lines)
        vocab_snippet = self.context.to_prompt_snippet()

        system_prompt = f"""You are a professional financial document translator. Translate Spanish to English.

CRITICAL INSTRUCTIONS:
1. You will receive a numbered list of Spanish text items
2. Translate EACH item to English
3. Return ALL translations in a numbered list (same numbers)
4. One translation per line - DO NOT combine items
5. No explanations, no markdown, just the numbered translations

EXAMPLE:
Spanish input:
0. Estado de cuenta  
1. Saldo inicial
2. Fecha de cierre

English output:
0. Account statement
1. Opening balance  
2. Closing date

RULES:
- Translate every item, one per line
- Keep the same number format (0., 1., etc.)
- Don't translate proper nouns (names, codes)
- Use US banking terms

{vocab_snippet}"""

        user_prompt = f"Translate EACH of these Spanish items to English. Return ALL {len(indices_to_translate)} translations as a numbered list:\n\n{batch_text}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # low temp for consistency
            "max_tokens": 4096,
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    OPENROUTER_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/pdf-translator",
                    },
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                # Track token usage
                usage = data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)

                # Check for API errors in response
                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    raise ValueError(f"API error: {error_msg}")
                
                if "choices" not in data or not data["choices"]:
                    raise ValueError(f"No choices in response: {str(data)[:200]}")

                raw = data["choices"][0]["message"]["content"].strip()

                # Parse the numbered list response: INDEX. TRANSLATED_TEXT
                translated_batch = {}
                response_lines = []
                
                for line in raw.strip().split("\n"):
                    line = line.strip()
                    if not line or line.startswith("```"):
                        continue
                    response_lines.append(line)
                
                # Debug: show first response line briefly
                if response_lines:
                    print(f"  [LLM] {len(response_lines)}/{len(indices_to_translate)} lines, first: {repr(response_lines[0][:60])}")
                
                # Try to parse each line
                for i, line in enumerate(response_lines):
                    idx_str = None
                    translated_text = None
                    
                    # Match patterns like "0. Text" or just ". Text" (number missing)
                    # Also handle "0) Text" or "0: Text"
                    match = re.match(r'^(\d*)[:.)\s]+\s*(.+)$', line)
                    if match:
                        num_part = match.group(1)
                        text_part = match.group(2).strip()
                        if num_part:
                            idx_num = int(num_part)
                            # Check if this index is in our translate list
                            if idx_num in indices_to_translate:
                                idx_str = str(idx_num)
                                translated_text = text_part
                            elif i < len(indices_to_translate):
                                # Model used sequential numbering, map by position
                                idx_str = str(indices_to_translate[i])
                                translated_text = text_part
                        else:
                            # Number missing, use position
                            if i < len(indices_to_translate):
                                idx_str = str(indices_to_translate[i])
                                translated_text = text_part
                    else:
                        # No prefix at all - use position
                        if i < len(indices_to_translate):
                            idx_str = str(indices_to_translate[i])
                            translated_text = line
                    
                    # Add to batch if we have valid data
                    if idx_str is not None and translated_text is not None:
                        translated_batch[idx_str] = translated_text
                
                # Validation: we should get at least one translation
                if len(translated_batch) == 0:
                    print(f"  WARNING: No items parsed. Raw preview: {repr(raw[:200])}")
                    raise ValueError(f"No translations parsed from {len(response_lines)} response lines")

                if not translated_batch:
                    print(f"\n  Raw response (debug):\n{raw[:800]}\n")
                    raise ValueError("No valid translations found in response")

                # Merge back into result and update vocabulary
                new_pairs = {}
                for idx_str, translated_text in translated_batch.items():
                    idx = int(idx_str)
                    if idx in indices_to_translate:
                        original = texts[idx]
                        result[idx] = translated_text
                        new_pairs[original] = translated_text

                self.context.update(new_pairs)
                print(f"  [Batch] Translated {len(new_pairs)} items")
                return result

            except (ValueError, KeyError, IndexError) as e:
                print(f"  WARNING: Parse error on attempt {attempt + 1}: {e}, retrying...")
                time.sleep(2)
            except requests.HTTPError as e:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text
                print(f"  ⚠️  API error: {e}")
                print(f"      Response: {err_body}")
                time.sleep(2)

        # Fallback: return originals for this batch
        print("  ❌ Translation failed after 3 attempts, keeping originals.")
        return result


# ─── PDF Processing ────────────────────────────────────────────────────────────

def parse_font_style(font_name: str) -> tuple[bool, bool]:
    """
    Parse font name to determine if it's bold and/or italic.
    Returns (is_bold, is_italic)
    """
    name_lower = font_name.lower()
    is_bold = any(b in name_lower for b in ["bold", "heavy", "black", "demi"])
    is_italic = any(i in name_lower for i in ["italic", "oblique", "slant"])
    return is_bold, is_italic


def map_to_base14_font(font_name: str) -> str:
    """
    Map any font name to closest base14 font based on style.
    PyMuPDF base14 font names: helv, hebo, hebi, heit, cour, cobo, cobi, coit,
    tiRo, tiBo, tiBI, tiIt, symb, zadb
    """
    is_bold, is_italic = parse_font_style(font_name)

    # Map to Helvetica variants (most common)
    if is_bold and is_italic:
        return "hebi"  # Helvetica-BoldOblique
    elif is_bold:
        return "hebo"  # Helvetica-Bold
    elif is_italic:
        return "heit"  # Helvetica-Oblique
    else:
        return "helv"  # Helvetica


def extract_blocks(pdf_path: str) -> list[dict]:
    """
    Extract all text blocks with their positions using PyMuPDF.
    Returns list of dicts: {page, block_no, text, bbox, font_size, font_name, font_flags, color}
    """
    blocks = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    # Get font name and flags for style detection
                    font_name = span.get("font", "Helvetica")
                    font_flags = span.get("flags", 0)  # Bitmask for style

                    blocks.append({
                        "page": page_num,
                        "text": text,
                        "bbox": span["bbox"],       # (x0, y0, x1, y1)
                        "font_size": span["size"],
                        "font_name": font_name,
                        "font_flags": font_flags,
                        "color": span.get("color", 0),  # int RGB
                        "origin": span.get("origin", (span["bbox"][0], span["bbox"][3])),
                    })

    doc.close()
    return blocks


def translate_blocks(blocks: list[dict], translator: Translator) -> list[dict]:
    """Translate all text blocks in batches."""
    texts = [b["text"] for b in blocks]
    total = len(texts)
    translated_texts = []

    print(f"  Translating {total} text spans in batches of {BATCH_SIZE}...")
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        translated_batch = translator.translate_batch(batch)
        translated_texts.extend(translated_batch)
        done = min(i + BATCH_SIZE, total)
        print(f"  ✓ {done}/{total} spans translated", end="\r")

    print()  # newline after progress

    result = []
    for block, translated in zip(blocks, translated_texts):
        result.append({**block, "translated": translated})
    return result


def get_best_font_for_block(block: dict, src_doc: fitz.Document, page: fitz.Page,
                             font_cache: dict, embedded_fonts: dict) -> tuple[str, bytes | None]:
    """
    Determine the best font for a text block.

    Returns tuple of (font_name_for_redaction, font_bytes_for_embedded_font)
    font_bytes is None if using base14 fonts.
    """
    font_name = block.get("font_name", "Helvetica")
    font_flags = block.get("font_flags", 0)

    # Check if we already processed this font
    cache_key = f"{font_name}_{font_flags}"
    if cache_key in font_cache:
        return font_cache[cache_key]

    # First, try to find and extract the font from the source PDF
    font_xref = None
    for xref in range(1, src_doc.xref_length()):
        try:
            font_dict = src_doc.xref_get_key(xref, "Type")
            if font_dict[1] == "/Font":
                base_font = src_doc.xref_get_key(xref, "BaseFont")
                if base_font[0] == "name" and font_name in base_font[1]:
                    font_xref = xref
                    break
        except:
            continue

    # Try to extract embedded font if found
    if font_xref:
        try:
            font_data = src_doc.extract_font(font_xref)
            if font_data and font_data[-1]:  # font_data is tuple, last element is bytes
                font_bytes = font_data[-1]
                # Create a unique name for this embedded font
                embedded_font_name = f"EmbeddedFont_{font_xref}"
                # Store in embedded fonts dict for later embedding
                embedded_fonts[embedded_font_name] = font_bytes
                font_cache[cache_key] = (embedded_font_name, font_bytes)
                return font_cache[cache_key]
        except:
            pass

    # Try to find font in local fonts/ folder
    try:
        fonts_dir = Path("fonts")
        if fonts_dir.exists():
            # Clean font name for file matching
            clean_name = font_name.replace("+", "").replace("-", "").replace(" ", "").replace("MT", "").replace("PS", "").lower()

            for font_file in fonts_dir.iterdir():
                if font_file.suffix.lower() in ('.ttf', '.otf', '.ttc'):
                    file_clean = font_file.stem.lower().replace("-", "").replace("_", "").replace(" ", "").replace("mt", "").replace("ps", "")
                    # Check if font names match
                    if clean_name in file_clean or file_clean in clean_name:
                        with open(font_file, 'rb') as f:
                            font_bytes = f.read()
                        local_font_name = f"LocalFont_{font_file.stem}"
                        embedded_fonts[local_font_name] = font_bytes
                        font_cache[cache_key] = (local_font_name, font_bytes)
                        return font_cache[cache_key]
    except:
        pass

    # Try to find system font with matching name
    try:
        # Common system font paths
        system_paths = [
            "/System/Library/Fonts",  # macOS
            "/Library/Fonts",  # macOS
            "C:/Windows/Fonts",  # Windows
            "/usr/share/fonts",  # Linux
        ]

        # Clean font name for file matching
        clean_name = font_name.replace("+", "").replace("-", "").replace(" ", "")

        for sys_path in system_paths:
            if os.path.exists(sys_path):
                for root, dirs, files in os.walk(sys_path):
                    for f in files:
                        if f.lower().endswith(('.ttf', '.otf', '.ttc')):
                            file_clean = f.lower().replace("-", "").replace("_", "").replace(" ", "")
                            if clean_name.lower() in file_clean or file_clean in clean_name.lower():
                                font_path = os.path.join(root, f)
                                with open(font_path, 'rb') as font_file:
                                    font_bytes = font_file.read()
                                # Create unique name
                                sys_font_name = f"SysFont_{clean_name}"
                                embedded_fonts[sys_font_name] = font_bytes
                                font_cache[cache_key] = (sys_font_name, font_bytes)
                                return font_cache[cache_key]
    except:
        pass

    # Fallback to base14 font based on style
    base_font = map_to_base14_font(font_name)
    font_cache[cache_key] = (base_font, None)
    return font_cache[cache_key]


def calculate_text_width(text: str, font_name: str, font_bytes: bytes | None, font_size: float) -> float:
    """Calculate the width of text in points given font and size."""
    try:
        if font_bytes:
            font = fitz.Font(fontbuffer=font_bytes)
        else:
            font = fitz.Font(fontname=font_name)
        return font.text_length(text, fontsize=font_size)
    except:
        # Fallback: estimate using average char width
        return len(text) * font_size * 0.5


def rebuild_pdf(original_path: str, translated_blocks: list[dict], output_path: str):
    """
    Replace text in PDF with translations using redaction.
    Preserves original font, size, and style.
    Strategy:
      1. Extract original fonts or find closest match
      2. Calculate text width and expand bbox if needed to preserve font size
      3. Add redaction annotations over Spanish text (removes it completely)
      4. Insert English text with original font/size/style
    """
    src_doc = fitz.open(original_path)
    replacement_count = 0

    # Group blocks by page
    by_page: dict[int, list[dict]] = {}
    for block in translated_blocks:
        by_page.setdefault(block["page"], []).append(block)

    # Font cache to avoid re-processing same fonts
    font_cache: dict[str, tuple[str, bytes | None]] = {}

    for page_num in range(len(src_doc)):
        page = src_doc[page_num]
        blocks_on_page = by_page.get(page_num, [])

        if not blocks_on_page:
            continue

        # Dictionary to hold embedded fonts for this page (font_name -> temp file path)
        embedded_fonts: dict[str, bytes] = {}
        font_temp_files: dict[str, str] = {}  # Track temp files for cleanup

        # First pass: determine fonts needed and embed them
        for block in blocks_on_page:
            font_name, font_bytes = get_best_font_for_block(
                block, src_doc, page, font_cache, embedded_fonts
            )

        # Embed all custom fonts BEFORE creating redactions
        for font_name, font_bytes in embedded_fonts.items():
            try:
                if font_bytes:
                    # Write font bytes to temp file for embedding
                    import tempfile
                    suffix = ".ttf" if font_bytes[:4] == b"\x00\x01\x00\x00" else ".otf"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(font_bytes)
                        temp_path = tmp.name
                    font_temp_files[font_name] = temp_path
                    page.insert_font(fontname=font_name, fontfile=temp_path)
                else:
                    # Base14 font, no embedding needed
                    pass
            except Exception as e:
                print(f"  ⚠️  Could not embed font {font_name}: {e}")

        # Second pass: create redaction annotations (just to remove text, no fill)
        blocks_to_replace = []  # Store blocks for later text insertion
        for block in blocks_on_page:
            original_text = block["text"]
            translated_text = block["translated"]

            if original_text == translated_text:
                continue  # Nothing changed, skip

            replacement_count += 1

            bbox = fitz.Rect(block["bbox"])
            font_size = block["font_size"]
            color_int = block["color"]

            # Convert int color to RGB tuple (0.0–1.0)
            r = ((color_int >> 16) & 0xFF) / 255.0
            g = ((color_int >> 8) & 0xFF) / 255.0
            b = (color_int & 0xFF) / 255.0

            # Get the best font for this block (already cached from first pass)
            font_name, font_bytes = get_best_font_for_block(
                block, src_doc, page, font_cache, embedded_fonts
            )

            # Store for text insertion after redactions are applied
            blocks_to_replace.append({
                "bbox": bbox,
                "text": translated_text,
                "font_name": font_name,
                "font_bytes": font_bytes,
                "font_size": font_size,
                "color": (r, g, b),
                "origin": block.get("origin", (bbox[0], bbox[3])),
            })

            # Create redaction annotation WITHOUT text (just removes original)
            page.add_redact_annot(bbox, fill=False)

        # Apply all redactions to remove original text
        page.apply_redactions()

        # Re-embed fonts AFTER redactions (they get lost during apply_redactions)
        for font_name, temp_path in font_temp_files.items():
            try:
                page.insert_font(fontname=font_name, fontfile=temp_path)
            except Exception as e:
                print(f"  ⚠️  Could not re-embed font {font_name}: {e}")

        # Third pass: insert translated text at exact position with exact font size
        for block_data in blocks_to_replace:
            try:
                # Use insert_text for exact control (no auto-scaling)
                text_x = block_data["origin"][0]
                text_y = block_data["origin"][1]

                # Use the font name (already re-embedded after redactions)
                font_name = block_data["font_name"]
                font_bytes = block_data["font_bytes"]

                page.insert_text(
                    point=(text_x, text_y),
                    text=block_data["text"],
                    fontsize=block_data["font_size"],  # Exact original size!
                    fontname=font_name,
                    color=block_data["color"],
                )
            except Exception as e:
                print(f"  ⚠️  Could not insert text: {e}")

        # Clean up temp font files for this page
        import os
        for temp_path in font_temp_files.values():
            try:
                os.unlink(temp_path)
            except:
                pass

    src_doc.save(output_path, garbage=4, deflate=True)
    src_doc.close()
    print(f"  📄 Saved: {output_path} ({replacement_count} blocks replaced)")


# ─── Main ──────────────────────────────────────────────────────────────────────

def process_pdf(
    input_path: str,
    output_path: str,
    translator: Translator,
):
    print(f"\n📂 Processing: {input_path}")
    blocks = extract_blocks(input_path)
    print(f"  Found {len(blocks)} text spans across {max(b['page'] for b in blocks) + 1 if blocks else 0} page(s)")

    if not blocks:
        print("  ⚠️  No text found. Is this a scanned PDF? OCR not supported in this mode.")
        return

    translated = translate_blocks(blocks, translator)
    rebuild_pdf(input_path, translated, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Translate Spanish PDF bank statements to English, preserving layout."
    )
    parser.add_argument("inputs", nargs="+", help="Input PDF file(s)")
    parser.add_argument("-o", "--output", help="Output path (single file mode only)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument(
        "--context-file",
        default="translation_context.json",
        help="Path to save/load vocabulary context (default: translation_context.json)",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Disable vocabulary context (don't load or save)",
    )
    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OpenRouter API key required.")
        print("   Set it with:  export OPENROUTER_API_KEY='sk-or-...'")
        print("   Or pass:      --api-key sk-or-...")
        sys.exit(1)

    # Vocabulary context
    context = VocabularyContext()
    if not args.no_context:
        context.load(args.context_file)

    translator = Translator(api_key=api_key, model=args.model, context=context)

    # Process file(s)
    input_files = args.inputs
    if len(input_files) == 1:
        inp = input_files[0]
        out = args.output or str(Path(inp).with_stem(Path(inp).stem + "_english"))
        process_pdf(inp, out, translator)
    else:
        if args.output:
            print("⚠️  --output is ignored in batch mode. Output files named <original>_english.pdf")
        for inp in input_files:
            out = str(Path(inp).with_stem(Path(inp).stem + "_english"))
            process_pdf(inp, out, translator)

    # Save updated context
    if not args.no_context:
        context.save(args.context_file)

    print(f"\n✅ Done! Total tokens used: {translator.total_tokens:,}")
    print(f"   Vocabulary context size: {len(context.glossary):,} entries")


if __name__ == "__main__":
    main()
