"""
Text compression extractor using LLM-based recursive compression
"""

import os
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken
import chak
from bs4 import BeautifulSoup


class CompressionExtractor:
    """
    Generic text compression extractor using LLM.

    This extractor compresses text to a target token count while preserving
    semantic meaning, numbers, and logical relationships. It uses:
    - Semantic chunking with overlap
    - Parallel chunk compression (ThreadPoolExecutor)
    - Recursive compression until target is reached
    - Automatic retry with exponential backoff

    Note: This is a generic compression framework. Domain-specific prompts
    (e.g., mortgage quick-check, shortlist rules) should be provided by
    the application layer through prompt_template parameter.

    Args:
        llm_uri: LLM URI in format: provider/model (e.g., "openai/gpt-4o-mini")
        api_key: API key for the LLM provider (defaults to env variable)
        encoding_name: Tiktoken encoding name (default: "cl100k_base")

    Example:
        >>> # Mortgage Quick-Check extraction
        >>> quick_check_prompt = '''Extract hard rejection criteria from:
        ... {text}
        ... Target tokens: {target_tokens}'''
        >>> 
        >>> extractor = CompressionExtractor("openai/gpt-4o-mini")
        >>> compressed = extractor.compress(
        ...     text=full_document,
        ...     prompt_template=quick_check_prompt,
        ...     target_tokens=2000
        ... )
    """

    def __init__(
        self,
        llm_uri: str,
        api_key: Optional[str] = None,
        encoding_name: str = "cl100k_base"
    ):
        self.llm_uri = llm_uri
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.encoding_name = encoding_name

        # Initialize tiktoken encoder
        self._encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self._encoder.encode(text))
    
    def _extract_html_tables(self, text: str) -> Tuple[str, List[str]]:
        """Extract HTML tables and replace with placeholders"""
        soup = BeautifulSoup(text, 'html.parser')
        tables = []
        
        for idx, table in enumerate(soup.find_all('table')):
            tables.append(str(table))
            placeholder = f"{{{{HTML_TABLE_{idx}}}}}"
            table.replace_with(placeholder)
        
        return str(soup), tables
    
    def _restore_html_tables(self, text: str, tables: List[str]) -> str:
        """Restore HTML tables from placeholders"""
        for idx, table in enumerate(tables):
            text = text.replace(f"{{{{HTML_TABLE_{idx}}}}}", table)
        return text

    def compress(
        self,
        text: str,
        prompt: str,
        target_tokens: int = 10000,
        chunk_size: int = 3000,
        overlap: int = 100,
        max_depth: int = 3,
        max_workers: int = 30,
        timeout: int = 120,
    ) -> str:
        """
        Compress a single text.

        Args:
            text: Text string to compress
            prompt: Compression prompt with {text} and {target_tokens} placeholders
            target_tokens: Target token count after compression
            chunk_size: Size of each chunk in tokens (for parallel processing)
            overlap: Overlap between chunks in tokens (for context continuity)
            max_depth: Maximum recursion depth
            max_workers: Maximum parallel workers for chunk compression
            timeout: LLM request timeout in seconds

        Returns:
            Compressed text
        """
        # Extract HTML tables before compression
        text_without_tables, tables = self._extract_html_tables(text)
        
        # Compress text without tables
        compressed = self._recursive_compress(
            text=text_without_tables,
            prompt=prompt,
            target_tokens=target_tokens,
            chunk_size=chunk_size,
            overlap=overlap,
            max_depth=max_depth,
            max_workers=max_workers,
            timeout=timeout,
            current_level=0
        )
        
        # Restore HTML tables
        return self._restore_html_tables(compressed, tables)

    def compress_batch(
        self,
        texts: list[str],
        prompt: str,
        target_tokens: int = 10000,
        chunk_size: int = 3000,
        overlap: int = 100,
        max_depth: int = 3,
        max_workers: int = 30,
        timeout: int = 120,
    ) -> list[str]:
        """
        Compress multiple texts in parallel with progress bar.

        Args:
            texts: List of text strings to compress
            prompt: Compression prompt with {text} and {target_tokens} placeholders
            target_tokens: Target token count after compression
            chunk_size: Size of each chunk in tokens (for parallel processing)
            overlap: Overlap between chunks in tokens (for context continuity)
            max_depth: Maximum recursion depth
            max_workers: Maximum parallel workers for chunk compression
            timeout: LLM request timeout in seconds

        Returns:
            List of compressed texts (same order as input)
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

        if not texts:
            return []

        results = [None] * len(texts)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Compressing {len(texts)} documents...", total=len(texts))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                for idx, text in enumerate(texts):
                    future = executor.submit(
                        self._recursive_compress,
                        text=text,
                        prompt=prompt,
                        target_tokens=target_tokens,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        max_depth=max_depth,
                        max_workers=max_workers,
                        timeout=timeout,
                        current_level=0
                    )
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    progress.update(task, advance=1)

        return results

    def _recursive_compress(
        self,
        text: str,
        prompt: str,
        target_tokens: int,
        chunk_size: int,
        overlap: int,
        max_depth: int,
        max_workers: int,
        timeout: int,
        current_level: int
    ) -> str:
        """Internal recursive compression implementation"""
        current_tokens = self.count_tokens(text)

        # Base case 1: Already below target
        if current_tokens <= target_tokens:
            return text

        # Base case 2: Max depth reached
        if current_level >= max_depth:
            print(f"⚠️  Max depth {max_depth} reached, stopping recursion")
            return text

        # Base case 3: Compression ratio too high (would lose too much quality)
        compression_ratio = target_tokens / current_tokens
        if compression_ratio > 0.9:
            print(
                f"⚠️  Compression ratio {compression_ratio:.2f} > 0.9, stopping to preserve quality")
            return text

        print(f"\n{'  ' * current_level}📊 Level {current_level + 1}: {current_tokens:,} → {target_tokens:,} tokens (ratio: {compression_ratio:.2f})")

        # Split into chunks for parallel processing
        chunks = self._split_into_chunks(
            text, chunk_size=chunk_size, overlap=overlap)
        print(f"{'  ' * current_level}📦 Split into {len(chunks)} chunks")

        # Calculate total tokens across all chunks
        chunk_tokens_list = [self.count_tokens(chunk) for chunk in chunks]
        total_chunk_tokens = sum(chunk_tokens_list)

        # Compress chunks in parallel
        compressed_chunks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            skipped_chunks = {}  # Separate dict for skipped chunks
            
            for idx, chunk in enumerate(chunks):
                chunk_tokens = chunk_tokens_list[idx]
                # Proportional target: (target_tokens / total_tokens) * chunk_tokens
                # Multiply by 0.8 to leave room for merging
                chunk_target = int((target_tokens / total_chunk_tokens) * chunk_tokens * 0.8)
                
                # Skip compression if chunk is already small enough
                if chunk_tokens <= chunk_target:
                    skipped_chunks[idx] = (chunk, chunk_tokens)
                else:
                    # Add small delay to avoid rate limit
                    import time
                    time.sleep(0.1)
                    future = executor.submit(
                        self._compress_with_llm,
                        chunk,
                        chunk_target,
                        chunk_tokens,
                        prompt,
                        timeout
                    )
                    future_to_idx[future] = (idx, chunk_tokens)

            # Collect results in order
            results = [None] * len(chunks)
            
            # Add skipped chunks directly
            for idx, (chunk, tokens) in skipped_chunks.items():
                results[idx] = chunk
                print(
                    f"{'  ' * current_level}  ✓ Chunk {idx + 1}/{len(chunks)}: {tokens:,} tokens (skipped, already small)")
            
            # Add compressed chunks
            for future in as_completed(future_to_idx):
                idx, original_tokens = future_to_idx[future]
                compressed_chunk = future.result()
                compressed_tokens = self.count_tokens(compressed_chunk)
                results[idx] = compressed_chunk
                print(
                    f"{'  ' * current_level}  ✓ Chunk {idx + 1}/{len(chunks)}: {original_tokens:,} → {compressed_tokens:,} tokens")

            compressed_chunks = results

        # Merge compressed chunks
        merged = '\n\n'.join(compressed_chunks)
        merged_tokens = self.count_tokens(merged)
        print(f"{'  ' * current_level}🔗 Merged: {merged_tokens:,} tokens")

        # Recursive compression if still above target
        if merged_tokens > target_tokens:
            print(f"{'  ' * current_level}🔄 Still above target, recursing...")
            return self._recursive_compress(
                text=merged,
                prompt=prompt,
                target_tokens=target_tokens,
                chunk_size=chunk_size,
                overlap=overlap,
                max_depth=max_depth,
                max_workers=max_workers,
                timeout=timeout,
                current_level=current_level + 1
            )

        return merged

    def _split_into_chunks(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Split text into chunks by tokens with overlap.

        Strategy:
        - Try to split at paragraph boundaries (double newline)
        - Maintain overlap between chunks for context continuity
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds chunk_size, force split
            if para_tokens > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split by sentences
                sentences = para.split('. ')
                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)
                    if current_tokens + sent_tokens > chunk_size:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [sent]
                        current_tokens = sent_tokens
                    else:
                        current_chunk.append(sent)
                        current_tokens += sent_tokens
                continue

            # Normal case: accumulate paragraphs
            if current_tokens + para_tokens > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Add overlap (last paragraph)
                    current_chunk = [current_chunk[-1],
                                     para] if len(current_chunk) > 0 else [para]
                    current_tokens = self.count_tokens(
                        '\n\n'.join(current_chunk))
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _compress_with_llm(
        self,
        text: str,
        target_tokens: int,
        current_tokens: int,
        prompt: str,
        timeout: int
    ) -> str:
        """
        Compress text chunk using LLM (with retry).

        Args:
            text: Text to compress
            target_tokens: Target token count
            current_tokens: Current token count
            prompt: Prompt template with {text} and {target_tokens} placeholders
            timeout: Request timeout in seconds

        Returns:
            Compressed text
        """
        # Create a fresh conversation for each compression to avoid context accumulation
        conv = chak.Conversation(self.llm_uri, api_key=self.api_key)

        # Format prompt with text and target_tokens
        formatted_prompt = prompt.format(
            text=text, target_tokens=target_tokens)

        # Call LLM
        response = conv.send(formatted_prompt, timeout=timeout)
        compressed = response.content.strip()

        return compressed
