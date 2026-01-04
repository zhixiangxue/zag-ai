"""
ChromaDB vector store implementation
"""

from typing import Any, Optional, Union
from zag.storages.vector.base import BaseVectorStore
from zag.schemas.base import BaseUnit
from zag.schemas.unit import TextUnit


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store implementation
    
    A concrete implementation using ChromaDB as the backend.
    
    Usage:
        >>> from zag.embedders import Embedder
        >>> embedder = Embedder.from_uri("openai://text-embedding-3-small")
        >>> store = ChromaVectorStore(
        ...     embedder=embedder,
        ...     collection_name="my_docs",
        ...     persist_directory="./chroma_db"
        ... )
        >>> store.add(units)
        >>> results = store.search("query text", top_k=5)
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        embedder: Optional['BaseEmbedder'] = None,
        text_embedder: Optional['BaseEmbedder'] = None,
        image_embedder: Optional['BaseEmbedder'] = None,
        **kwargs
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist data (None for in-memory)
            embedder: Single embedder for all content types (multimodal)
            text_embedder: Embedder specifically for text/table units
            image_embedder: Embedder specifically for image units
            **kwargs: Additional ChromaDB client parameters
        
        Note:
            See BaseVectorStore.__init__ for detailed embedder usage patterns
        """
        super().__init__(
            embedder=embedder,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            **kwargs
        )
        
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )
        
        # Initialize Chroma client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add(self, units: list[BaseUnit]) -> None:
        """
        Add units to Chroma vector store
        
        Supports mixed unit types (text, table, image) in a single call.
        The method automatically:
        1. Groups units by type
        2. Routes each group to appropriate embedder
        3. Batches embedding calls for efficiency
        4. Stores all units together
        
        Args:
            units: List of units to store (can be mixed types)
        """
        from zag.schemas.base import UnitType
        
        if not units:
            return
        
        # Group units by type for efficient batch embedding
        # text/table units share the same embedder, image units use separate one
        text_like_units = []  # TextUnit and TableUnit
        image_units = []      # ImageUnit
        
        for unit in units:
            if unit.unit_type == UnitType.IMAGE:
                image_units.append(unit)
            else:
                # text and table both use text_embedder
                text_like_units.append(unit)
        
        # Prepare collections for final storage
        all_ids = []
        all_embeddings = []
        all_metadatas = []
        all_documents = []
        
        # Process text-like units (text + table)
        if text_like_units:
            # Extract content for embedding
            contents = []
            for unit in text_like_units:
                if isinstance(unit, TextUnit):
                    contents.append(unit.content)
                else:
                    # For non-text units (e.g., TableUnit), convert to string
                    # TableUnit should already have text representation in content
                    contents.append(str(unit.content))
            
            # Get appropriate embedder and embed
            embedder = self._get_embedder_for_unit(text_like_units[0])
            embeddings = embedder.embed_batch(contents)
            
            # Collect data
            for unit, content, embedding in zip(text_like_units, contents, embeddings):
                all_ids.append(unit.unit_id)
                all_embeddings.append(embedding)
                all_documents.append(content)
                all_metadatas.append(self._extract_metadata(unit))
        
        # Process image units
        if image_units:
            # Extract image content
            image_contents = [unit.content for unit in image_units]
            
            # Get image embedder and embed
            embedder = self._get_embedder_for_unit(image_units[0])
            embeddings = embedder.embed_batch(image_contents)
            
            # Collect data
            for unit, content, embedding in zip(image_units, image_contents, embeddings):
                all_ids.append(unit.unit_id)
                all_embeddings.append(embedding)
                # For images, store a placeholder or empty string as document
                # (Chroma requires documents field)
                all_documents.append(f"[Image: {unit.unit_id}]")
                all_metadatas.append(self._extract_metadata(unit))
        
        # Add all units to Chroma in one batch
        self.collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            documents=all_documents
        )
    
    def _extract_metadata(self, unit: BaseUnit) -> dict:
        """
        Extract metadata from unit for storage
        
        Args:
            unit: Unit to extract metadata from
        
        Returns:
            Dictionary of metadata (Chroma compatible)
        """
        metadata = {
            "unit_type": unit.unit_type.value,  # Store enum value as string
            "source_doc_id": unit.source_doc_id or "",
        }
        
        # Add context_path if available
        if unit.metadata and unit.metadata.context_path:
            metadata["context_path"] = unit.metadata.context_path
        
        # Add custom metadata (only Chroma-compatible types)
        if unit.metadata and unit.metadata.custom:
            for key, value in unit.metadata.custom.items():
                # Chroma metadata values must be str, int, float, or bool
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"custom_{key}"] = value
        
        return metadata
    
    def search(
        self,
        query: Union[str, BaseUnit],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None
    ) -> list[BaseUnit]:
        """
        Search for similar units in Chroma
        
        Automatically routes query to appropriate embedder based on query type:
        - String query → text_embedder
        - TextUnit/TableUnit → text_embedder
        - ImageUnit → image_embedder
        
        Args:
            query: Query content (can be text string or Unit)
            top_k: Number of results to return
            filter: Optional metadata filters (Chroma where clause)
            
        Returns:
            List of matching units, sorted by similarity
        """
        from zag.schemas.base import UnitType
        
        # Determine query type and extract content
        if isinstance(query, str):
            # String query → use text_embedder
            query_content = query
            embedder = self.text_embedder
        elif isinstance(query, BaseUnit):
            # Unit query → route to appropriate embedder
            embedder = self._get_embedder_for_unit(query)
            
            if isinstance(query, TextUnit):
                query_content = query.content
            elif query.unit_type == UnitType.IMAGE:
                query_content = query.content  # bytes for image
            else:
                query_content = str(query.content)
        else:
            raise TypeError(
                f"Query must be str or BaseUnit, got {type(query).__name__}"
            )
        
        # Embed the query using appropriate embedder
        query_embedding = embedder.embed(query_content)
        
        # Search in Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        # Convert results to Units
        units = []
        if results['ids'] and results['ids'][0]:
            for i, unit_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                
                # Reconstruct unit based on type
                # Note: In production, you might want to store full unit data
                # or retrieve from a separate document store
                unit = TextUnit(
                    unit_id=unit_id,
                    content=document,
                    unit_type=metadata.get('unit_type', 'text')
                )
                
                # Restore metadata
                if 'context_path' in metadata:
                    unit.metadata.context_path = metadata['context_path']
                
                if 'source_doc_id' in metadata and metadata['source_doc_id']:
                    unit.source_doc_id = metadata['source_doc_id']
                
                units.append(unit)
        
        return units
    
    def delete(self, unit_ids: list[str]) -> None:
        """
        Delete units by IDs from Chroma
        
        Args:
            unit_ids: List of unit IDs to delete
        """
        if not unit_ids:
            return
        
        self.collection.delete(ids=unit_ids)
    
    def get(self, unit_ids: list[str]) -> list[BaseUnit]:
        """
        Get units by IDs from Chroma
        
        Args:
            unit_ids: List of unit IDs
            
        Returns:
            List of corresponding units
        """
        if not unit_ids:
            return []
        
        results = self.collection.get(ids=unit_ids)
        
        units = []
        if results['ids']:
            for i, unit_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                document = results['documents'][i]
                
                # Reconstruct unit
                unit = TextUnit(
                    unit_id=unit_id,
                    content=document,
                    unit_type=metadata.get('unit_type', 'text')
                )
                
                # Restore metadata
                if 'context_path' in metadata:
                    unit.metadata.context_path = metadata['context_path']
                
                if 'source_doc_id' in metadata and metadata['source_doc_id']:
                    unit.source_doc_id = metadata['source_doc_id']
                
                units.append(unit)
        
        return units
    
    def update(self, units: list[BaseUnit]) -> None:
        """
        Update existing units in Chroma
        
        Args:
            units: List of units to update
        """
        if not units:
            return
        
        # Chroma doesn't have a native update, so we delete and re-add
        unit_ids = [unit.unit_id for unit in units]
        self.delete(unit_ids)
        self.add(units)
    
    def clear(self) -> None:
        """
        Clear all vectors from Chroma collection
        """
        # Delete and recreate collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    @property
    def dimension(self) -> int:
        """
        Get vector dimension
        
        Returns the dimension of the primary embedder (text_embedder).
        Note: In multimodal mode, all embedders should have the same dimension.
        
        Returns:
            Vector dimension
        """
        # Use text_embedder as the primary embedder
        # (In multimodal mode, text_embedder == image_embedder)
        return self.text_embedder.dimension
    
    def count(self) -> int:
        """
        Get total number of units in collection
        
        Returns:
            Number of units stored
        """
        return self.collection.count()
