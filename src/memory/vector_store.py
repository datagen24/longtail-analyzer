"""
Vector store for similarity search and embeddings using ChromaDB.

This module provides vector storage capabilities for pattern embeddings,
profile summaries, and similarity search functionality.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store implementation using ChromaDB.
    
    This class provides vector storage and similarity search capabilities
    for pattern embeddings, profile summaries, and attack clusters.
    """
    
    def __init__(self, db_path: str = "data/chroma", collection_name: str = "patterns"):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Attack patterns and profile embeddings"}
        )
        
        logger.info(f"VectorStore initialized with collection: {collection_name}")
    
    def add_pattern_embedding(
        self,
        pattern_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a pattern embedding to the vector store.
        
        Args:
            pattern_id: Unique identifier for the pattern
            embedding: Vector embedding of the pattern
            metadata: Additional metadata about the pattern
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert numpy array to list
            embedding_list = embedding.tolist()
            
            # Add to collection
            self.collection.add(
                ids=[pattern_id],
                embeddings=[embedding_list],
                metadatas=[metadata]
            )
            
            logger.debug(f"Added pattern embedding: {pattern_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding pattern embedding {pattern_id}: {e}")
            return False
    
    def add_profile_embedding(
        self,
        profile_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a profile embedding to the vector store.
        
        Args:
            profile_id: Unique identifier for the profile
            embedding: Vector embedding of the profile
            metadata: Additional metadata about the profile
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert numpy array to list
            embedding_list = embedding.tolist()
            
            # Add to collection
            self.collection.add(
                ids=[profile_id],
                embeddings=[embedding_list],
                metadatas=[metadata]
            )
            
            logger.debug(f"Added profile embedding: {profile_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding profile embedding {profile_id}: {e}")
            return False
    
    def search_similar_patterns(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar patterns using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of similar patterns to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (pattern_id, similarity_score, metadata) tuples
        """
        try:
            # Convert numpy array to list
            query_list = query_embedding.tolist()
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            similar_patterns = []
            if results['ids'] and results['ids'][0]:
                for i, pattern_id in enumerate(results['ids'][0]):
                    similarity = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    similar_patterns.append((pattern_id, similarity, metadata))
            
            logger.debug(f"Found {len(similar_patterns)} similar patterns")
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Error searching similar patterns: {e}")
            return []
    
    def search_similar_profiles(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        entity_type: Optional[str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar profiles using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of similar profiles to return
            entity_type: Optional entity type filter
            
        Returns:
            List of (profile_id, similarity_score, metadata) tuples
        """
        try:
            # Convert numpy array to list
            query_list = query_embedding.tolist()
            
            # Build filter
            filter_metadata = {}
            if entity_type:
                filter_metadata["entity_type"] = entity_type
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=top_k,
                where=filter_metadata if filter_metadata else None
            )
            
            # Format results
            similar_profiles = []
            if results['ids'] and results['ids'][0]:
                for i, profile_id in enumerate(results['ids'][0]):
                    similarity = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    similar_profiles.append((profile_id, similarity, metadata))
            
            logger.debug(f"Found {len(similar_profiles)} similar profiles")
            return similar_profiles
            
        except Exception as e:
            logger.error(f"Error searching similar profiles: {e}")
            return []
    
    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """
        Get an embedding by ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            Embedding vector or None if not found
        """
        try:
            results = self.collection.get(
                ids=[item_id],
                include=['embeddings']
            )
            
            if results['embeddings'] and results['embeddings'][0]:
                return np.array(results['embeddings'][0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting embedding {item_id}: {e}")
            return None
    
    def update_embedding(
        self,
        item_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing embedding.
        
        Args:
            item_id: ID of the item to update
            embedding: New embedding vector
            metadata: Optional new metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert numpy array to list
            embedding_list = embedding.tolist()
            
            # Update the item
            self.collection.update(
                ids=[item_id],
                embeddings=[embedding_list],
                metadatas=[metadata] if metadata else None
            )
            
            logger.debug(f"Updated embedding: {item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating embedding {item_id}: {e}")
            return False
    
    def delete_embedding(self, item_id: str) -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[item_id])
            logger.debug(f"Deleted embedding: {item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embedding {item_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_embeddings": count,
                "db_path": self.db_path
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all embeddings from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all IDs
            results = self.collection.get(include=['embeddings'])
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
