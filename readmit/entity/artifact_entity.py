"""
Artifact entities define OUTPUTS from each component
These flow from one component to the next in the pipeline
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionArtifact:
    """
    Output of Data Ingestion component
    
    Contains paths to ingested data files
    Next component (DataValidation) will use these paths
    """
    raw_data_path: Path
    train_file_path: Path
    test_file_path: Path