import os


def sample_path(sample_id: str, output_directory: str) -> str:
    return os.path.join(output_directory, sample_id)
