import pandas as pd
import GEOparse
import tempfile
import time
import requests
from tqdm import tqdm
import os

def download_gse_with_retries(accession_id, destdir, retries=3, delay=5):
    """
    Downloads a GSE SOFT file with a progress bar and retry mechanism, then parses it.
    """
    # Construct the URL for the SOFT file on the NCBI GEO FTP server.
    series_prefix = accession_id[3:-3]
    url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{series_prefix}nnn/{accession_id}/soft/"
        f"{accession_id}_family.soft.gz"
    )
    local_filename = os.path.join(destdir, f"{accession_id}_family.soft.gz")

    for attempt in range(retries):
        try:
            print(f"Downloading from {url} (Attempt {attempt + 1}/{retries})...")
            # Stream the download to handle large files and show progress.
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size_in_bytes = int(r.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=accession_id)
                with open(local_filename, 'wb') as f:
                    for data in r.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()

                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    raise IOError("ERROR: Downloaded size did not match expected size.")
            
            print("\nDownload complete. Parsing file...")
            gse = GEOparse.get_GEO(filepath=local_filename, silent=True)
            print("Parsing successful.")
            return gse

        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {e}")
            if os.path.exists(local_filename):
                os.remove(local_filename)
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All download attempts failed.")
    return None

def fetch_data(accession_id):
    """
    Orchestrates the download and parsing of a GEO dataset.
    Returns the expression matrix and the GSE object for metadata.
    """
    # Use a temporary directory to store downloaded files, which is cleaned up automatically.
    with tempfile.TemporaryDirectory() as tmpdir:
        gse = download_gse_with_retries(accession_id, destdir=tmpdir)
        if gse is None:
            return None, None
        
        # Extract the data table from each sample (GSM) in the study.
        all_samples_data = [
            gsm.table[['ID_REF', 'VALUE']].rename(columns={'VALUE': gsm_name}).set_index('ID_REF')
            for gsm_name, gsm in gse.gsms.items() if not gsm.table.empty
        ]

        if not all_samples_data:
            print("Error: No data tables found in GEO samples.")
            return None, None

        # Concatenate all sample data into a single DataFrame (expression matrix).
        expression_matrix = pd.concat(all_samples_data, axis=1)
        print("Successfully built expression matrix.")
        return expression_matrix, gse

