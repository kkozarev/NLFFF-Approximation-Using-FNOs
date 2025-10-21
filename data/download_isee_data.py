import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

def load_urls_from_file(file_path):
    """Read URLs from a text file, ignoring blank lines and comments."""
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    return urls


def download_files_from_pages(url_list, output_dir="downloads"):
    """
    For each URL in `url_list`, finds a 'Click to download' link and downloads the file.
    Shows dynamic progress bars and skips already-downloaded files.
    """
    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()

    total = len(url_list)
    print(f"📋 Found {total} pages to process.\n")

    for i, page_url in enumerate(url_list, start=1):
        print(f"\n[{i}/{total}] 🌐 Processing: {page_url}")
        try:
            # Fetch the page
            r = session.get(page_url, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Find the link containing 'click to download'
            link = soup.find("a", string=lambda s: s and "click to download" in s.lower())
            if not link:
                print("⚠️ No 'click to download' link found.")
                continue

            # Resolve relative file URL
            file_url = urljoin(page_url, link.get("href"))
            filename = os.path.basename(file_url.split("?")[0]) or f"download_{i}"
            file_path = os.path.join(output_dir, filename)

            # Skip if already downloaded
            if os.path.exists(file_path):
                print(f"⏭️ Skipping (already exists): {filename}")
                continue

            # Download with progress bar
            with session.get(file_url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total_size = int(resp.headers.get("content-length", 0))
                block_size = 8192  # 8 KB chunks
                t = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"⬇️ {filename}",
                    dynamic_ncols=True,
                    leave=False
                )

                with open(file_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            t.update(len(chunk))
                t.close()

            print(f"✅ Saved to {file_path}")

        except Exception as e:
            print(f"❌ Error processing {page_url}: {e}")


if __name__ == "__main__":
    # === Usage ===
    # Create a text file 'urls.txt' with one URL per line.
    # Then run:
    #   python download_from_urls.py

    url_file = "download_links.txt"          # Input file with URLs
    output_dir = "../ISEE_NLFFF_Data/downloads/"       # Folder for downloads

    urls = load_urls_from_file(url_file)
    download_files_from_pages(urls, output_dir)