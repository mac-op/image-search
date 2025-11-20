import asyncio
import io
import os
import uuid
from random import randint

import asyncpraw
import aiohttp
import aiofiles
import json
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from queue import Empty

from gcloud.aio.storage import Storage
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _is_single_image_post(submission) -> Optional[str]:
    """Check if post has exactly one image and return URL."""
    if hasattr(submission, 'post_hint') and submission.post_hint == 'image':
        url = submission.url
        if url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            return url

    if hasattr(submission, 'is_gallery') and submission.is_gallery:
        return None

    url = submission.url
    if any(domain in url for domain in ['i.redd.it', 'i.imgur.com', 'imgur.com']):
        if 'imgur.com' in url and not url.startswith('https://i.imgur.com'):
            img_id = url.split('/')[-1].split('.')[0]
            url = f"https://i.imgur.com/{img_id}.jpg"

        if url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')) or 'i.imgur.com' in url:
            return url

    return None

class PrawImageScraper:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        output_dir: str = "downloaded_images",
        metadata_file: str = "metadata.json",
        max_concurrent_downloads: int = 5
    ):
        """
        Initialize the async Reddit scraper.

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string
            output_dir: Directory to save images
            metadata_file: JSON file to track downloaded posts
            max_concurrent_downloads: Maximum simultaneous downloads
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.metadata_file = Path(metadata_file)
        self.metadata = self._load_metadata()

        # Track downloaded posts
        self.downloaded_ids: Set[str] = set(self.metadata.keys())

        # Semaphore for concurrent downloads
        self.download_semaphore = asyncio.Semaphore(max_concurrent_downloads)

    def _load_metadata(self) -> Dict:
        """Load metadata from file or create new."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted metadata file, starting fresh")
                return {}
        return {}

    async def _save_metadata(self):
        """Save metadata to file asynchronously."""
        async with aiofiles.open(self.metadata_file, 'w') as f:
            await f.write(json.dumps(self.metadata, indent=2))

    async def fetch_posts(
        self,
        subreddits: dict[str, int],
        time_filter: str = 'year',
    ) -> List[Dict]:
        """Fetch posts from subreddits matching criteria."""
        all_posts = []

        async with asyncpraw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            ratelimit_seconds=300
        ) as reddit:

            for subreddit_name, min_upvotes in subreddits:
                logger.info(f"Fetching posts from r/{subreddit_name}")

                try:
                    subreddit = await reddit.subreddit(subreddit_name)

                    async for submission in subreddit.top(time_filter=time_filter):
                        if submission.id in self.downloaded_ids:
                            continue

                        if submission.score < min_upvotes:
                            continue

                        image_url = _is_single_image_post(submission)
                        if not image_url:
                            continue

                        post_data = {
                            'id': submission.id,
                            'title': submission.title,
                            'subreddit': subreddit_name,
                            'score': submission.score,
                            'url': submission.url,
                            'image_url': image_url,
                            'author': str(submission.author),
                            'created_utc': submission.created_utc,
                            'permalink': f"https://reddit.com{submission.permalink}",
                            'num_comments': submission.num_comments
                        }

                        all_posts.append(post_data)
                        logger.info(f"Found post: {submission.title[:50]}... ({submission.score} upvotes)")

                except Exception as e:
                    logger.error(f"Error accessing r/{subreddit_name}: {e}")
                    continue

        logger.info(f"Found {len(all_posts)} total posts matching criteria")
        return all_posts

    async def download_image(
        self,
        session: aiohttp.ClientSession,
        image_url: str,
        post_id: str,
        post_data: Dict
    ) -> Optional[Dict]:
        """Download image asynchronously and return job data."""
        async with self.download_semaphore:
            ext = image_url.split('.')[-1].split('?')[0].lower()
            if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                ext = 'jpg'

            filename = f"{post_id}.{ext}"
            filepath = self.output_dir / filename

            if filepath.exists():
                logger.info(f"Image already exists: {filename}")
                return {
                    'filepath': str(filepath),
                    'metadata': post_data
                }

            try:
                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    response.raise_for_status()

                    async with aiofiles.open(filepath, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)

                logger.info(f"Downloaded: {filename}")

                self.metadata[post_id] = {
                    **post_data,
                    'downloaded_at': datetime.now().isoformat(),
                    'local_path': str(filepath)
                }
                self.downloaded_ids.add(post_id)
                await self._save_metadata()

                return {
                    'filepath': str(filepath),
                    'metadata': post_data
                }

            except Exception as e:
                logger.error(f"Failed to download {image_url}: {e}")
                return None

    async def download_and_queue(
        self,
        posts: List[Dict],
        work_queue: mp.Queue
    ):
        """Download images and add to multiprocess queue."""
        async with aiohttp.ClientSession(
            headers={'User-Agent': self.user_agent}
        ) as session:

            tasks = []
            for post in posts:
                task = self.download_image(
                    session,
                    post['image_url'],
                    post['id'],
                    post
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Add to multiprocess queue
            queued_count = 0
            for result in results:
                if isinstance(result, dict) and result is not None:
                    work_queue.put(result)
                    queued_count += 1
                elif isinstance(result, Exception):
                    logger.error(f"Error in download: {result}")

            logger.info(f"Queued {queued_count} images for processing")

            # Signal completion with sentinel values (one per worker)
            return queued_count

class HTTPImageScraper:
    def __init__(self, max_concurrent_downloads: int = 5,):
        self.download_semaphore = asyncio.Semaphore(max_concurrent_downloads)

    async def fetch_posts(self, subreddits: dict[str, int], time_filter: str = 'week'):
        all_posts = []

        for subreddit_name, min_upvotes in subreddits.items() :
            logger.info(f"Fetching posts from r/{subreddit_name}")
            async with aiohttp.ClientSession(
                headers={'User-Agent': 'kotlin:com.httq.imgdb:v0.1.0 (by /u/_character_name)'}
            ) as session:
                try:
                    async with session.get(
                        f"https://www.reddit.com/r/{subreddit_name}/top.json?t={time_filter}&limit=100",
                        timeout=aiohttp.ClientTimeout(total=200)
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        posts = data['data']['children']
                        for post in posts:
                            data = post['data']
                            if data['score'] < min_upvotes:
                                break
                            all_posts.append({
                                'id': data['id'],
                                'title': data['title'],
                                'subreddit': subreddit_name,
                                'score': data['score'],
                                'url': data['permalink'],
                                'image_url': data['url'],
                                'author': data['author'],
                            })
                except Exception as e:
                    logger.error(f"Error accessing r/{subreddit_name}: {e}")
                    continue
            logger.info(f"Found {len(all_posts)} posts in r/{subreddit_name}")
            logger.info(f"All posts: {[post['id'] for post in all_posts]}")
        return all_posts

    async def download_image(
        self,
        session: aiohttp.ClientSession,
        post: Dict[str, any],
        client: Storage
    ) -> Optional[Image.Image]:
        """Download image asynchronously and return job data."""
        url = post['image_url']
        ext = url.split('.')[-1].split('?')[0].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            return None

        async with self.download_semaphore:
            async with session.get(post['image_url'], timeout=aiohttp.ClientTimeout(total=100)) as response:
                if response.status == 403 or response.status == 404:
                    return None
                response.auto_decompress = False
                buffer = io.BytesIO(await response.read())
                img = Image.open(buffer)
                img = img.convert('RGB')
            async with client:
                await client.upload("reddit-im-search-data", "images/" + post['id'], buffer.getvalue(), session=session)
                await client.upload("reddit-im-search-data", post['id'], json.dumps(post).encode('utf-8'), session=session)
        return img

    async def download_and_queue(
        self,
        posts: List[Dict[str, any]],
        work_queue: mp.Queue,
    ):
        """Download images and add to multiprocess queue."""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
            'android:app/com.httq.acio/v0.0.2 (by /u/_character_name)',
        ]
        async with Storage(service_file=os.environ.get("GCP_JSON")) as client:
            count = 0
            for post in posts:
                async with aiohttp.ClientSession(
                    headers={
                        'User-Agent': user_agents[randint(1, 80) % len(user_agents)],
                        'Connection': 'keep-alive',
                        # 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    }
                ) as session:
                    result = await self.download_image(session, post, client)
                    await asyncio.sleep(0.8)
                    if result:
                        work_queue.put((result, post['id']))
                        count += 1
            return count

def ml_processor_worker(
    work_queue: mp.Queue,
    worker_id: int,
    batch_size: int = 4,
    batch_timeout: float = 2.0,
    status_queue: mp.Queue = None,
):
    """
    ML processor worker with batched inference (runs in separate process).
    Handles CPU/GPU intensive ML inference and uploads to GCS.

    Args:
        work_queue: Queue to receive image processing jobs
        worker_id: Unique worker identifier
        batch_size: Number of images to batch for inference
        batch_timeout: Max seconds to wait for full batch
        status_queue: Queue to report worker status/errors back to main process
    """
    # Setup logging for this process
    logger = logging.getLogger(f"Worker-{worker_id}")
    logger.info(f"Worker {worker_id} started with batch_size={batch_size}")

    try:
        import torch
        from models import get_siglip_preprocessor, get_compiled_siglip_vision, load_blip_preprocessor
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor, to_pil_image
        import numpy as np
        from qdrant_client import AsyncQdrantClient
        import os
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        siglip_preprocessor = get_siglip_preprocessor()
        siglip_vis = get_compiled_siglip_vision()
        blip_preprocessor = load_blip_preprocessor()

        from gcloud.aio.storage import Storage
        qdrant = AsyncQdrantClient(url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY"))
        logger.info(f"Worker {worker_id} initialized ML model and GCS client")
    except Exception as e:
        logger.error(f"Worker {worker_id} initialization failed: {e}")
        if status_queue:
            status_queue.put({"worker_id": worker_id, "status": "initialization_failed", "error": str(e)})
        return

    async def upload_batch_results(processed_batch: tuple[np.ndarray, list[Image.Image], list[str]], storage_client: Storage):
        """Upload batch results to GCS asynchronously."""
        for i, img in enumerate(processed_batch[1]):
            img_id = processed_batch[2][i]
            await storage_client.upload("img-search-data", img_id, img.tobytes())
        qdrant.upload_collection("image-search",
                                 processed_batch[0],
                                 payload=[{"id": id_} for id_ in processed_batch[2]],
                                 ids=[uuid.uuid5(uuid.NAMESPACE_URL, id_) for id_ in processed_batch[2]])

    def process_batch(batch: List[tuple[Image.Image, str]]) -> tuple[np.ndarray, list[Image.Image], list[str]]:
        """
        Process a batch of images through ML model.
        Args:
            batch: List of images
        Returns:
            List of results for each image in batch
        """
        assert batch, "Empty batch"
        logger.info(f"Worker {worker_id} processing batch of {len(batch)} images")
        images, ids = zip(*batch)
        images, ids = list(images), list(ids)
        siglip_images = [siglip_preprocessor(img) for img in images]
        siglip_images = torch.stack(siglip_images).cuda()
        encoded_images = siglip_vis(siglip_images).detach().cpu().numpy()

        blip_images = blip_preprocessor(images, return_tensors="pt").pixel_values
        blip_images = [to_pil_image(img) for img in blip_images]
        return encoded_images, blip_images, ids

    async def async_worker_loop():
        """Async main loop that properly manages Storage client."""
        async with Storage(service_file=os.environ.get("GCP_JSON")) as storage_client:
            processed_count = 0
            batch_buffer: list[tuple[Image.Image, str]] = []

            while True:
                try:
                    job: Optional[Image] = work_queue.get(timeout=batch_timeout)

                    # Check for sentinel value
                    if job is None:
                        logger.info(f"Worker {worker_id} received shutdown signal")
                        # Process remaining batch before shutting down
                        if batch_buffer:
                            logger.info(f"Worker {worker_id} processing final batch of {len(batch_buffer)}")
                            results = process_batch(batch_buffer)
                            await upload_batch_results(results, storage_client)
                            processed_count += len(batch_buffer)
                        break

                    batch_buffer.append(job)
                except Empty:
                    # Timeout reached - process partial batch if we have items
                    if batch_buffer:
                        logger.info(f"Worker {worker_id} processing partial batch of {len(batch_buffer)}")
                        results = process_batch(batch_buffer)
                        await upload_batch_results(results, storage_client)
                        processed_count += len(batch_buffer)

                        batch_buffer.clear()
                    continue

                # Process batch when full
                if len(batch_buffer) >= batch_size:
                    results = process_batch(batch_buffer)
                    await upload_batch_results(results, storage_client)
                    processed_count += len(batch_buffer)
                    logger.info(f"Worker {worker_id} total processed: {processed_count}")

                    batch_buffer.clear()

            logger.info(f"Worker {worker_id} shutting down. Processed {processed_count} images")

    try:
        loop.run_until_complete(async_worker_loop())
        if status_queue:
            status_queue.put({"worker_id": worker_id, "status": "completed"})
    except Exception as e:
        logger.error(f"Worker {worker_id} event loop crashed: {e}", exc_info=True)
        if status_queue:
            status_queue.put({"worker_id": worker_id, "status": "crashed", "error": str(e)})
    finally:
        loop.close()
        logger.info(f"Worker {worker_id} process terminated")

async def main():
    NUM_WORKERS = 1

    config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
    subreddits = config["subreddits"]
    period = config["period"]

    # Create multiprocess queues
    work_queue = mp.Queue(maxsize=30)

    # Start ML processor workers
    workers = []
    status_queue = mp.Queue()  # New queue for worker statuses
    for i in range(NUM_WORKERS):
        worker = mp.Process(
            target=ml_processor_worker,
            args=(work_queue, i, 12, 5.0, status_queue)  # Pass status_queue to workers
        )
        worker.start()
        workers.append(worker)

    logger.info(f"Started {NUM_WORKERS} ML processor workers")

    # Initialize scraper
    scraper = HTTPImageScraper(5)

    # Fetch posts
    posts = await scraper.fetch_posts(
        subreddits=subreddits,
        time_filter=period,
    )

    # Download images and queue for processing
    queued_count = await scraper.download_and_queue(posts, work_queue)

    # Send sentinel values to workers
    for _ in range(NUM_WORKERS):
        work_queue.put(None)

    logger.info("Waiting for workers to complete...")

    # Wait for all workers to finish with health monitoring
    worker_crashed = False

    while any(worker.is_alive() for worker in workers):
        # Check for dead workers
        for i, worker in enumerate(workers):
            if not worker.is_alive() and worker.exitcode is not None and worker.exitcode != 0:
                logger.error(f"Worker {i} died with exit code {worker.exitcode}")

        # Check for worker statuses
        while not status_queue.empty():
            status = status_queue.get()
            logger.info(f"Worker {status['worker_id']} status: {status['status']}")
            if status['status'] == "crashed":
                logger.error(f"Worker {status['worker_id']} crashed: {status.get('error', '')}")
                worker_crashed = True
            elif status['status'] == "initialization_failed":
                logger.error(f"Worker {status['worker_id']} failed to initialize: {status.get('error', '')}")
                worker_crashed = True

        # If a worker crashed, terminate all workers
        if worker_crashed:
            logger.warning("Worker crash detected. Terminating all workers...")
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
                    if worker.is_alive():
                        worker.kill()
            break

        await asyncio.sleep(0.5)

    # Final join to ensure cleanup
    for worker in workers:
        worker.join(timeout=2)

    logger.info("Pipeline completed")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    asyncio.run(main())