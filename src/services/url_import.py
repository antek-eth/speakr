"""
URL import service for downloading audio/video via yt-dlp.

Includes URL validation with SSRF protection and safe error sanitization.
"""

import ipaddress
import os
import re
import logging
import socket
import subprocess
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Private/reserved IP ranges to block (SSRF protection)
_BLOCKED_HOSTS = {
    'localhost', '127.0.0.1', '::1', '0.0.0.0',
    'metadata.google.internal', '169.254.169.254',
}

# Known yt-dlp errors mapped to user-friendly messages
_ERROR_MAP = [
    ('Video unavailable', 'This video is unavailable or private'),
    ('is not available', 'This video is unavailable or private'),
    ('not made this video available in your country', 'This video is geo-restricted and not available from this server\'s location'),
    ('Private video', 'This video is private'),
    ('HTTP Error 429', 'Too many downloads, try again later'),
    ('HTTP Error 403', 'Access denied — video may be geo-restricted'),
    ('Unsupported URL', 'This URL is not supported'),
    ('is not a valid URL', 'Invalid URL format'),
    ('is a live event', 'Live streams cannot be imported'),
    ('is a playlist', 'Playlists are not supported — use a single video URL'),
    ('Sign in to confirm', 'YouTube is requiring sign-in from this server — try a different video or source'),
]


def _is_private_ip(ip_str):
    """Check if an IP address is private, loopback, link-local, or reserved."""
    try:
        addr = ipaddress.ip_address(ip_str)
        return (
            addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_reserved or addr.is_multicast
        )
    except ValueError:
        return False


def validate_import_url(url):
    """
    Validate a URL for import with SSRF protection including DNS resolution.

    Returns (is_valid, error_message).
    """
    if not url or not url.strip():
        return False, "URL is required"

    url = url.strip()

    # Must be http or https
    if not re.match(r'^https?://', url, re.IGNORECASE):
        return False, "URL must start with http:// or https://"

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"

    if not parsed.hostname:
        return False, "Invalid URL format"

    hostname = parsed.hostname.lower()

    # Block known internal/localhost hosts
    if hostname in _BLOCKED_HOSTS:
        return False, "Internal URLs are not allowed"

    # Basic structure check
    if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url, re.IGNORECASE):
        return False, "Invalid URL format"

    # DNS resolution check: resolve hostname and reject private/internal IPs
    try:
        addrinfos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in addrinfos:
            ip_str = sockaddr[0]
            if _is_private_ip(ip_str):
                logger.warning(f"SSRF blocked: {hostname} resolves to private IP {ip_str}")
                return False, "URL resolves to a private network address"
    except socket.gaierror:
        return False, "Could not resolve hostname"

    return True, None


def sanitize_ytdlp_error(stderr):
    """
    Map yt-dlp stderr to a user-friendly error message.
    Never returns raw stderr to the client.
    """
    if not stderr:
        return "Download failed"

    for pattern, message in _ERROR_MAP:
        if pattern.lower() in stderr.lower():
            return message

    # Generic fallback — don't leak internal paths or details
    logger.warning(f"Unmapped yt-dlp error: {stderr[:500]}")
    return "Download failed — the video may be unavailable or the URL unsupported"


def fetch_metadata(url, timeout=30):
    """
    Fetch video metadata via yt-dlp without downloading.

    Returns dict with {title, duration} or None on failure.
    """
    try:
        result = subprocess.run(
            ['yt-dlp', '--ignore-config', '--dump-json', '--no-download',
             '--no-playlist', '--no-warnings',
             '--remote-components', 'ejs:github', url],
            shell=False, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            import json
            meta = json.loads(result.stdout)
            return {
                'title': (meta.get('title') or '').strip(),
                'duration': meta.get('duration'),
            }
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"Metadata fetch failed for {url}: {e}")
    return None


def download_audio(url, output_path, timeout=120):
    """
    Download audio from URL via yt-dlp.

    Args:
        url: Source URL
        output_path: Output file path template (yt-dlp %(ext)s format)
        timeout: Download timeout in seconds

    Returns:
        (success: bool, error_message: str or None)
    """
    try:
        result = subprocess.run(
            ['yt-dlp', '--ignore-config', '-x', '--audio-format', 'mp3',
             '--audio-quality', '128K', '--no-playlist',
             '--max-filesize', '500m',  # Reject huge files
             '--remote-components', 'ejs:github',
             '-o', output_path, url],
            shell=False, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return False, sanitize_ytdlp_error(result.stderr)
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Download timed out (video may be too long)"


def cleanup_partial_download(pattern):
    """Remove partial download artifacts matching glob pattern."""
    import glob
    for f in glob.glob(pattern):
        try:
            os.unlink(f)
            logger.info(f"Cleaned up partial download: {f}")
        except OSError:
            pass
