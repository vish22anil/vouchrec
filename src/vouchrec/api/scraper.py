"""Live Amazon product details and customer reviews extractor using curl_cffi and BeautifulSoup."""

import logging
import re
from typing import Any

from bs4 import BeautifulSoup
from curl_cffi import requests

logger = logging.getLogger(__name__)

ASIN_PATTERN = re.compile(r"/(?:dp|gp/product|product-reviews|product|d)/([A-Z0-9]{10})(?:[/?&#]|$)", re.IGNORECASE)
RAW_ASIN_PATTERN = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(r"https?://(?:www\.)?(amazon\.[a-z\.]+)", re.IGNORECASE)


def extract_asin_and_domain(url_or_id: str) -> tuple[str | None, str]:
    """Extract 10-character ASIN and marketplace domain from an Amazon URL or identifier.

    Args:
        url_or_id: Raw URL or ASIN string.

    Returns:
        Tuple of (asin, domain). Domain defaults to 'amazon.in' if unspecified.
    """
    clean_input = url_or_id.strip()

    # Extract marketplace domain
    dom_match = DOMAIN_PATTERN.search(clean_input)
    domain = dom_match.group(1).lower() if dom_match else "amazon.in"

    # Match ASIN from URL pattern
    asin_match = ASIN_PATTERN.search(clean_input)
    if asin_match:
        return asin_match.group(1).upper(), domain

    # Direct ASIN string (e.g. 'B0C3HCD34R')
    if RAW_ASIN_PATTERN.match(clean_input):
        return clean_input.upper(), domain

    return None, domain


def fetch_live_amazon_product(url_or_id: str) -> dict[str, Any]:
    """Fetch live product details and customer reviews directly from Amazon.

    Uses browser-grade TLS fingerprint impersonation to bypass bot challenge walls.

    Args:
        url_or_id: Product page URL, review page URL, or raw ASIN.

    Returns:
        Dictionary containing extracted product metadata, reviews, and status flags.
    """
    asin, domain = extract_asin_and_domain(url_or_id)
    if not asin:
        return {
            "success": False,
            "status": "invalid_url",
            "error": "Could not identify an Amazon Standard Identification Number (ASIN) in the provided URL.",
            "is_captcha": False,
        }

    canonical_url = f"https://www.{domain}/dp/{asin}"
    logger.info("Fetching live Amazon listing: %s (ASIN: %s, domain: %s)", canonical_url, asin, domain)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-IN,en-GB,en-US,en;q=0.9",
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }

    impersonation_chain = ["chrome120", "safari15_5", "chrome119", "safari_ios_16_5", "edge101"]
    html_content: str | None = None
    hit_captcha = False

    for impersonation in impersonation_chain:
        try:
            resp = requests.get(
                canonical_url,
                impersonate=impersonation,
                headers=headers,
                timeout=14,
            )
            if resp.status_code == 200:
                body_lower = resp.text.lower()
                if "validatecaptcha" in body_lower or "/errors/validatecaptcha" in body_lower:
                    hit_captcha = True
                    logger.debug("Amazon served captcha on impersonation %s", impersonation)
                    continue

                if len(resp.text) > 8000:
                    html_content = resp.text
                    hit_captcha = False
                    break
        except Exception as exc:
            logger.debug("Request failed with impersonation %s: %s", impersonation, exc)
            continue

    if not html_content:
        if hit_captcha:
            return {
                "success": False,
                "status": "captcha_challenge",
                "asin": asin,
                "domain": domain,
                "error": (
                    "Amazon returned an anti-bot CAPTCHA verification challenge for this listing. "
                    "You can copy and paste review text directly into the text analyzer below."
                ),
                "is_captcha": True,
            }
        return {
            "success": False,
            "status": "fetch_failed",
            "asin": asin,
            "domain": domain,
            "error": f"Failed to retrieve Amazon product page for ASIN '{asin}'. The item may be unavailable or unlisted.",
            "is_captcha": False,
        }

    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Product title
    title_el = soup.find("span", id="productTitle")
    product_title = title_el.get_text(strip=True) if title_el else f"Amazon Product ({asin})"

    # 2. Product image
    img_el = soup.find("img", id="landingImage") or soup.find("img", {"data-a-image-name": "landingImage"})
    product_image = img_el.get("src") if img_el else None

    product_star_el = soup.select_one("[data-hook='rating-out-of-text']") or soup.select_one(".a-icon-alt")
    product_rating = None
    if product_star_el:
        rating_match = re.search(r"([0-9.]+)\s+out\s+of\s+5", product_star_el.get_text(strip=True))
        if rating_match:
            try:
                product_rating = float(rating_match.group(1))
            except ValueError:
                product_rating = None

    # 4. Total ratings count
    review_count_el = soup.find("span", id="acrCustomerReviewText")
    total_reviews_text = review_count_el.get_text(strip=True) if review_count_el else None

    # 5. Extract customer reviews
    review_cards = soup.find_all("div", {"data-hook": "review"})
    parsed_reviews: list[dict[str, Any]] = []

    for rev in review_cards:
        author_el = rev.select_one(".a-profile-name")
        rating_el = rev.select_one(".a-icon-alt")
        title_el = rev.select_one("[data-hook='review-title']") or rev.select_one("a.a-text-bold")
        text_box = (
            rev.select_one("[data-hook='review-body']")
            or rev.select_one("[data-hook='reviewTextContainer']")
            or rev.select_one("[data-hook='reviewText']")
        )
        date_el = rev.select_one("[data-hook='review-date']")
        verified_el = rev.select_one("[data-hook='avp-badge']")

        author = author_el.get_text(strip=True) if author_el else "Amazon Customer"
        rdate = date_el.get_text(strip=True) if date_el else ""
        verified = bool(verified_el)

        rating_val = None
        if rating_el:
            r_match = re.search(r"([0-9.]+)\s+out\s+of\s+5", rating_el.get_text(strip=True))
            if r_match:
                try:
                    rating_val = float(r_match.group(1))
                except ValueError:
                    rating_val = None

        rtitle = title_el.get_text(strip=True) if title_el else ""

        body = ""
        if text_box:
            for btn in text_box.find_all(class_=lambda c: c and "see-more" in c):
                btn.decompose()
            raw_text = text_box.get_text(separator=" ", strip=True)
            # Remove Amazon expander boilerplate
            raw_text = raw_text.replace("Brief content visible, double tap to read full content.", "")
            raw_text = raw_text.replace("Full content visible, double tap to read brief content.", "").strip()
            body = raw_text

        if body or rtitle:
            parsed_reviews.append({
                "author": author,
                "rating": rating_val,
                "title": rtitle,
                "date": rdate,
                "verified": verified,
                "body": body if body else rtitle,
            })

    return {
        "success": True,
        "status": "fetched",
        "asin": asin,
        "domain": domain,
        "canonical_url": canonical_url,
        "product_title": product_title,
        "product_image": product_image,
        "product_rating": product_rating,
        "total_reviews_text": total_reviews_text,
        "reviews_count": len(parsed_reviews),
        "reviews": parsed_reviews,
        "is_captcha": False,
    }
